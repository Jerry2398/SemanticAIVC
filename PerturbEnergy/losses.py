"""Losses for the Energy-ELBO (paper Eq. 8/12/16)."""
import torch
import torch.nn.functional as F


def kl_standard(mu, logvar):
    """KL( N(mu, exp(logvar)) || N(0, I) ), summed over dims, averaged over the batch."""
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()


def mse_recon(x, x_hat):
    """Per-cell SSE (summed over genes), averaged over the batch -- as in our Tahoe VAEs."""
    return ((x_hat - x) ** 2).sum(1).mean()


def mmd(x, y, scales=(1.0, 2.0, 5.0, 10.0, 20.0)):
    """Multi-scale RBF MMD^2 with a median-heuristic bandwidth."""
    xy = torch.cat([x, y], 0)
    d2 = torch.cdist(xy, xy).pow(2)
    med = d2.detach().median().clamp(min=1e-6)
    k = sum(torch.exp(-d2 / (s * med)) for s in scales)
    n = x.shape[0]
    return k[:n, :n].mean() + k[n:, n:].mean() - 2.0 * k[:n, n:].mean()


def recon_loss(x, x_hat, kind="mse+mmd", mmd_weight=10.0, scales=(1.0, 2.0, 5.0, 10.0, 20.0)):
    """Returns (total, mse, mmd) -- components are reported for monitoring."""
    zero = torch.zeros((), device=x.device)
    m = mse_recon(x, x_hat) if "mse" in kind else zero
    d = mmd(x_hat, x, scales) if "mmd" in kind else zero
    return m + mmd_weight * d, m.detach(), d.detach()


def energy_objective(e_pos, e_neg, margin=1.0, use_margin=True, l2_reg=0.001):
    """Train E_alpha to score data-inferred z_a LOW and Langevin negatives HIGH (Eq.12).

    use_margin -> hinge  max(0, margin + E(z+) - E(z-)):  stops pushing once the two are
    separated by `margin`, which prevents the runaway energy scale a plain difference
    loss produces. l2_reg keeps the absolute energies near 0.
    """
    loss = (F.relu(margin + e_pos - e_neg).mean() if use_margin
            else e_pos.mean() - e_neg.mean())
    if l2_reg:
        loss = loss + l2_reg * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
    return loss


def contrastive_align(zb_p, zb_c, mask, cell_ids=None, mode="pair"):
    """PerturbedVAE-style invariance of z_b across perturbed / control cells (Eq.4):
    L = || z_b(x) - z_b(x^(u0)) ||^2.

    mode='pair'     literal per-sample form on context-matched partners. Cells are not
                    truly paired, so in expectation this also penalises the genuine
                    within-condition spread of z_b (it equals ||dmu||^2 + tr(S_p) + tr(S_c)).
    mode='centroid' per cell line, || mean z_b(perturbed) - mean z_b(control) ||^2:
                    removes only the systematic shift, keeps within-condition diversity.
    Returns (loss, shift) with `shift` = mean ||delta mu|| for monitoring.
    """
    zero = zb_p.sum() * 0.0
    if mask.sum() == 0:
        return zero, zero
    if mode == "pair":
        d = zb_p[mask] - zb_c[mask]
        return d.pow(2).sum(1).mean(), d.mean(0).norm().detach()
    tot, shifts = zero, []
    for c in cell_ids[mask].unique():
        m = mask & (cell_ids == c)
        if m.sum() < 2:
            continue
        d = zb_p[m].mean(0) - zb_c[m].mean(0)
        tot = tot + d.pow(2).sum()
        shifts.append(d.norm().detach())
    if not shifts:
        return zero, zero
    return tot / len(shifts), torch.stack(shifts).mean()
