"""
PerturbEnergy networks (scGen-style MLPs).

  z_b = q_phi_b(x, c)                 basal representation
  z_a = q_phi_a(z_b, a)               perturbation representation
  E_alpha(z_a, z_b, a) -> scalar      compatibility energy (LOW = compatible)
  x_hat = p_theta(z_b, z_a)           reconstructed log-norm expression

Two embedding tables exist in the whole model -- `cond_emb` (c) and `pert_emb` (a) --
owned by PerturbEnergy and SHARED by the encoders and the energy network, which only
hold the tensor ops. They belong to the VAE parameter group; the energy update sees
them detached, so the two optimisation phases never interfere.

forward():  z_b, then M candidate z_a, then the energy PICKS one (lowest energy, or
sampled with p propto exp(-E)), then the generator decodes the selected pair.
Priors for both latents are N(0, I) (no learned prior module).
"""
import torch
import torch.nn as nn


def mlp(in_dim, hidden, depth, dropout, norm="layer", out_dim=None):
    norm_layer = nn.BatchNorm1d if norm == "batch" else nn.LayerNorm
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), norm_layer(hidden), nn.SiLU(), nn.Dropout(dropout)]
        d = hidden
    if out_dim is not None:
        layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def reparameterize(mu, logvar):
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class GaussianHead(nn.Module):
    """MLP trunk -> (mu, logvar)."""
    def __init__(self, in_dim, out_dim, hidden, depth, dropout, norm):
        super().__init__()
        self.body = mlp(in_dim, hidden, depth, dropout, norm)
        self.fc_mu = nn.Linear(hidden, out_dim)
        self.fc_lv = nn.Linear(hidden, out_dim)

    def forward(self, h):
        h = self.body(h)
        return self.fc_mu(h), self.fc_lv(h).clamp(-8.0, 8.0)


class PerturbEnergy(nn.Module):
    def __init__(self, n_genes, n_cond, n_pert, mcfg):
        super().__init__()
        zb, za = mcfg["z_b_dim"], mcfg["z_a_dim"]
        h, d, p, nm = mcfg["hidden"], mcfg["depth"], mcfg["dropout"], mcfg["norm"]
        dc, da = mcfg["cond_emb_dim"], mcfg["pert_emb_dim"]
        self.z_b_dim, self.z_a_dim = zb, za
        self.n_candidates = mcfg["n_candidates"]
        self.select_mode = mcfg["select_mode"]
        self.select_tau = mcfg["select_tau"]

        # the model's ONLY two embedding tables, shared by every submodule
        self.cond_emb = nn.Embedding(n_cond, dc)
        self.pert_emb = nn.Embedding(n_pert, da)

        self.enc_b = GaussianHead(n_genes + dc, zb, h, d, p, nm)       # q(z_b | x, c)
        self.enc_a = GaussianHead(zb + da, za, h, d, p, nm)            # q(z_a | z_b, a)
        self.dec = mlp(zb + za, h, d, p, nm, out_dim=n_genes)          # p(x | z_b, z_a), linear out
        e = mcfg["energy"]
        self.energy_net = mlp(za + zb + da, e["hidden"], e["depth"], e["dropout"], nm, out_dim=1)

    # ---- parameter groups (disjoint: embeddings live with the VAE) ----------
    def vae_parameters(self):
        mods = (self.enc_b, self.enc_a, self.dec, self.cond_emb, self.pert_emb)
        return [p for m in mods for p in m.parameters()]

    def energy_parameters(self):
        return list(self.energy_net.parameters())

    # ---- building blocks ---------------------------------------------------
    def embed(self, c, a, detach=False):
        c_e, a_e = self.cond_emb(c), self.pert_emb(a)
        return (c_e.detach(), a_e.detach()) if detach else (c_e, a_e)

    def energy(self, z_a, z_b, a_e):
        """E_alpha(z_a, z_b, a) -> [N]. Accepts already-embedded a."""
        return self.energy_net(torch.cat([z_a, z_b, a_e], -1)).squeeze(-1)

    def decode(self, z_b, z_a):
        return self.dec(torch.cat([z_b, z_a], -1))

    def basal(self, x, c_e, sample=True):
        mu, lv = self.enc_b(torch.cat([x, c_e], -1))
        return (reparameterize(mu, lv) if sample else mu), mu, lv

    def pert(self, z_b, a_e, sample=True):
        mu, lv = self.enc_a(torch.cat([z_b, a_e], -1))
        return (reparameterize(mu, lv) if sample else mu), mu, lv

    def select_z_a(self, mu_a, lv_a, z_b, a_e, n=None, mode=None, tau=None):
        """Draw n candidates from q(z_a|z_b,a) and let the energy pick one.

        mode='min'     -> the lowest-energy candidate (deterministic)
        mode='softmax' -> sampled with p propto exp(-E/tau) (normalised over candidates)
        Returns (z_a [B,d], e_sel [B], e_all [B,n]).
        """
        n = n or self.n_candidates
        mode = mode or self.select_mode
        tau = tau or self.select_tau
        B = mu_a.shape[0]
        Z = torch.stack([reparameterize(mu_a, lv_a) for _ in range(n)], 1)        # [B,n,d]
        flat = Z.reshape(B * n, -1)
        zb_r = z_b.repeat_interleave(n, 0)
        ae_r = a_e.repeat_interleave(n, 0)
        E = self.energy(flat, zb_r, ae_r).view(B, n)                              # [B,n]
        if mode == "softmax":
            idx = torch.multinomial(torch.softmax(-E / tau, dim=1), 1).squeeze(1)
        else:
            idx = E.argmin(1)
        rows = torch.arange(B, device=Z.device)
        return Z[rows, idx], E[rows, idx], E

    # ---- full forward ------------------------------------------------------
    def forward(self, x, c, a, sample=True, n_candidates=None, mode=None, tau=None):
        c_e, a_e = self.embed(c, a)
        z_b, mu_b, lv_b = self.basal(x, c_e, sample)
        _, mu_a, lv_a = self.pert(z_b, a_e, sample=False)
        if sample:
            z_a, e_sel, e_all = self.select_z_a(mu_a, lv_a, z_b, a_e, n_candidates, mode, tau)
        else:                                   # deterministic path (eval/reconstruction)
            z_a, e_all = mu_a, None
            e_sel = self.energy(z_a, z_b, a_e)
        return dict(z_b=z_b, z_a=z_a, mu_b=mu_b, lv_b=lv_b, mu_a=mu_a, lv_a=lv_a,
                    a_e=a_e, c_e=c_e, e_sel=e_sel, e_all=e_all, x_hat=self.decode(z_b, z_a))
