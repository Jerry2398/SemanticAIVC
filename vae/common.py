"""Shared: raw-counts -> log-norm transform, and validation metrics."""
import numpy as np
import torch


def lognorm(counts, target_sum=1e4):
    """raw counts [B,G] -> log1p(normalize_total(target_sum)). Matches scGen input."""
    lib = counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.log1p(counts / lib * target_sum)


def _colwise_pearson(a, b, eps=1e-8):
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + eps)


@torch.no_grad()
def collect_recon(model, loader, dev, max_batches=20):
    """Encode(mu)->decode over up to max_batches val batches; return (X, Xhat) log-norm."""
    model.eval()
    xs, rs = [], []
    for i, (counts, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = lognorm(counts.to(dev))
        mu, _ = model.encode(x)
        xh = model.decode(mu)                       # deterministic reconstruction
        xs.append(x.cpu().numpy()); rs.append(xh.cpu().numpy())
    return np.concatenate(xs, 0), np.concatenate(rs, 0)


def reconstruction_metrics(X, Xhat):
    mse = float(((X - Xhat) ** 2).mean())
    gene_r = float(np.nanmean(_colwise_pearson(Xhat, X)))
    cell_r = float(np.nanmean(_colwise_pearson(Xhat.T, X.T)))
    ss_res = ((X - Xhat) ** 2).sum()
    ss_tot = ((X - X.mean()) ** 2).sum() + 1e-8
    r2 = float(1 - ss_res / ss_tot)
    return {"recon_mse": mse, "gene_pearson": gene_r, "cell_pearson": cell_r, "r2": r2}


@torch.no_grad()
def generative_mean_match(model, X_real_mean, dev, n=2000):
    """Sample z~N(0,I), decode, compare mean expression profile to real val mean."""
    model.eval()
    z = torch.randn(n, model.latent_dim, device=dev)
    gen = model.decode(z).cpu().numpy()
    r = float(np.corrcoef(gen.mean(0), X_real_mean)[0, 1])
    return {"gen_mean_pearson": r}
