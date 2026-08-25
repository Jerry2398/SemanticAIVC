"""
Probe models + training loops shared by the Decodability metrics.

The PDF specifies a *Linear Probe* everywhere (a strong decoder would evaluate
the decoder, not the latent). We default to a single ``nn.Linear`` and expose an
optional 1-hidden-layer MLP for ablation. Probes are trained on the TRAIN-split
embeddings and evaluated on the VAL-split embeddings, per the study design.

Features are standardized using TRAIN statistics only (fit on train, applied to
val) to avoid leakage.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _make_probe(kind, in_dim, out_dim):
    if kind == "linear":
        return LinearProbe(in_dim, out_dim)
    if kind == "mlp":
        return MLPProbe(in_dim, out_dim)
    raise ValueError(f"unknown probe kind {kind!r}")


class _Standardizer:
    """Fit on train, apply to both splits. Avoids val->train leakage."""
    def __init__(self, X):
        self.mu = X.mean(0, keepdims=True)
        self.sd = X.std(0, keepdims=True) + 1e-6

    def __call__(self, X):
        return (X - self.mu) / self.sd


def _train_loop(model, loss_fn, Xtr, Ytr, *, epochs, lr, batch_size,
                weight_decay, val_frac=0.1, patience=10, seed=0, verbose=True):
    """Generic mini-batch trainer with early stopping on an internal val slice."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    va, tr = perm[:n_val], perm[n_val:]

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.as_tensor(Ytr)
    ds = TensorDataset(Xtr_t[tr], Ytr_t[tr])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    Xva = Xtr_t[va].to(DEVICE)
    Yva = Ytr_t[va].to(DEVICE)

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva), Yva).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, bad = vloss, {k: v.detach().cpu().clone()
                                                for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  internal-val loss {vloss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_classifier(Ztr, ytr, Zva, *, n_classes, kind="linear",
                     epochs=200, lr=1e-3, batch_size=4096, weight_decay=1e-4,
                     seed=0, verbose=True):
    """Train a classification probe on (Ztr, ytr); return val logits (numpy).

    ytr must be integer class ids in [0, n_classes).
    """
    std = _Standardizer(Ztr)
    Ztr_s, Zva_s = std(Ztr), std(Zva)
    model = _make_probe(kind, Ztr.shape[1], n_classes)
    model = _train_loop(model, nn.CrossEntropyLoss(),
                        Ztr_s, ytr.astype(np.int64),
                        epochs=epochs, lr=lr, batch_size=batch_size,
                        weight_decay=weight_decay, seed=seed, verbose=verbose)
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(Zva_s, dtype=torch.float32).to(DEVICE))
    return logits.cpu().numpy()


def train_regressor(Ztr, Ytr, Zva, *, kind="linear",
                    epochs=300, lr=1e-3, batch_size=4096, weight_decay=1e-4,
                    seed=0, verbose=True):
    """Train a regression probe (MSE) on (Ztr, Ytr); return val predictions.

    Ytr shape (n,) or (n, out_dim). Returns predictions with the same trailing
    shape. Targets are standardized (train stats) then de-standardized on output.
    """
    y2d = Ytr.reshape(len(Ytr), -1).astype(np.float32)
    ymu, ysd = y2d.mean(0, keepdims=True), y2d.std(0, keepdims=True) + 1e-6
    y2d_s = (y2d - ymu) / ysd

    std = _Standardizer(Ztr)
    Ztr_s, Zva_s = std(Ztr), std(Zva)
    model = _make_probe(kind, Ztr.shape[1], y2d.shape[1])
    model = _train_loop(model, nn.MSELoss(), Ztr_s, y2d_s,
                        epochs=epochs, lr=lr, batch_size=batch_size,
                        weight_decay=weight_decay, seed=seed, verbose=verbose)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.as_tensor(Zva_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    pred = pred_s * ysd + ymu  # shape (n_val, out_dim); rows = val cells, not train cells
    return pred.ravel() if Ytr.ndim == 1 else pred
