"""
Shared utilities for the semantic study.

Embedding file format (what extract/ writes and every metric reads)
-------------------------------------------------------------------
An AnnData .h5ad where:
  * .X                  = the encoder embedding, shape (n_cells, latent_dim), float32
  * .obsm["X_expr"]     = the cell's (log-normalized) gene expression, (n_cells, n_genes)
  * .obs                = all metadata / labels (cell_type, condition, dose, pathway, ...)
  * .uns["embedding_source"] = a string tag naming the encoder (bookkeeping)

Keeping expression inside obsm makes the file self-contained: geometry metrics
(EGC, LNP) that compare latent-space vs expression-space geometry need no extra
file, and the whole study stays encoder-agnostic.
"""
import json
import os

import numpy as np
import scanpy as sc


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def load_embedding(path, need_expr=False):
    """Load an embedding .h5ad. Returns (Z, obs, expr).

    Z    : (n, latent_dim) float32 embedding matrix
    obs  : pandas DataFrame of labels
    expr : (n, n_genes) expression matrix if present (or need_expr), else None
    """
    ad = sc.read_h5ad(path)
    Z = to_dense(ad.X).astype(np.float32)
    expr = None
    if "X_expr" in ad.obsm:
        expr = np.asarray(ad.obsm["X_expr"]).astype(np.float32)
    elif need_expr:
        raise KeyError(f"{path} has no obsm['X_expr']; re-run extraction with expression stored.")
    return Z, ad.obs.copy(), expr


def save_results(name, payload, results_dir=None):
    """Write a metric result dict to results/[<encoder>/]<name>.json and echo it.

    If env var SEMANTIC_ENCODER is set (run_all sets it), results are namespaced
    under results/<encoder>/ so different encoders don't overwrite each other.
    """
    from config import RESULTS_DIR
    if results_dir is None:
        enc = os.environ.get("SEMANTIC_ENCODER")
        results_dir = os.path.join(RESULTS_DIR, enc) if enc else RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"{name}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\n[{name}] result written to {out}")
    print(json.dumps(payload, indent=2, default=_json_default))
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# Correlation / regression metric helpers
# --------------------------------------------------------------------------- #
def columnwise_pearson(pred, true, eps=1e-8):
    """Pearson r for each column, returns array of length n_cols (nan-safe)."""
    pred = pred - pred.mean(0, keepdims=True)
    true = true - true.mean(0, keepdims=True)
    num = (pred * true).sum(0)
    den = np.sqrt((pred ** 2).sum(0) * (true ** 2).sum(0)) + eps
    r = num / den
    return r


def rowwise_pearson(pred, true, eps=1e-8):
    """Pearson r for each row (cell). Returns array of length n_rows."""
    return columnwise_pearson(pred.T, true.T, eps)


def columnwise_r2(pred, true, min_var=1e-6):
    """Coefficient of determination per column (gene).

    Columns whose target variance is below `min_var` are returned as NaN rather
    than dividing by a near-zero denominator (which otherwise produces spurious
    huge-negative R2 that dominate the mean). Combine with nanmean().
    """
    n = true.shape[0]
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0, keepdims=True)) ** 2).sum(0)
    var = ss_tot / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where(var < min_var, np.nan, r2)
    return r2


def rowwise_r2(pred, true, min_var=1e-6):
    return columnwise_r2(pred.T, true.T, min_var)


def nanmean(x):
    return float(np.nanmean(x))


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def pairwise_distances(X, metric="euclidean"):
    """Condensed-free full (n,n) distance matrix for the given metric."""
    from scipy.spatial.distance import pdist, squareform
    if metric == "pearson":
        # 1 - Pearson correlation between rows
        Xc = X - X.mean(1, keepdims=True)
        norm = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
        Xn = Xc / norm
        corr = Xn @ Xn.T
        return 1.0 - corr
    if metric == "cosine":
        # eps-stabilized cosine so zero-norm rows give 0 (not NaN, as scipy pdist does)
        norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norm
        return 1.0 - Xn @ Xn.T
    return squareform(pdist(X, metric=metric))


def mantel_test(D1, D2, n_perm=999, rng=None):
    """Mantel test between two square distance matrices.

    Returns (r, p_value) where r is the Pearson correlation of the off-diagonal
    upper-triangular entries and p is a one-sided permutation p-value (H1: r>0).
    """
    rng = rng or np.random.default_rng(0)
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    a = D1[iu]
    b = D2[iu]
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    r_obs = float((a * b).mean())
    if n_perm <= 0:
        return r_obs, float("nan")
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Dp = D2[np.ix_(perm, perm)]
        bp = Dp[iu]
        bp = (bp - bp.mean()) / (bp.std() + 1e-12)
        if (a * bp).mean() >= r_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return r_obs, float(p)


def distance_correlation(X, Y):
    """Distance correlation between two point sets X (n,p) and Y (n,q).

    Works on the DISTANCE matrices, so pass the raw feature matrices. Returns a
    value in [0, 1]; 0 iff X and Y are independent.
    """
    from scipy.spatial.distance import pdist, squareform

    def _centered(M):
        A = squareform(pdist(M))
        return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()

    A = _centered(X)
    B = _centered(Y)
    n = X.shape[0]
    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)
    denom = np.sqrt(dcov2_xx * dcov2_yy)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2_xy, 0.0)) / np.sqrt(denom))


def subsample_idx(n, k, seed=0):
    """Deterministic subsample of k indices out of n (all if k>=n)."""
    if k <= 0 or k >= n:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, k, replace=False)
