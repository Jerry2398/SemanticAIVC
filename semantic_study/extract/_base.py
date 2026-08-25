"""
Reusable helpers for encoder extractors.

Every model-specific extractor (scVI, scGen, trVAE, scDisInFact, a foundation
model, ...) does the same three things:
  1. get the canonical sciplex cells for a split (train / val),
  2. run its own encoder to produce a latent matrix (n_cells, latent_dim),
  3. call write_embedding(name, split, latent) to persist it.

write_embedding attaches the SAME obs labels and the SAME X_expr (200-HVG
log-norm expression) that Squidiff used, so GER / EGC / LNP compare every
encoder against an identical reference and the metrics stay untouched. The only
requirement on an extractor is that its latent rows are in the canonical order
of the split file (i.e. it encoded load_split(split) in order, or a
re-ordered-to-match view of the same cells).

NOTE on counts: this sciplex file has NO raw counts (X is log-normalized).
reconstruct_counts() gives an APPROXIMATE count matrix by inverting
normalize_total(1e4)+log1p using obs['n_counts']; use it only for count-based
models (scVI-NB, scDisInFact, Geneformer) and treat results as approximate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anndata as ad
import numpy as np
import scanpy as sc

import config as C
from common import to_dense


def load_split(split):
    """Canonical treated cells for a split: (200-HVG, log-norm) AnnData."""
    path = C.TRAIN_DATA_PATH if split == "train" else C.VAL_DATA_PATH
    return sc.read_h5ad(path)


def reconstruct_counts(adata, target_sum=1e4):
    """APPROXIMATE raw counts from log-normalized X, inverting
    normalize_total(target_sum) + log1p via obs['n_counts'] as the library size.
    Only the retained genes are reconstructed (library size is the original
    all-gene total, which scVI-style size-factor inference tolerates)."""
    X = to_dense(adata.X)
    if "n_counts" in adata.obs:
        lib = adata.obs["n_counts"].values.astype(float)[:, None]
    else:  # fall back to per-cell sum of expm1 (rough)
        lib = np.expm1(X).sum(1, keepdims=True)
    counts = np.expm1(X) * lib / target_sum
    return np.rint(np.clip(counts, 0, None)).astype(np.float32)


def write_embedding(name, split, latent, obs=None, expr=None):
    """Persist a latent matrix to the canonical embedding path for `name`.

    latent : (n_cells, latent_dim) array, rows in canonical split order.
    obs/expr default to the canonical split's labels / log-norm X. If BOTH are
    passed explicitly (e.g. a SMILES-dropped subset for drug-Squidiff), they are
    used as-is and the canonical size check is skipped.
    """
    latent = np.asarray(latent, dtype=np.float32)
    if obs is None or expr is None:
        src = load_split(split)
        assert latent.shape[0] == src.n_obs, (
            f"latent has {latent.shape[0]} rows but split '{split}' has {src.n_obs} "
            "cells -- extractor must encode the canonical cells in order.")
        if obs is None:
            obs = src.obs.copy()
        if expr is None:
            expr = to_dense(src.X).astype(np.float32)
    assert latent.shape[0] == len(obs) == expr.shape[0], "latent/obs/expr row mismatch"

    out = ad.AnnData(X=latent, obs=obs)
    out.var_names = [f"z{i}" for i in range(latent.shape[1])]
    out.obsm["X_expr"] = expr
    out.uns["embedding_source"] = name

    tr, va = C.emb_paths(name)
    path = tr if split == "train" else va
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.write_h5ad(path)
    print(f"[{name}] wrote {split}: {path}  latent={latent.shape}")
    return path
