"""
LNP -- Local Neighborhood Preservation.

Classic manifold-learning criterion: for each cell, take its k nearest
neighbors in expression space and in latent space, and measure the overlap with
the Jaccard index (averaged over cells). Reported for several k.

A random embedding scores ~ k/n (near 0); a perfect local map scores 1.
Runs on a subsample of VAL cells; neighbors are computed within that subsample.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.neighbors import NearestNeighbors

import config as C
from common import load_embedding, save_results, subsample_idx


def knn_indices(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)  # +1: first neighbor is self
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    return idx


def mean_jaccard(idx_a, idx_b):
    js = []
    for a, b in zip(idx_a, idx_b):
        sa, sb = set(a.tolist()), set(b.tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        js.append(inter / union if union else 0.0)
    return float(np.mean(js))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_emb", default=C.VAL_EMB_PATH)
    ap.add_argument("--n_cells", type=int, default=5000)
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 30, 50, 100])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Z, _, X = load_embedding(args.val_emb, need_expr=True)
    idx = subsample_idx(Z.shape[0], args.n_cells, seed=args.seed)
    Z, X = Z[idx], X[idx]
    n = len(idx)
    print(f"LNP on {n} cells")

    per_k = {}
    for k in args.ks:
        if k >= n:
            continue
        j = mean_jaccard(knn_indices(X, k), knn_indices(Z, k))
        per_k[str(k)] = {"jaccard": j, "chance": k / (n - 1)}
        print(f"  k={k:4d}  jaccard={j:.4f}  (chance {k/(n-1):.4f})")

    result = {
        "metric": "LNP",
        "n_cells": n,
        "jaccard_by_k": per_k,
        "mean_jaccard": float(np.mean([v["jaccard"] for v in per_k.values()])) if per_k else float("nan"),
    }
    save_results("lnp", result)


if __name__ == "__main__":
    main()
