"""
EGC -- Expression Geometry Consistency.

Does the latent preserve the global geometry of expression space? For a
subsample of VAL cells, build a pairwise distance matrix in expression space
and in latent space, then measure their agreement with:
  * Mantel correlation (Pearson r of the off-diagonal distances + permutation p)
  * Distance correlation (dcor)

Reported for several expression distance metrics (pearson / cosine / euclidean);
latent distance uses the matching metric. No probe training -- this is a pure
geometry comparison, so it runs directly on val embeddings.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
from common import (load_embedding, save_results, subsample_idx,
                    pairwise_distances, mantel_test, distance_correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_emb", default=C.VAL_EMB_PATH)
    ap.add_argument("--n_cells", type=int, default=3000, help="subsample size for the n x n matrices")
    ap.add_argument("--n_perm", type=int, default=999)
    ap.add_argument("--metrics", nargs="+", default=["pearson", "cosine", "euclidean"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Z, _, X = load_embedding(args.val_emb, need_expr=True)
    idx = subsample_idx(Z.shape[0], args.n_cells, seed=args.seed)
    Z, X = Z[idx], X[idx]
    print(f"EGC on {len(idx)} cells | expr={X.shape} latent={Z.shape}")

    rng = np.random.default_rng(args.seed)
    per_metric = {}
    for m in args.metrics:
        D_expr = pairwise_distances(X, metric=m)
        D_lat = pairwise_distances(Z, metric=m)
        r, p = mantel_test(D_expr, D_lat, n_perm=args.n_perm, rng=rng)
        per_metric[m] = {"mantel_r": r, "mantel_p": p}
        print(f"  [{m:9s}] mantel_r={r:.4f}  p={p:.4f}")

    # distance correlation is metric-free (uses euclidean distances internally)
    dcor = distance_correlation(X, Z)
    print(f"  distance_correlation = {dcor:.4f}")

    result = {
        "metric": "EGC",
        "n_cells": int(len(idx)),
        "n_perm": args.n_perm,
        "mantel": per_metric,
        "distance_correlation": dcor,
    }
    save_results("egc", result)


if __name__ == "__main__":
    main()
