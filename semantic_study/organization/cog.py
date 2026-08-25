"""
COG -- Cell Ontology Geometry.

Accuracy only tells you whether cell types are separable; it cannot tell you
whether their *arrangement* in the latent is biologically sensible. COG asks
that: are cell types that are close in the Cell Ontology tree also close in the
latent?

Computation:
  1. latent centroid per cell type
  2. centroid distance matrix (cosine by default)
  3. ontology distance matrix (Cell Ontology is a tree -> pairwise tree distance)
  4. compare the two matrices with a Mantel test (higher = better)

The ontology distances are supplied via --ontology_csv: a square CSV whose row
and column headers are the cell-type names in `label_col`, holding pairwise
tree distances. Provide it for an ontology-structured dataset (e.g. PBMC).

sciplex has only 3 cancer cell LINES (A549 / MCF7 / K562), which do not form a
meaningful ontology tree, so by default (config.COG_SUPPORTED = False) this
metric is skipped with an explanatory note.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config as C
from common import load_embedding, save_results, pairwise_distances, mantel_test


def centroids(Z, labels):
    names = sorted(np.unique(labels).tolist())
    cent = np.stack([Z[labels == n].mean(0) for n in names])
    return names, cent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_emb", default=C.VAL_EMB_PATH)
    ap.add_argument("--label_col", default=C.LABELS["cell_type"])
    ap.add_argument("--ontology_csv", default=C.ONTOLOGY_CSV,
                    help="square CSV of pairwise ontology (tree) distances, indexed by cell-type name")
    ap.add_argument("--centroid_metric", default="cosine")
    ap.add_argument("--n_perm", type=int, default=999)
    ap.add_argument("--force", action="store_true", help="run even if COG_SUPPORTED is False")
    args = ap.parse_args()

    if not C.COG_SUPPORTED and args.ontology_csv is None and not args.force:
        result = {
            "metric": "COG",
            "status": "skipped",
            "reason": ("sciplex has only 3 cancer cell lines, which do not form a "
                       "meaningful Cell Ontology tree. Provide --ontology_csv (and/or "
                       "use an ontology-structured dataset like PBMC) to enable COG."),
        }
        save_results("cog", result)
        return

    Z, obs, _ = load_embedding(args.val_emb)
    labels = obs[args.label_col].astype(str).values
    names, cent = centroids(Z, labels)
    D_lat = pairwise_distances(cent, metric=args.centroid_metric)
    print(f"COG on {len(names)} cell types: {names}")

    if args.ontology_csv is None:
        result = {
            "metric": "COG", "status": "no_ontology",
            "cell_types": names,
            "latent_centroid_distance": D_lat.tolist(),
            "reason": "no --ontology_csv provided; emitted latent centroid distances only.",
        }
        save_results("cog", result)
        return

    onto = pd.read_csv(args.ontology_csv, index_col=0)
    onto = onto.loc[names, names]  # align order to the latent centroids
    D_onto = onto.values.astype(float)
    r, p = mantel_test(D_lat, D_onto, n_perm=args.n_perm)

    result = {
        "metric": "COG", "status": "ok",
        "cell_types": names,
        "centroid_metric": args.centroid_metric,
        "mantel_r": r, "mantel_p": p,
        "latent_centroid_distance": D_lat.tolist(),
    }
    save_results("cog", result)


if __name__ == "__main__":
    main()
