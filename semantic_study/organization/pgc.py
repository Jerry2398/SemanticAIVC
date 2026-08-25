"""
PGC -- Perturbation Geometry Consistency.

Drugs are not independent: they share mechanisms (Drug -> MOA -> Target ->
Pathway). A good latent should place drugs with the same MOA closer together
than drugs with different MOAs.

Computation:
  1. latent centroid per drug
  2. centroid distance matrix
  3. score how well the MOA labels cluster those centroids:
       * Silhouette score (drug centroids as points, MOA as cluster label)
       * Within/Between distance ratio (lower = tighter MOA grouping) + a
         permutation p-value (shuffle MOA labels).

Drugs whose MOA class has < 2 members are dropped for the silhouette (undefined
for singleton clusters) and counted.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import silhouette_score

import config as C
from common import load_embedding, save_results, pairwise_distances


def drug_centroids(Z, drug):
    names = sorted(np.unique(drug).tolist())
    cent = np.stack([Z[drug == d].mean(0) for d in names])
    return names, cent


def within_between_ratio(D, labels):
    """mean within-group / mean between-group pairwise distance."""
    iu = np.triu_indices(len(labels), k=1)
    same = labels[iu[0]] == labels[iu[1]]
    d = D[iu]
    within = d[same].mean() if same.any() else np.nan
    between = d[~same].mean() if (~same).any() else np.nan
    return float(within), float(between), float(within / between)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_emb", default=C.VAL_EMB_PATH)
    ap.add_argument("--drug_col", default=C.LABELS["drug"])
    ap.add_argument("--moa_col", default=C.LABELS["moa"])
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--n_perm", type=int, default=999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Z, obs, _ = load_embedding(args.val_emb)
    drug = obs[args.drug_col].astype(str).values
    names, cent = drug_centroids(Z, drug)

    # map each drug -> its MOA (take the modal MOA for the drug)
    moa_series = obs[args.moa_col].astype(str)
    drug_to_moa = (obs.assign(_d=drug, _m=moa_series.values)
                      .groupby("_d")["_m"].agg(lambda s: s.value_counts().index[0]))
    moa = np.array([drug_to_moa[d] for d in names])

    # drop MOA classes with a single drug (silhouette undefined for singletons)
    counts = {m: int((moa == m).sum()) for m in np.unique(moa)}
    keep = np.array([counts[m] >= 2 for m in moa])
    n_dropped = int((~keep).sum())
    cent_k, moa_k = cent[keep], moa[keep]
    print(f"PGC: {len(names)} drugs, {len(np.unique(moa))} MOA classes "
          f"({args.moa_col}); dropped {n_dropped} singleton-MOA drugs")

    D = pairwise_distances(cent_k, metric=args.metric)
    sil = float(silhouette_score(cent_k, moa_k, metric=args.metric))
    within, between, ratio = within_between_ratio(D, moa_k)

    # permutation p-value: how often does a random MOA labeling beat the observed ratio?
    rng = np.random.default_rng(args.seed)
    better = 0
    for _ in range(args.n_perm):
        _, _, r = within_between_ratio(D, rng.permutation(moa_k))
        if r <= ratio:
            better += 1
    p = (better + 1) / (args.n_perm + 1)

    result = {
        "metric": "PGC",
        "drug_col": args.drug_col,
        "moa_col": args.moa_col,
        "distance_metric": args.metric,
        "n_drugs": int(len(names)),
        "n_drugs_used": int(keep.sum()),
        "n_moa_classes": int(len(np.unique(moa_k))),
        "n_dropped_singleton_moa": n_dropped,
        "silhouette": sil,
        "within_moa_dist": within,
        "between_moa_dist": between,
        "within_between_ratio": ratio,
        "ratio_perm_p": float(p),
    }
    save_results("pgc", result)


if __name__ == "__main__":
    main()
