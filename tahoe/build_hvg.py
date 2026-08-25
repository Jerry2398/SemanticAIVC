"""
Select the 5000 HVG for Tahoe-100M from a SUBSAMPLE of shards (HVG on 100M cells
directly is unnecessary; a representative subsample gives a stable gene set).

Reconstructs counts from a set of shards, subsamples cells, runs scanpy HVG on
log-normalized data, and saves the chosen genes (token_ids + symbols + ensembl)
to tahoe/hvg5000.json. The streaming Dataset then always subsets to these genes.

  python build_hvg.py --n_shards 40 --cells_per_shard 5000 --n_top 5000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

import tahoe_common as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shards", type=int, default=40, help="how many shards to sample HVG from")
    ap.add_argument("--cells_per_shard", type=int, default=5000)
    ap.add_argument("--n_top", type=int, default=5000)
    ap.add_argument("--flavor", default="seurat_v3", choices=["seurat_v3", "seurat"])
    ap.add_argument("--split", default="train", help="pick HVG from this split (train recommended)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gm = T.gene_metadata()
    n_genes = T.n_vocab(gm)
    shards = T.list_shards(args.split)         # pick HVG from train shards (no val leakage)
    rng = np.random.default_rng(args.seed)
    pick = shards if len(shards) <= args.n_shards else \
        [shards[i] for i in np.sort(rng.choice(len(shards), args.n_shards, replace=False))]
    print(f"HVG from {len(pick)} shards x ~{args.cells_per_shard} cells, vocab={n_genes}")

    blocks = []
    for p in pick:
        X, _ = T.shard_to_csr(p, n_genes, want_obs=False)
        n = X.shape[0]
        idx = np.sort(rng.choice(n, min(args.cells_per_shard, n), replace=False))
        blocks.append(sp.csr_matrix(X[idx]))
    Xs = (blocks[0] if len(blocks) == 1 else sp.vstack(blocks, format="csr")).tocsr()
    print("subsample matrix:", Xs.shape)

    a = ad.AnnData(X=Xs)
    a.var["token_id"] = np.arange(n_genes)
    a.layers["counts"] = a.X.copy()
    if args.flavor == "seurat_v3":
        try:
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=args.n_top, layer="counts")
        except Exception as e:
            print("seurat_v3 unavailable:", e, "-> seurat"); args.flavor = "seurat"
    if args.flavor == "seurat":
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=args.n_top)

    hv_tokens = np.sort(a.var["token_id"].values[a.var["highly_variable"].values])
    # keep only real genes (token_id present in gene_metadata; excludes special 0-2)
    valid = set(gm.token_id.tolist())
    hv_tokens = np.array([t for t in hv_tokens if int(t) in valid], dtype=np.int64)
    path = T.save_hvg(hv_tokens, gm)
    print(f"selected {len(hv_tokens)} HVG -> {path}")
    print("sample:", T.load_hvg()["gene_symbol"][:10])


if __name__ == "__main__":
    main()
