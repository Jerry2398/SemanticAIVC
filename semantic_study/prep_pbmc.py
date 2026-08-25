"""
Prepare the Kang 2018 IFN-beta PBMC dataset (kang.h5ad) for the semantic study.

Input : kang.h5ad -- X = raw counts (24,673 x 15,706), obs has cell_type (8),
        label (ctrl/stim), replicate (8 patients).
Output: pbmc_hvg{N}_{train,val}.h5ad with
          .X               = log-normalized expression (N HVGs)
          .layers["counts"]= raw counts (N HVGs)
          .obs             = cell_type / condition(=label) / batch(=replicate) / Group

Run:  SEMANTIC_DATASET=pbmc HDF5_USE_FILE_LOCKING=FALSE python prep_pbmc.py
"""
import argparse
import os

import numpy as np
import scanpy as sc

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=C.RAW_COUNTS_FILE)
    ap.add_argument("--out_dir", default=C.COUNTS_DIR)
    ap.add_argument("--n_top_genes", type=int, default=C.GENE_SIZE)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hvg_flavor", default="seurat_v3", choices=["seurat_v3", "seurat"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    a = sc.read_h5ad(args.in_path)
    print("loaded:", a.shape)
    a.layers["counts"] = a.X.copy()   # X is raw counts

    # canonical labels
    a.obs["cell_type"] = a.obs["cell_type"].astype(str).astype("category")
    a.obs["condition"] = a.obs["label"].astype(str)        # ctrl / stim (binary perturbation)
    a.obs["batch"]     = a.obs["replicate"].astype(str)    # patient
    a.obs["Group"]     = a.obs["cell_type"].cat.codes
    print("cell types:", list(a.obs['cell_type'].cat.categories))
    print("condition :", list(a.obs['condition'].unique()))

    # HVG (seurat_v3 on counts if scikit-misc available; else seurat on log-norm)
    if args.hvg_flavor == "seurat_v3":
        try:
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=args.n_top_genes, layer="counts")
        except Exception as e:
            print(f"seurat_v3 unavailable ({e}); using seurat on log-norm")
            args.hvg_flavor = "seurat"
    # log-normalize X (keep counts layer intact)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    if args.hvg_flavor == "seurat":
        sc.pp.highly_variable_genes(a, n_top_genes=args.n_top_genes)
    a = a[:, a.var["highly_variable"]].copy()
    print("after HVG:", a.shape)

    # split
    is_val = np.random.default_rng(args.seed).random(a.n_obs) < args.val_frac
    tr, va = a[~is_val].copy(), a[is_val].copy()
    for x in (tr, va):
        x.obs_names_make_unique()
    trp = os.path.join(args.out_dir, f"pbmc_hvg{args.n_top_genes}_train.h5ad")
    vap = os.path.join(args.out_dir, f"pbmc_hvg{args.n_top_genes}_val.h5ad")
    tr.write_h5ad(trp); va.write_h5ad(vap)
    print(f"train {tr.shape} -> {trp}")
    print(f"val   {va.shape} -> {vap}")


if __name__ == "__main__":
    main()
