"""
Build the canonical sci-Plex-3 dataset (v2) from REAL raw counts.

Source : scPerturb SrivatsanTrapnell2020_sciplex3.h5ad (raw int counts, ~799k
         cells x ~111k genes, gene symbols in var_names).
Output : train / val .h5ad where
           .X               = log-normalized expression (5000 HVG)
           .layers["counts"]= raw integer counts        (5000 HVG)
           .obs             = canonical labels (cell_type/condition/dose/pathway/...)
so scVI/scDisInFact read layers['counts'], scGen/trVAE/Squidiff read X, and every
metric reads the same X_expr + labels.

Treated cells only by default (matches the v1 study; ~745k cells across the 3
lines); pass --keep_control to also include vehicle cells.

Memory: two-pass to stay tractable -- HVGs are chosen on a cell SUBSAMPLE (all
genes), then only those genes are loaded for all cells. Peak ~15-20 GB.

seurat_v3 HVG (recommended for scVI) needs scikit-misc:  pip install scikit-misc
Otherwise the default 'seurat' flavor (on log-norm) is used.

  HDF5_USE_FILE_LOCKING=FALSE python prep_sciplex_counts.py --n_top_genes 5000 --val_frac 0.25
"""
import argparse
import os

import numpy as np
import scanpy as sc

import config as C

CELL_LINES = ["A549", "MCF7", "K562"]


def keep_mask(obs, keep_control):
    cl = obs["cell_line"].astype(str)
    dose = obs["dose_value"].astype(float)
    pert = obs["perturbation"].astype(str)
    is_line = cl.isin(CELL_LINES).values
    is_treated = (~pert.isin(["control", "nan"]).values) & (dose.values > 0) & np.isfinite(dose.values)
    is_control = (pert.values == "control")
    return is_line & (is_treated | (is_control if keep_control else False))


def choose_hvg(adata_backed, cell_idx, n_top, flavor, subsample, seed):
    """Pick HVGs on a cell subsample (all genes) -> boolean gene mask."""
    n = min(subsample, len(cell_idx))
    sub = np.random.default_rng(seed).choice(cell_idx, n, replace=False)
    sub.sort()
    a = adata_backed[sub].to_memory()
    a.layers["counts"] = a.X.copy()
    if flavor == "seurat_v3":
        try:
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_top, layer="counts")
            return a.var["highly_variable"].values
        except Exception as e:
            print(f"seurat_v3 unavailable ({e}); using 'seurat' on log-norm")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=n_top)
    return a.var["highly_variable"].values


def add_labels(adata):
    o = adata.obs
    adata.obs["cell_type"] = o["cell_line"].astype(str).astype("category")
    adata.obs["condition"] = o["perturbation"].astype(str)
    adata.obs["dose"]      = o["dose_value"].astype(float)
    adata.obs["pathway"]   = o["pathway"].astype(str)
    for c in ["pathway_level_1", "pathway_level_2", "target", "replicate",
              "plate", "well", "time", "ncounts", "dose_unit"]:
        if c in o.columns:
            adata.obs[c] = o[c]
    adata.obs["Group"] = adata.obs["cell_type"].cat.codes


def split_and_write(adata, out_dir, tag, val_frac, seed):
    adata.layers["counts"] = adata.X.copy()          # raw counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)                               # X -> log-norm
    is_val = np.random.default_rng(seed).random(adata.n_obs) < val_frac
    tr, va = adata[~is_val].copy(), adata[is_val].copy()
    for a in (tr, va):
        a.obs_names_make_unique()
    tr_path = os.path.join(out_dir, f"sciplex3_{tag}_train.h5ad")
    va_path = os.path.join(out_dir, f"sciplex3_{tag}_val.h5ad")
    tr.write_h5ad(tr_path); va.write_h5ad(va_path)
    print(f"[{tag}] train {tr.shape} -> {tr_path}")
    print(f"[{tag}] val   {va.shape} -> {va_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=C.RAW_COUNTS_FILE)
    ap.add_argument("--out_dir", default=C.COUNTS_DIR)
    ap.add_argument("--n_top_genes", type=int, default=C.GENE_SIZE)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_control", action="store_true")
    ap.add_argument("--hvg_subsample", type=int, default=80000)
    ap.add_argument("--hvg_flavor", default="seurat_v3", choices=["seurat_v3", "seurat"])
    ap.add_argument("--emit_full", action="store_true",
                    help="also write a full-gene version for FMs (memory-heavy; big-mem node)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("reading (backed):", args.in_path)
    backed = sc.read_h5ad(args.in_path, backed="r")
    cell_idx = np.where(keep_mask(backed.obs, args.keep_control))[0]
    print(f"kept {len(cell_idx)} cells of {backed.n_obs}")

    print("selecting HVGs on a subsample...")
    gene_mask = choose_hvg(backed, cell_idx, args.n_top_genes, args.hvg_flavor,
                           args.hvg_subsample, args.seed)
    print(f"selected {int(gene_mask.sum())} HVGs")

    # load only HVG genes for all kept cells (column subset first keeps memory low)
    print("loading HVG submatrix for all kept cells...")
    adata = backed[:, gene_mask].to_memory()[cell_idx].copy()
    add_labels(adata)
    split_and_write(adata, args.out_dir, f"hvg{args.n_top_genes}", args.val_frac, args.seed)

    if args.emit_full:
        print("loading FULL-gene matrix for all kept cells (memory-heavy)...")
        full = backed[cell_idx].to_memory()
        add_labels(full)
        split_and_write(full, args.out_dir, "full", args.val_frac, args.seed)


if __name__ == "__main__":
    main()
