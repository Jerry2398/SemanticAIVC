"""
Build the drug-structure Squidiff training data (v2) from the canonical treated
files + scPerturb controls + PRnet SMILES.

The drug-structure Squidiff encoder needs, per treated cell:
  * obs['SMILES'], obs['dose'], obs['Group']  (in the treated file)
  * a ROW-ALIGNED control cell's expression   (a separate control file)
scPerturb has no SMILES, so drug names are mapped to PRnet's SMILES (with
paren/salt-form normalization); cells whose drug can't be mapped OR whose SMILES
RDKit can't parse are dropped (exactly as the official prep does).

Reuses the canonical treated files (same cells / genes / normalization as every
other encoder), so the drug Squidiff is comparable up to the SMILES-droppable
subset. Cheap (controls are ~17k cells) -- runs fine without a big-mem node.

  HDF5_USE_FILE_LOCKING=FALSE python prep_squidiff_drug.py
"""
import argparse
import os
import re

import anndata as ad
import numpy as np
import scanpy as sc
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import config as C
from common import to_dense

CELL_LINES = ["A549", "MCF7", "K562"]
SALT_TOKENS = ["hcl", "2hcl", "hbr", "2hbr", "mesylate", "sodium", "citrate",
               "tosylate", "diphosphate", "acetonide", "salt", "trihydrate",
               "dihydrochloride", "hydrochloride", "maleate", "phosphate",
               "sulfate", "acetate", "fumarate", "besylate", "free", "base",
               "disodium", "potassium", "calcium", "monohydrate"]


def _norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _strip_salt(name):
    toks = [t for t in re.split(r"[\s]+", str(name)) if _norm(t) not in SALT_TOKENS]
    return " ".join(toks)


def build_smiles_lookup(prnet_path):
    p = sc.read_h5ad(prnet_path)
    lut = {}
    for n, s in zip(p.obs["product_name"].astype(str), p.obs["SMILES"].astype(str)):
        for key in (_norm(n), _norm(n.split("(")[0]), _norm(_strip_salt(n.split("(")[0]))):
            lut.setdefault(key, s)
    return lut


def map_smiles(name, lut):
    cands = [name, name.split("(")[0], _strip_salt(name.split("(")[0])]
    m = re.search(r"\(([^)]*)\)", name)
    if m:
        cands += [m.group(1), _strip_salt(m.group(1))]
    for c in cands:
        s = lut.get(_norm(c))
        if s and Chem.MolFromSmiles(s) is not None:
            return s
    return None


def load_controls(raw_path, canon_var_names, seed):
    """Control cells (perturbation=='control') for the 3 lines, log-normalized,
    restricted+reordered to the canonical gene panel. Returns dict cell_type -> X."""
    raw = sc.read_h5ad(raw_path, backed="r")
    o = raw.obs
    mask = (o["perturbation"].astype(str).values == "control") & \
           o["cell_line"].astype(str).isin(CELL_LINES).values
    idx = np.where(mask)[0]
    ctrl = raw[idx].to_memory()
    # reorder genes to the canonical panel
    gi = ctrl.var_names.get_indexer(canon_var_names)
    assert (gi >= 0).all(), "some canonical genes missing from raw control file"
    ctrl = ctrl[:, gi].copy()
    sc.pp.normalize_total(ctrl, target_sum=1e4)
    sc.pp.log1p(ctrl)
    X = to_dense(ctrl.X).astype(np.float32)
    ct = ctrl.obs["cell_line"].astype(str).values
    return {c: X[ct == c] for c in CELL_LINES}


def build_split(canon_path, ctrl_by_ct, lut, out_treated, out_control, seed):
    a = sc.read_h5ad(canon_path)
    smiles = np.array([map_smiles(d, lut) for d in a.obs[C.LABELS["drug"]].astype(str)])
    keep = np.array([s is not None for s in smiles])
    print(f"{os.path.basename(canon_path)}: {keep.sum()}/{a.n_obs} cells kept "
          f"({a.n_obs - keep.sum()} dropped, no SMILES)")
    a = a[keep].copy()
    a.obs["SMILES"] = smiles[keep].astype(str)
    a.obs["dose"] = a.obs[C.LABELS["dose"]].astype(float)

    # row-aligned control: sample a control of the same cell type per treated cell
    rng = np.random.default_rng(seed)
    ct = a.obs["cell_type"].astype(str).values
    cX = to_dense(a.X).astype(np.float32)          # keep same expr matrix layout
    ctrl_rows = np.zeros_like(cX)
    for c in CELL_LINES:
        rows = np.where(ct == c)[0]
        pool = ctrl_by_ct[c]
        pick = rng.integers(0, len(pool), size=len(rows))
        ctrl_rows[rows] = pool[pick]

    ctrl_ad = ad.AnnData(X=ctrl_rows, obs=a.obs[["cell_type", "Group"]].copy())
    ctrl_ad.var_names = a.var_names
    a.write_h5ad(out_treated)
    ctrl_ad.write_h5ad(out_control)
    print(f"  wrote treated {a.shape} -> {out_treated}")
    print(f"  wrote control {ctrl_ad.shape} -> {out_control}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=C.RAW_COUNTS_FILE)
    ap.add_argument("--prnet", default=C.PRNET_SMILES_FILE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lut = build_smiles_lookup(args.prnet)
    print("SMILES lookup keys:", len(lut))

    canon = sc.read_h5ad(C.TRAIN_DATA_PATH)
    ctrl_by_ct = load_controls(args.raw, canon.var_names, args.seed)
    print("control cells per type:", {k: len(v) for k, v in ctrl_by_ct.items()})

    build_split(C.TRAIN_DATA_PATH, ctrl_by_ct, lut,
                C.TRAIN_DRUG_PATH, C.TRAIN_DRUG_CONTROL_PATH, args.seed)
    build_split(C.VAL_DATA_PATH, ctrl_by_ct, lut,
                C.VAL_DRUG_PATH, C.VAL_DRUG_CONTROL_PATH, args.seed)


if __name__ == "__main__":
    main()
