"""
Shared helpers for Tahoe-100M (streaming, HVG-based).

Tahoe stores each cell as sparse (genes=token_ids into a 62,710-gene vocab,
expressions=raw counts). token_ids 0-2 are special tokens (the first entry per
cell is a CLS-like token with a -2 sentinel); real genes have token_id>=3 (see
gene_metadata). We reconstruct a CSR over the full vocab by using the parquet
list offsets directly (fast, no per-cell loop); the special entries land in
unused columns (<3) and negative sentinels are clipped to 0.

All data lives on /scratch (never the project dir).
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

TAHOE_DIR = os.environ.get("TAHOE_DIR", "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe")
DATA_DIR  = os.path.join(TAHOE_DIR, "data")
META_DIR  = os.path.join(TAHOE_DIR, "metadata")
HVG_JSON  = os.path.join(TAHOE_DIR, "hvg5000.json")

# obs columns kept as conditions/labels (strings)
OBS_COLS = ["drug", "cell_line_id", "moa-fine", "canonical_smiles", "plate", "sample"]


def gene_metadata():
    return pq.read_table(os.path.join(META_DIR, "gene_metadata.parquet")).to_pandas()


def n_vocab(gm=None):
    gm = gm if gm is not None else gene_metadata()
    return int(gm.token_id.max()) + 1


def list_shards(split="all", val_every=20):
    """Shard file paths. Deterministic shard-level split: every `val_every`-th
    shard is val, the rest train (keeps whole shards intact for streaming)."""
    shards = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    paths = [os.path.join(DATA_DIR, f) for f in shards]
    if split == "all":
        return paths
    val = [p for i, p in enumerate(paths) if i % val_every == 0]
    tr = [p for i, p in enumerate(paths) if i % val_every != 0]
    return val if split == "val" else tr


def shard_to_csr(path, n_genes, want_obs=True):
    """Vectorized: parquet shard -> (X csr [ncell, n_genes] raw counts, obs df|None)."""
    cols = ["genes", "expressions"] + (OBS_COLS if want_obs else [])
    tbl = pq.read_table(path, columns=cols)
    ga = tbl["genes"].combine_chunks()
    ea = tbl["expressions"].combine_chunks()
    offs = np.asarray(ga.offsets)
    gflat = np.asarray(ga.values)
    eflat = np.clip(np.asarray(ea.values).astype(np.float32), 0, None)  # drop -2 sentinels
    ncell = len(offs) - 1
    X = sp.csr_matrix((eflat, gflat, offs), shape=(ncell, n_genes))
    obs = tbl.select([c for c in OBS_COLS if c in tbl.column_names]).to_pandas() if want_obs else None
    return X, obs


def save_hvg(token_ids, gm):
    gm = gm.set_index("token_id")
    obj = {"token_ids": [int(t) for t in token_ids],
           "gene_symbol": [str(gm.loc[t, "gene_symbol"]) for t in token_ids],
           "ensembl_id": [str(gm.loc[t, "ensembl_id"]) for t in token_ids]}
    json.dump(obj, open(HVG_JSON, "w"))
    return HVG_JSON


def load_hvg():
    d = json.load(open(HVG_JSON))
    d["token_ids"] = np.asarray(d["token_ids"], dtype=np.int64)
    return d
