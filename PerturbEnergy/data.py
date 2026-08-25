"""Data plumbing: Tahoe stream -> (x log-norm, c id, a id), plus vocabs.

Uses the SAME 5000-HVG panel, log-norm transform and shard-level train/val split as
every previously trained Tahoe model, so all metrics remain comparable.
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tahoe"))

import numpy as np
import torch

import tahoe_common as T
from tahoe_dataset import TahoeStream

DRUG_VOCAB = os.path.join(T.TAHOE_DIR, "drug_vocab.json")
CL_VOCAB = os.path.join(T.TAHOE_DIR, "cell_line_vocab.json")


def load_vocabs():
    d = json.load(open(DRUG_VOCAB))
    if not os.path.exists(CL_VOCAB):
        raise FileNotFoundError(f"{CL_VOCAB} missing -- run: python build_vocab.py")
    c = json.load(open(CL_VOCAB))
    return c["cell_line_to_id"], d["drug_to_id"], c["n"], d["n_drugs"]


def lognorm(counts, target=1e4):
    """raw counts -> log1p(normalize_total(1e4)); accepts torch or numpy."""
    if isinstance(counts, torch.Tensor):
        lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
        return torch.log1p(counts / lib * target)
    lib = counts.sum(-1, keepdims=True).clip(min=1.0)
    return np.log1p(counts / lib * target).astype(np.float32)


def make_stream(split, cfg):
    return TahoeStream(split=split, batch_size=cfg.data.batch_size,
                       shuffle=(split == "train"), val_every=cfg.data.val_every,
                       cond_cols=(cfg.data.cond_col, cfg.data.pert_col, "plate"))


def to_ids(cond, cfg, c2i, a2i, device):
    """obs string lists -> LongTensors (unknown labels are mapped to 0 and masked out)."""
    cl = cond[cfg.data.cond_col]
    dr = cond[cfg.data.pert_col]
    c_raw = np.array([c2i.get(str(v), -1) for v in cl], dtype=np.int64)
    a_raw = np.array([a2i.get(str(v), -1) for v in dr], dtype=np.int64)
    keep = (c_raw >= 0) & (a_raw >= 0)
    c = torch.from_numpy(np.where(keep, c_raw, 0)).to(device)
    a = torch.from_numpy(np.where(keep, a_raw, 0)).to(device)
    return c, a, torch.from_numpy(keep).to(device)


# --------------------------------------------------------------------------- #
# Control bank
# Tahoe shards are chunked by sample (~66 drugs each), so MOST shards contain no
# DMSO_TF cells at all -> within-shard control matching is impossible. Instead we
# scan once and cache a bank of real control cells grouped by (cell_line, plate);
# training then draws a context-matched control partner for every perturbed cell.
# --------------------------------------------------------------------------- #
CONTROL_BANK = os.path.join(T.TAHOE_DIR, "control_bank.npz")


def build_control_bank(cfg, max_per_group=40, shard_budget=600, seed=0, cache=CONTROL_BANK):
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], [tuple(k) for k in d["keys"]], d["offsets"]
    import collections
    hvg = T.load_hvg()["token_ids"]; nvoc = T.n_vocab()
    shards = T.list_shards("train")
    np.random.default_rng(seed).shuffle(shards)
    pools = collections.defaultdict(list)
    for si, p in enumerate(shards[:shard_budget]):
        X, obs = T.shard_to_csr(p, nvoc, want_obs=True)
        dr = obs[cfg.data.pert_col].astype(str).values
        m = dr == cfg.data.control_label
        if not m.any():
            continue
        cl = obs[cfg.data.cond_col].astype(str).values
        pl = obs["plate"].astype(str).values if "plate" in obs else np.array(["NA"] * len(dr))
        idx = np.where(m)[0]
        hv = X[idx][:, hvg].toarray().astype(np.float32)
        for j, i in enumerate(idx):
            k = (cl[i], pl[i])
            if len(pools[k]) < max_per_group:
                pools[k].append(hv[j])
        if si % 50 == 0:
            full = sum(1 for v in pools.values() if len(v) >= max_per_group)
            print(f"  [control bank] shard {si}: {len(pools)} groups ({full} full)", flush=True)
        if len(pools) >= 700 and all(len(v) >= max_per_group for v in pools.values()):
            break
    keys = sorted(pools)
    offsets = np.cumsum([0] + [len(pools[k]) for k in keys]).astype(np.int64)
    Xb = np.concatenate([np.stack(pools[k]) for k in keys]).astype(np.float32)
    print(f"[control bank] {len(keys)} (cell_line,plate) groups, {len(Xb)} control cells")
    if cache:
        np.savez(cache, X=Xb, keys=np.array(keys, dtype=object), offsets=offsets)
    return Xb, keys, offsets


class ControlBank:
    """Context-matched control cells: sample(cell_line_ids, plates) -> (x_ctl, mask)."""

    def __init__(self, cfg, device, c2i, **kw):
        Xb, keys, offs = build_control_bank(cfg, **kw)
        self.X = torch.from_numpy(lognorm(Xb)).to(device)           # log-norm, on device
        self.by_key, self.by_cl = {}, {}
        for i, (cl, pl) in enumerate(keys):
            rows = torch.arange(offs[i], offs[i + 1], device=device)
            cid = c2i.get(str(cl), -1)
            if cid < 0:
                continue
            self.by_key[(cid, str(pl))] = rows
            self.by_cl[cid] = torch.cat([self.by_cl[cid], rows]) if cid in self.by_cl else rows
        self.device = device

    def sample(self, c_ids, plates):
        """c_ids: LongTensor[B]; plates: list[str] (constant within a shard)."""
        pick = torch.zeros(len(c_ids), dtype=torch.long, device=self.device)
        mask = torch.zeros(len(c_ids), dtype=torch.bool, device=self.device)
        cpu = c_ids.tolist()
        for j, cid in enumerate(cpu):
            rows = self.by_key.get((cid, plates[j])) if plates is not None else None
            if rows is None:
                rows = self.by_cl.get(cid)                          # fall back to cell line only
            if rows is None or len(rows) == 0:
                continue
            pick[j] = rows[int(torch.randint(len(rows), (1,)))]
            mask[j] = True
        return self.X[pick], mask
