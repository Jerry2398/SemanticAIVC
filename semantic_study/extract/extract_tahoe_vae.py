"""
Extract embeddings from a Tahoe-trained VAE for the semantic study + LDM.

Encodes subsampled Tahoe cells (frozen VAE, posterior mean mu) and writes the
canonical embedding .h5ad (X=latent, obs=labels, obsm['X_expr']=log-norm HVG)
via semantic_study's write_embedding -> the SAME files feed both the semantic
metrics and latent_diffusion (which trains on the latents + obs conditions).

Sampling (native Tahoe, HVG panel):
  * TRAIN  : broad uniform sample across train shards (probe train + LDM train).
  * VAL    : broad uniform (semantic breadth: all cell lines / drugs)
             + POWERED (cell_line,drug) groups filled to a cap  (so the LDM
               perturbation eval has >=50-cell groups)
             + DMSO_TF controls per powered cell line (delta baseline).

Run in the squidiff env (torch 2.5 + pyarrow + scanpy):
  SEMANTIC_DATASET=tahoe python extract_tahoe_vae.py --vae tahoe_vae
  SEMANTIC_DATASET=tahoe python extract_tahoe_vae.py --vae tahoe_vae_nb
  SEMANTIC_DATASET=tahoe python extract_tahoe_vae.py --vae tahoe_vae_scgen
"""
import argparse
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                               # semantic_study/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "tahoe"))  # tahoe/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "vae"))    # vae/

import numpy as np
import pandas as pd
import torch

import tahoe_common as T
from _base import write_embedding

MODELS_DIR = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models"
OBS_KEEP = ["drug", "cell_line_id", "moa-fine", "canonical_smiles", "plate"]


def load_vae(name):
    """Instantiate + load the right VAE class from its checkpoint tag."""
    ck = torch.load(os.path.join(MODELS_DIR, name, "vae.pt"), map_location="cpu")
    a, lik, ng = ck["args"], ck.get("likelihood", "gaussian"), ck["n_genes"]
    if lik == "nb":
        from model_nb import NBVAE
        model = NBVAE(ng, a["latent_dim"], a["hidden"], a["depth"])
    elif lik == "gaussian_scgen":
        from model_scgen import SCGenLikeVAE
        model = SCGenLikeVAE(ng, a["latent_dim"], a["hidden"], a["depth"], a.get("dropout", 0.2))
    else:
        from model import SimpleVAE
        model = SimpleVAE(ng, a["latent_dim"], a["hidden"], a["depth"])
    model.load_state_dict(ck["model"]); model.eval()
    return model


def lognorm_np(counts, target=1e4):
    lib = counts.sum(1, keepdims=True).clip(min=1.0)
    return np.log1p(counts / lib * target).astype(np.float32)


def sample_train(hvg_tokens, n_vocab, n_total, cells_per_shard, seed):
    shards = T.list_shards("train")
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(shards))
    counts, obs_rows, got = [], [], 0
    for si in pick:
        X, obs = T.shard_to_csr(shards[si], n_vocab, want_obs=True)
        hvg = X[:, hvg_tokens].toarray().astype(np.float32)
        idx = rng.choice(hvg.shape[0], size=min(cells_per_shard, hvg.shape[0]), replace=False)
        counts.append(hvg[idx]); obs_rows.append(obs.iloc[idx][OBS_KEEP])
        got += len(idx)
        if got >= n_total:
            break
    C = np.concatenate(counts)[:n_total]
    O = pd.concat(obs_rows, ignore_index=True).iloc[:n_total].reset_index(drop=True)
    print(f"[train] sampled {len(C)} cells from {len(counts)} shards")
    return C, O


def sample_val(hvg_tokens, n_vocab, n_broad, n_powered_cls, max_powered,
               cap_grp, cap_ctrl, broad_per_shard, seed):
    shards = T.list_shards("val")
    rng = np.random.default_rng(seed)
    powered_cls = None
    powered, grp_count, ctrl_count = set(), defaultdict(int), defaultdict(int)
    counts, obs_rows, broad_kept = [], [], 0

    def store(hvg, obs, rows):
        counts.append(hvg[rows]); obs_rows.append(obs.iloc[rows][OBS_KEEP])

    for path in shards:
        X, obs = T.shard_to_csr(path, n_vocab, want_obs=True)
        hvg = X[:, hvg_tokens].toarray().astype(np.float32)
        cl = obs["cell_line_id"].astype(str).values
        dr = obs["drug"].astype(str).values
        if powered_cls is None:                    # top-K cell lines from the first val shard
            vc = pd.Series(cl).value_counts()
            powered_cls = set(vc.index[:n_powered_cls].tolist())

        # --- DMSO_TF controls for powered cell lines ---
        for c in powered_cls:
            sel = np.where((cl == c) & (dr == "DMSO_TF"))[0]
            take = min(len(sel), cap_ctrl - ctrl_count[c])
            if take > 0:
                store(hvg, obs, sel[:take]); ctrl_count[c] += take

        # --- powered (cell_line, drug) groups ---
        pmask = np.array([c in powered_cls for c in cl]) & (dr != "DMSO_TF")
        if pmask.any():
            pidx = np.where(pmask)[0]
            gdf = pd.DataFrame({"i": pidx, "cl": cl[pidx], "dr": dr[pidx]})
            for (c, d), sub in gdf.groupby(["cl", "dr"], sort=False):
                k = (c, d)
                if k not in powered:
                    if len(powered) >= max_powered:
                        continue
                    powered.add(k)
                take = min(len(sub), cap_grp - grp_count[k])
                if take > 0:
                    rows = sub["i"].values[:take]
                    store(hvg, obs, rows); grp_count[k] += take

        # --- broad uniform (all cell lines / drugs) for semantic breadth ---
        if broad_kept < n_broad:
            nb = min(broad_per_shard, n_broad - broad_kept, hvg.shape[0])
            rows = rng.choice(hvg.shape[0], size=nb, replace=False)
            store(hvg, obs, rows); broad_kept += nb

        done_ctrl = all(ctrl_count[c] >= cap_ctrl for c in powered_cls)
        done_pow = len(powered) >= max_powered and all(grp_count[k] >= cap_grp for k in powered)
        if broad_kept >= n_broad and done_ctrl and done_pow:
            break

    C = np.concatenate(counts)
    O = pd.concat(obs_rows, ignore_index=True).reset_index(drop=True)
    n_pow_full = sum(1 for k in powered if grp_count[k] >= 50)
    print(f"[val] {len(C)} cells | broad={broad_kept} powered_combos={len(powered)} "
          f"(>=50 cells: {n_pow_full}) ctrl_cells={sum(ctrl_count.values())} "
          f"powered_cls={sorted(powered_cls)}")
    return C, O


@torch.no_grad()
def encode(model, counts, dev, bs=4096):
    outs = []
    for s in range(0, len(counts), bs):
        x = torch.from_numpy(lognorm_np(counts[s:s + bs])).to(dev)
        mu, _ = model.encode(x)
        outs.append(mu.cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True,
                    help="checkpoint dir name under models/ (tahoe_vae, tahoe_vae_nb, "
                         "tahoe_vae_scgen, tahoe_vae_aux20, ...). load_vae infers the class.")
    ap.add_argument("--n_train", type=int, default=120000)
    ap.add_argument("--train_cells_per_shard", type=int, default=500)
    ap.add_argument("--n_val_broad", type=int, default=45000)
    ap.add_argument("--val_broad_per_shard", type=int, default=350)
    ap.add_argument("--n_powered_cls", type=int, default=6)
    ap.add_argument("--max_powered", type=int, default=150)
    ap.add_argument("--cap_grp", type=int, default=80)
    ap.add_argument("--cap_ctrl", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vae(args.vae).to(dev)
    n_vocab = T.n_vocab()                      # full gene-token vocab (62710), for CSR width
    hvg_tokens = T.load_hvg()["token_ids"]     # 5000 HVG token_ids -> VAE input columns
    print(f"[{args.vae}] n_vocab={n_vocab} n_hvg={len(hvg_tokens)} latent={model.latent_dim} device={dev}")

    Ctr, Otr = sample_train(hvg_tokens, n_vocab, args.n_train, args.train_cells_per_shard, args.seed)
    Cva, Ova = sample_val(hvg_tokens, n_vocab, args.n_val_broad, args.n_powered_cls,
                          args.max_powered, args.cap_grp, args.cap_ctrl,
                          args.val_broad_per_shard, args.seed + 1)

    for split, C, O in [("train", Ctr, Otr), ("val", Cva, Ova)]:
        Z = encode(model, C, dev)
        Xexpr = lognorm_np(C)
        write_embedding(args.vae, split, Z, obs=O, expr=Xexpr)


if __name__ == "__main__":
    main()
