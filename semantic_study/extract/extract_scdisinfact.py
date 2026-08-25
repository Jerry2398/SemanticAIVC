"""
Extract scDisInFact latent embeddings -> shared embedding .h5ad (train + val).

scDisInFact (ZhangLabGT/scDisInFact) is a COUNT-based VAE that disentangles a
shared biological factor (z_c / mu_c, condition-invariant) from condition
factors (drug, dose). The embedding used for the semantic study is mu_c.

API (import name is `scDisInFact`, capital):
  data_dict = create_scdisinfact_dataset(counts, meta, condition_key=[...], batch_key=..)
              -> dict with 'datasets' (per batch x condition group), 'meta_cells', 'scaler'
  model = scdisinfact(data_dict, Ks=[k_shared, k_diff...], device=..)
  model.train_model(nepochs, recon_loss='NB')
  model.inference(counts_norm, batch_ids) -> dict with 'mu_c' (shared latent)

We add an "_ord" column so we can reassemble per-cell mu_c back into the
canonical cell order that _base.write_embedding expects.

ENV: /scratch/yuchen.yan/envs/sem_scdisinfact (py3.10 + torch cu121 + scDisInFact).
Run via: qsub -v MODEL=scdisinfact submit_encoder.sh
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp
import torch

import config as C
from _base import load_split, write_embedding


def build_dataset(sdf, adata, condition_cols, batch_col):
    counts = adata.layers["counts"]
    counts = sp.csr_matrix(counts.toarray() if hasattr(counts, "toarray") else counts)
    meta = adata.obs.copy()
    meta["_ord"] = np.arange(len(meta))
    for c in condition_cols + [batch_col]:
        meta[c] = meta[c].astype(str)
    dd = sdf.create_scdisinfact_dataset(counts, meta, condition_key=condition_cols, batch_key=batch_col)
    return dd


def extract_mu_c(model, dd, n_cells, device):
    """Run inference over every group, collect mu_c, reorder to original cells."""
    model.eval()
    mus, orders = [], []
    with torch.no_grad():
        for ds, mc in zip(dd["datasets"], dd["meta_cells"]):
            cn = ds.counts_norm.to(device)
            bid = ds.batch_id.reshape(-1, 1).to(device)
            out = model.inference(cn, bid)
            mu = out["mu_c"] if isinstance(out, dict) else out[0]
            mus.append(np.asarray(mu.detach().cpu().numpy()))
            orders.append(np.asarray(mc["_ord"].values))
    mu_all = np.concatenate(mus, 0)
    order = np.concatenate(orders, 0)
    Z = np.zeros((n_cells, mu_all.shape[1]), dtype=np.float32)
    Z[order] = mu_all
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition_cols", nargs="+", default=C.SCDISINFACT_CONDITION_COLS)
    ap.add_argument("--batch_col", default=C.BATCH_KEY)
    ap.add_argument("--k_shared", type=int, default=C.LATENT_DIM)   # shared latent = the embedding; unified across encoders
    ap.add_argument("--k_cond", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max_train_cells", type=int, default=100000,
                    help="scDisInFact holds data on-GPU and OOMs on ~559k cells; "
                         "train on a subsample, then embed ALL cells via inference.")
    ap.add_argument("--name", default="scdisinfact")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    import scDisInFact as sdf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va = load_split("train"), load_split("val")

    # --- train on a subsample (memory) ---
    tr_fit = tr
    if args.max_train_cells and tr.n_obs > args.max_train_cells:
        idx = np.random.default_rng(0).choice(tr.n_obs, args.max_train_cells, replace=False)
        idx.sort()
        tr_fit = tr[idx].copy()
        print(f"scDisInFact: training on {tr_fit.n_obs}/{tr.n_obs} cells (subsample)")
    dd_fit = build_dataset(sdf, tr_fit, args.condition_cols, args.batch_col)

    Ks = [args.k_shared] + [args.k_cond] * len(args.condition_cols)
    model = sdf.scdisinfact(data_dict=dd_fit, Ks=Ks, device=device)
    model.train_model(nepochs=args.epochs, recon_loss="NB")

    # --- extract embeddings for ALL cells (inference is per-group, low memory) ---
    dd_tr = build_dataset(sdf, tr, args.condition_cols, args.batch_col)
    Z_tr = extract_mu_c(model, dd_tr, tr.n_obs, device)
    dd_va = build_dataset(sdf, va, args.condition_cols, args.batch_col)
    Z_va = extract_mu_c(model, dd_va, va.n_obs, device)

    write_embedding(args.name, "train", Z_tr)
    write_embedding(args.name, "val", Z_va)

    if args.save_model:
        d = C.models_dir(args.name)
        os.makedirs(d, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(d, "model.pt"))
        print(f"[{args.name}] saved checkpoint -> {d}")


if __name__ == "__main__":
    main()
