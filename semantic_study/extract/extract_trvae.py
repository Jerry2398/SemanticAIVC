"""
Extract trVAE latent embeddings -> shared embedding .h5ad (train + val).

trVAE (scArches) is a conditional VAE with an MMD penalty that integrates OVER a
chosen condition. With recon_loss='mse' it trains on log-norm data (no counts
needed). Choose the condition to remove via --condition_key:
  * batch      : remove technical batch, KEEP biology  (recommended default)
  * cell_type  : integrate over cell line (will hurt CTD -- usually not wanted)

IMPORTANT: val conditions must be a subset of train conditions.

ENV (own conda env):
  conda create -n trvae python=3.9 -y && conda activate trvae
  pip install scarches scanpy
Run:
  HDF5_USE_FILE_LOCKING=FALSE python extract/extract_trvae.py --condition_key batch
  python run_all.py --encoder trvae
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scarches.models import TRVAE

import config as C
from common import to_dense
from _base import load_split, write_embedding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_latent", type=int, default=C.LATENT_DIM)   # unified across encoders
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--max_train_cells", type=int, default=100000,
                    help="trVAE's MMD over 52 plates x 559k cells is far too slow "
                         "(hit walltime at 3%); fit on a subsample, embed ALL cells.")
    ap.add_argument("--condition_key", default=C.BATCH_KEY, help="condition to integrate over (technical; keeps biology)")
    ap.add_argument("--recon_loss", choices=["nb", "mse"], default="nb",
                    help="nb on raw counts (official) or mse on log-norm X")
    ap.add_argument("--name", default="trvae")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    tr, va = load_split("train"), load_split("val")
    tr.obs[args.condition_key] = tr.obs[args.condition_key].astype(str)
    va.obs[args.condition_key] = va.obs[args.condition_key].astype(str)
    conditions = sorted(tr.obs[args.condition_key].unique().tolist())  # all plates (superset)

    # nb likelihood needs raw counts as X; mse uses the log-norm X.
    if args.recon_loss == "nb":
        tr.X = tr.layers["counts"].copy()
        va.X = va.layers["counts"].copy()

    # fit on a subsample (keeps all conditions in `conditions` so get_latent works on all cells)
    tr_fit = tr
    if args.max_train_cells and tr.n_obs > args.max_train_cells:
        idx = np.random.default_rng(0).choice(tr.n_obs, args.max_train_cells, replace=False)
        idx.sort(); tr_fit = tr[idx].copy()
    print(f"trVAE: recon_loss={args.recon_loss} latent={args.n_latent} cond={args.condition_key} "
          f"fit_cells={tr_fit.n_obs}/{tr.n_obs} epochs={args.epochs} bs={args.batch_size}")

    model = TRVAE(adata=tr_fit, condition_key=args.condition_key, conditions=conditions,
                  latent_dim=args.n_latent, recon_loss=args.recon_loss)
    early_stopping_kwargs = {
        "early_stopping_metric": "val_unweighted_loss", "mode": "min",
        "threshold": 0, "patience": 15, "reduce_lr": True,
        "lr_patience": 10, "lr_factor": 0.1,
    }
    model.train(n_epochs=args.epochs, batch_size=args.batch_size,
                early_stopping_kwargs=early_stopping_kwargs)

    # batch get_latent so 559k x 5000 doesn't blow up GPU memory
    def latent(adata, chunk=50000):
        X = to_dense(adata.X); c = adata.obs[args.condition_key].values
        return np.concatenate([np.asarray(model.get_latent(X[i:i+chunk], c[i:i+chunk]))
                               for i in range(0, X.shape[0], chunk)], 0)

    write_embedding(args.name, "train", latent(tr))
    write_embedding(args.name, "val", latent(va))

    if args.save_model:
        d = C.models_dir(args.name)
        os.makedirs(d, exist_ok=True)
        model.save(d, overwrite=True)
        print(f"[{args.name}] saved checkpoint -> {d}")


if __name__ == "__main__":
    main()
