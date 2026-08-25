"""
Extract scGen latent embeddings -> shared embedding .h5ad (train + val).

scGen is a Gaussian VAE built on scvi-tools, trained on LOG-NORMALIZED data --
so it works directly on this sciplex file (no counts needed). It is the natural
"vanilla VAE on expression" baseline.

ENV (own conda env):
  conda create -n scgen python=3.10 -y && conda activate scgen
  pip install scgen scanpy
Run:
  HDF5_USE_FILE_LOCKING=FALSE python extract/extract_scgen.py
  python run_all.py --encoder scgen
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scgen

import config as C
from _base import load_split, write_embedding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_latent", type=int, default=C.LATENT_DIM, help="latent dim; unified across encoders")
    ap.add_argument("--batch_key", default=None, help="obs col for batch conditioning; None = pure autoencoder")
    ap.add_argument("--labels_key", default=None, help="obs col for label conditioning; None = none")
    ap.add_argument("--batch_size", type=int, default=512, help="larger batch -> avoid walltime on 559k cells")
    ap.add_argument("--max_train_cells", type=int, default=50000,
                    help="scGen's legacy torch 1.12 can't run on the Hopper GPU (no sm_90 "
                         "kernels) and scvi-0.16/lightning-1.5 can't use new torch, so scGen "
                         "trains on CPU; subsample to keep it tractable. Embeds ALL cells.")
    ap.add_argument("--use_gpu", type=int, default=0, help="0 = CPU (default; old torch can't use Hopper)")
    ap.add_argument("--name", default="scgen")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    tr, va = load_split("train"), load_split("val")

    tr_fit = tr
    if args.max_train_cells and tr.n_obs > args.max_train_cells:
        idx = np.random.default_rng(0).choice(tr.n_obs, args.max_train_cells, replace=False)
        idx.sort(); tr_fit = tr[idx].copy()
    print(f"scGen: fit_cells={tr_fit.n_obs}/{tr.n_obs} epochs={args.epochs} "
          f"bs={args.batch_size} use_gpu={bool(args.use_gpu)}")

    scgen.SCGEN.setup_anndata(tr_fit, batch_key=args.batch_key, labels_key=args.labels_key)
    model = scgen.SCGEN(tr_fit, n_latent=args.n_latent)
    model.train(max_epochs=args.epochs, batch_size=args.batch_size, use_gpu=bool(args.use_gpu),
                early_stopping=True, early_stopping_patience=25)

    z_tr = model.get_latent_representation(tr)
    z_va = model.get_latent_representation(va)

    write_embedding(args.name, "train", z_tr)
    write_embedding(args.name, "val", z_va)

    if args.save_model:
        d = C.models_dir(args.name)
        os.makedirs(d, exist_ok=True)
        model.save(d, overwrite=True)
        print(f"[{args.name}] saved checkpoint -> {d}")


if __name__ == "__main__":
    main()
