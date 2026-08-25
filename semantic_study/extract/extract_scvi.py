"""
Extract scVI latent embeddings -> shared embedding .h5ad (train + val).

scVI is an NB/ZINB VAE for RAW COUNTS. The canonical v2 dataset carries real
counts in layers['counts'], so --counts_source layer (default) is the correct,
standard setup. ('reconstruct' / 'lognorm' remain for count-less data.)

ENV (own conda env; conflicts with squidiff env):
  conda create -n scvi python=3.10 -y && conda activate scvi && pip install scvi-tools scanpy
Run:
  HDF5_USE_FILE_LOCKING=FALSE python extract/extract_scvi.py --save_model
  python run_all.py --encoder scvi
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scvi

import config as C
from _base import load_split, reconstruct_counts, write_embedding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_latent", type=int, default=C.LATENT_DIM)   # unified across encoders
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--counts_source", choices=["layer", "reconstruct", "lognorm"], default="layer")
    ap.add_argument("--batch_key", default=None, help="obs column for batch (e.g. 'plate'); None = ignore")
    ap.add_argument("--name", default="scvi")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    tr, va = load_split("train"), load_split("val")

    if args.counts_source == "layer":
        layer, likelihood = "counts", "zinb"
    elif args.counts_source == "reconstruct":
        tr.layers["counts"] = reconstruct_counts(tr)
        va.layers["counts"] = reconstruct_counts(va)
        layer, likelihood = "counts", "zinb"
    else:
        layer, likelihood = None, "normal"  # model log-norm X with a Gaussian head
    print(f"scVI: counts_source={args.counts_source} likelihood={likelihood} n_latent={args.n_latent}")

    scvi.model.SCVI.setup_anndata(tr, layer=layer, batch_key=args.batch_key)
    model = scvi.model.SCVI(tr, n_latent=args.n_latent, gene_likelihood=likelihood)
    # early stopping on the internal validation ELBO ensures convergence
    # (scvi holds out train_size=0.9 / 0.1 internally for this).
    model.train(max_epochs=args.epochs, early_stopping=True,
                early_stopping_monitor="elbo_validation", early_stopping_patience=15)

    z_tr = model.get_latent_representation(tr)
    z_va = model.get_latent_representation(va)  # scVI transfers the trained setup

    write_embedding(args.name, "train", z_tr)
    write_embedding(args.name, "val", z_va)

    if args.save_model:
        d = C.models_dir(args.name)
        os.makedirs(d, exist_ok=True)
        model.save(d, overwrite=True)
        print(f"[{args.name}] saved checkpoint -> {d}")


if __name__ == "__main__":
    main()
