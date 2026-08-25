"""
Standalone validation of a trained Tahoe NB-VAE: reconstruction of val HVG
expression (metrics computed in log-norm space, comparable to the Gaussian VAE)
plus a generative mean-profile check.

  python evaluate_nb.py --max_batches 200
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))

import torch
from torch.utils.data import DataLoader

from model_nb import NBVAE
from common import collect_recon, reconstruction_metrics, generative_mean_match
from tahoe_dataset import TahoeStream

OUT_DIR = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models/tahoe_vae_nb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT_DIR, "vae.pt"))
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_batches", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=dev)
    a = ck["args"]
    model = NBVAE(ck["n_genes"], a["latent_dim"], a["hidden"], a["depth"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    va = TahoeStream(split="val", batch_size=args.batch_size, shuffle=False, cond_cols=())
    X, Xh = collect_recon(model, DataLoader(va, batch_size=None, num_workers=0), dev, max_batches=args.max_batches)
    m = reconstruction_metrics(X, Xh)
    g = generative_mean_match(model, X.mean(0), dev)
    m.update(g); m.update({"n_val_cells": int(X.shape[0]), "step": ck.get("step"), "likelihood": "nb"})
    print("=== Tahoe NB-VAE validation ===")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    json.dump(m, open(os.path.join(OUT_DIR, "val_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
