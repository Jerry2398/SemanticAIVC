"""
Squidiff extractor (v2) -- supports both trained variants:

  --variant nodrug : expression-only encoder, z_sem = encoder(x_expr).
                     Reads the canonical treated files. Emb name 'squidiff_nodrug'.
  --variant drug   : drug-structure encoder, z_sem = encoder(control, drug x dose).
                     Reads the *_drug + *_drug_control files. Emb name 'squidiff_drug'.

Writes the shared embedding format via _base.write_embedding, so metrics run with
  python run_all.py --encoder squidiff_nodrug   (or squidiff_drug)

Usage:
  HDF5_USE_FILE_LOCKING=FALSE python extract/extract_squidiff_embeddings.py --variant nodrug --split both
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scanpy as sc
import torch

import config as C
from common import to_dense
from _base import load_split, write_embedding

sys.path.insert(0, C.SQUIDIFF_ROOT)
from Squidiff.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from Squidiff.scrna_datasets import Drug_dose_encoder

VARIANTS = {
    "nodrug": dict(cfg=C.SQUIDIFF_NODRUG_CFG, model=C.SQUIDIFF_NODRUG_MODEL_PATH, emb="squidiff_nodrug"),
    "drug":   dict(cfg=C.SQUIDIFF_DRUG_CFG,   model=C.SQUIDIFF_DRUG_MODEL_PATH,   emb="squidiff_drug"),
}


def load_model(cfg_update, model_path, dev):
    cfg = model_and_diffusion_defaults()
    cfg.update(cfg_update)
    model, _ = create_model_and_diffusion(**cfg)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    return model.to(dev).eval()


def encode_nodrug(model, adata, dev, batch=4096):
    x = torch.tensor(to_dense(adata.X), dtype=torch.float32).to(dev)
    zs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            zs.append(model.encoder(x[i:i + batch]).cpu().numpy())
    return np.concatenate(zs, 0).astype(np.float32)


def encode_drug(model, treated, ctrl, dev, batch=4096):
    dd = torch.tensor(Drug_dose_encoder(treated.obs["SMILES"].astype(str).tolist(),
                                        treated.obs["dose"].astype(float).tolist()),
                      dtype=torch.float32).to(dev)
    cf = torch.tensor(to_dense(ctrl.X), dtype=torch.float32).to(dev)
    zs = []
    with torch.no_grad():
        for i in range(0, cf.shape[0], batch):
            zs.append(model.encoder(None, label=None, drug_dose=dd[i:i + batch],
                                    control_feature=cf[i:i + batch]).cpu().numpy())
    return np.concatenate(zs, 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["nodrug", "drug"], required=True)
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--model_path", default=None, help="override checkpoint path")
    args = ap.parse_args()

    v = VARIANTS[args.variant]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model_path or v["model"]
    print(f"variant={args.variant} device={dev} model={model_path}")
    model = load_model(v["cfg"], model_path, dev)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        if args.variant == "nodrug":
            z = encode_nodrug(model, load_split(split), dev)
            obs = expr = None  # _base pulls canonical obs / X_expr
        else:
            tre_path = C.TRAIN_DRUG_PATH if split == "train" else C.VAL_DRUG_PATH
            ctl_path = C.TRAIN_DRUG_CONTROL_PATH if split == "train" else C.VAL_DRUG_CONTROL_PATH
            treated, ctrl = sc.read_h5ad(tre_path), sc.read_h5ad(ctl_path)
            z = encode_drug(model, treated, ctrl, dev)
            obs, expr = treated.obs.copy(), to_dense(treated.X).astype(np.float32)
        write_embedding(v["emb"], split, z, obs=obs, expr=expr)


if __name__ == "__main__":
    main()
