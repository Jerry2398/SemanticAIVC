"""
GER -- Gene Expression Recoverability.

Question: does the latent preserve the whole transcriptome? (Information
completeness.) Train a linear probe  nn.Linear(latent_dim, n_genes)  with MSE on
the TRAIN split, evaluate on the VAL split.

Metrics (all averaged, nan-safe):
  * gene-wise Pearson  : mean over genes of corr(pred[:,g], true[:,g])
  * cell-wise Pearson  : mean over cells of corr(pred[c,:], true[c,:])
  * gene-wise R2
  * cell-wise R2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
from common import (load_embedding, save_results, nanmean,
                    columnwise_pearson, rowwise_pearson,
                    columnwise_r2, rowwise_r2)
from probes import train_regressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_emb", default=C.TRAIN_EMB_PATH)
    ap.add_argument("--val_emb", default=C.VAL_EMB_PATH)
    ap.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    args = ap.parse_args()

    Ztr, _, Xtr = load_embedding(args.train_emb, need_expr=True)
    Zva, _, Xva = load_embedding(args.val_emb, need_expr=True)
    print(f"train Z={Ztr.shape} expr={Xtr.shape} | val Z={Zva.shape} expr={Xva.shape}")

    pred = train_regressor(Ztr, Xtr, Zva, kind=args.probe)

    gene_r = columnwise_pearson(pred, Xva)
    cell_r = rowwise_pearson(pred, Xva)
    gene_r2 = columnwise_r2(pred, Xva)
    cell_r2 = rowwise_r2(pred, Xva)

    result = {
        "metric": "GER",
        "probe": args.probe,
        "n_genes": int(Xva.shape[1]),
        "n_val_cells": int(Xva.shape[0]),
        "gene_wise_pearson": nanmean(gene_r),
        "cell_wise_pearson": nanmean(cell_r),
        "gene_wise_r2": nanmean(gene_r2),
        "cell_wise_r2": nanmean(cell_r2),
        "gene_wise_pearson_median": float(np.nanmedian(gene_r)),
        "cell_wise_pearson_median": float(np.nanmedian(cell_r)),
    }
    save_results("ger", result)


if __name__ == "__main__":
    main()
