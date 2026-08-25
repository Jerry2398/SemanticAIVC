"""
DoseD -- Dose Decodability.

Is the applied dose decodable from the latent? Linear regression probe:
latent -> dose (log10-scaled by default, since sciplex doses span 10..10000).
Loss: MSE. Metrics: MAE, R2 (val split).

MAE and R2 are reported on the modeling scale (log10 dose if DOSE_LOG10) and,
for interpretability, MAE is also reported back on the raw dose scale.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

import config as C
from common import load_embedding, save_results
from probes import train_regressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_emb", default=C.TRAIN_EMB_PATH)
    ap.add_argument("--val_emb", default=C.VAL_EMB_PATH)
    ap.add_argument("--dose_col", default=C.LABELS["dose"])
    ap.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--log10", type=int, default=int(C.DOSE_LOG10))
    args = ap.parse_args()

    Ztr, obs_tr, _ = load_embedding(args.train_emb)
    Zva, obs_va, _ = load_embedding(args.val_emb)

    dtr = obs_tr[args.dose_col].astype(float).values
    dva = obs_va[args.dose_col].astype(float).values
    ytr = np.log10(dtr + 1.0) if args.log10 else dtr
    yva = np.log10(dva + 1.0) if args.log10 else dva

    pred = train_regressor(Ztr, ytr.astype(np.float32), Zva, kind=args.probe)

    # back to raw dose scale for an interpretable MAE
    pred_raw = (10.0 ** pred - 1.0) if args.log10 else pred

    result = {
        "metric": "DoseD",
        "dose_col": args.dose_col,
        "probe": args.probe,
        "log10_scaled": bool(args.log10),
        "n_val": int(len(yva)),
        "mae": float(mean_absolute_error(yva, pred)),
        "r2": float(r2_score(yva, pred)),
        "mae_raw_dose": float(mean_absolute_error(dva, pred_raw)),
        "unique_doses": sorted(np.unique(dva).tolist()),
    }
    save_results("dose_d", result)


if __name__ == "__main__":
    main()
