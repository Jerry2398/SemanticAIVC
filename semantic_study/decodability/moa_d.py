"""
MOA-D -- Mechanism-of-Action Decodability.

Is the drug's mechanism of action decodable from the latent? Linear softmax
probe: latent -> MOA. For sciplex we use the `pathway` annotation as MOA
(configurable via --label_col; e.g. pathway_level_1 or target).
Metrics: Accuracy, macro-F1 (val split).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from _classify import run_classification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_emb", default=C.TRAIN_EMB_PATH)
    ap.add_argument("--val_emb", default=C.VAL_EMB_PATH)
    ap.add_argument("--label_col", default=C.LABELS["moa"])
    ap.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    args = ap.parse_args()
    run_classification("MOA-D", args.label_col, args.train_emb, args.val_emb,
                       args.probe, results_name="moa_d")


if __name__ == "__main__":
    main()
