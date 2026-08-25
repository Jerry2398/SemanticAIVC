"""
DD -- Drug Decodability.

Is the applied drug decodable from the latent? Linear softmax probe:
latent -> drug identity. Metrics: Accuracy, macro-F1 (val split).
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
    ap.add_argument("--label_col", default=C.LABELS["drug"])
    ap.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    args = ap.parse_args()
    run_classification("DD", args.label_col, args.train_emb, args.val_emb, args.probe)


if __name__ == "__main__":
    main()
