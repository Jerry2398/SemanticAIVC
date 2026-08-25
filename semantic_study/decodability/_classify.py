"""
Shared classification-probe runner for CTD / DD / MOA decodability.

All three answer the same shape of question -- "is <label> linearly decodable
from the latent?" -- so they share one routine: fit a linear softmax probe on
the TRAIN embeddings, evaluate Accuracy + macro-F1 on the VAL embeddings.

Classes are defined from the TRAIN split; VAL cells whose label is unseen in
train are dropped (a probe cannot predict a class it never saw) and counted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from common import load_embedding, save_results
from probes import train_classifier


def run_classification(metric, label_col, train_emb, val_emb,
                       probe="linear", results_name=None):
    Ztr, obs_tr, _ = load_embedding(train_emb)
    Zva, obs_va, _ = load_embedding(val_emb)

    ytr_raw = obs_tr[label_col].astype(str).values
    yva_raw = obs_va[label_col].astype(str).values

    classes = sorted(np.unique(ytr_raw).tolist())
    cls_to_id = {c: i for i, c in enumerate(classes)}

    keep = np.array([c in cls_to_id for c in yva_raw])
    n_dropped = int((~keep).sum())
    Zva, yva_raw = Zva[keep], yva_raw[keep]

    ytr = np.array([cls_to_id[c] for c in ytr_raw], dtype=np.int64)
    yva = np.array([cls_to_id[c] for c in yva_raw], dtype=np.int64)

    print(f"[{metric}] label={label_col!r} n_classes={len(classes)} "
          f"train={len(ytr)} val={len(yva)} (dropped {n_dropped} unseen-class val cells)")

    logits = train_classifier(Ztr, ytr, Zva, n_classes=len(classes), kind=probe)
    pred = logits.argmax(1)

    result = {
        "metric": metric,
        "label_col": label_col,
        "probe": probe,
        "n_classes": len(classes),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_val_dropped_unseen_class": n_dropped,
        "accuracy": float(accuracy_score(yva, pred)),
        "macro_f1": float(f1_score(yva, pred, average="macro")),
        "weighted_f1": float(f1_score(yva, pred, average="weighted")),
        "chance_accuracy": float(np.bincount(yva, minlength=len(classes)).max() / len(yva)),
    }
    save_results(results_name or metric.lower(), result)
    return result
