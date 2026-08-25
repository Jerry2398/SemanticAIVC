"""
Ceiling check: can DRUG identity be decoded directly from raw cell EXPRESSION?

A 2-layer MLP classifier   expression (5000-HVG log-norm) -> drug (380 classes),
trained on the Tahoe TRAIN cells and evaluated on the held-out VAL cells -- the
exact same cells / preprocessing / train->val protocol as the semantic-study DD
metric, but the input is the EXPRESSION (obsm['X_expr']) instead of a VAE latent.

If this expression-space DD is high while the latent-space DD is low, the VAE
bottleneck is discarding drug signal. If BOTH are low, drug identity is simply
weakly decodable from (HVG) expression -- not the VAE's fault.

Self-contained: reads the embedding .h5ad (X_expr + obs['drug']) + drug_vocab.json.
Does NOT import or modify any existing project code.

  python expr_dd_probe.py                      # 2-layer MLP (default)
  python expr_dd_probe.py --probe both         # + linear reference
"""
import argparse
import json
import os

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

EMB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/embeddings"
VOCAB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/drug_vocab.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_xy(path, drug_to_id):
    a = ad.read_h5ad(path)
    X = a.obsm["X_expr"]
    X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)
    y = np.array([drug_to_id.get(str(d), -1) for d in a.obs["drug"]], dtype=np.int64)
    return X, y


class MLP2(nn.Module):
    def __init__(self, n_in, n_out, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def train_probe(kind, Xtr, ytr, Xva, n_classes, hidden, dropout, epochs, lr, bs, seed=0):
    torch.manual_seed(seed)
    model = (nn.Linear(Xtr.shape[1], n_classes) if kind == "linear"
             else MLP2(Xtr.shape[1], n_classes, hidden, dropout)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    # internal train/val split for early stopping (no peeking at the real val)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr)); ncut = max(1, int(0.1 * len(Xtr)))
    iva, itr = perm[:ncut], perm[ncut:]
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr[itr]), torch.from_numpy(ytr[itr])),
                    batch_size=bs, shuffle=True)
    Xiv = torch.from_numpy(Xtr[iva]).to(DEV); yiv = torch.from_numpy(ytr[iva]).to(DEV)

    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xiv), yiv).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    [{kind}] epoch {ep:3d} internal-val CE {vl:.3f} (best {best:.3f}, bad {bad})", flush=True)
        if bad >= 8:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xva).to(DEV)).cpu().numpy()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_emb", default=os.path.join(EMB, "tahoe_vae_train.h5ad"))
    ap.add_argument("--val_emb", default=os.path.join(EMB, "tahoe_vae_val.h5ad"))
    ap.add_argument("--probe", choices=["mlp", "linear", "both"], default="mlp")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"))
    args = ap.parse_args()

    vocab = json.load(open(VOCAB)); drug_to_id = vocab["drug_to_id"]; n_cls = vocab["n_drugs"]
    Xtr, ytr = load_xy(args.train_emb, drug_to_id)
    Xva, yva = load_xy(args.val_emb, drug_to_id)
    keep = yva >= 0; Xva, yva = Xva[keep], yva[keep]     # drop val cells whose drug unseen (none expected)

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6   # standardize (train stats)
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    chance = float(np.bincount(yva, minlength=n_cls).max() / len(yva))
    print(f"input=EXPRESSION({Xtr.shape[1]}-HVG) n_train={len(Xtr)} n_val={len(Xva)} "
          f"n_classes={n_cls} chance_acc={chance:.4f} device={DEV}")

    kinds = ["mlp", "linear"] if args.probe == "both" else [args.probe]
    out = {"input": "expression_5000HVG_lognorm", "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
           "n_classes": n_cls, "chance_accuracy": chance,
           "reference_latent_DD": {"baseline_gauss": 0.037, "best_identity_mlp_l100": 0.058,
                                    "best_identity_linear_l100": 0.062}}
    for kind in kinds:
        print(f"\n=== training {kind} probe (expression -> drug) ===")
        logits = train_probe(kind, Xtr, ytr, Xva, n_cls, args.hidden, args.dropout,
                             args.epochs, args.lr, args.batch_size)
        pred = logits.argmax(1)
        top5 = np.mean([yva[i] in logits[i].argsort()[-5:] for i in range(len(yva))])
        res = {"accuracy": float(accuracy_score(yva, pred)),
               "macro_f1": float(f1_score(yva, pred, average="macro")),
               "weighted_f1": float(f1_score(yva, pred, average="weighted")),
               "top5_accuracy": float(top5)}
        out[kind] = res
        print(f"  [{kind}] acc {res['accuracy']:.4f} | macro-F1 {res['macro_f1']:.4f} | "
              f"weighted-F1 {res['weighted_f1']:.4f} | top5-acc {res['top5_accuracy']:.4f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
