"""
Sanity/verification for the expression->drug probe.

Runs the SAME pipeline (X_expr -> label, 2-layer MLP + linear, train->val) on an
arbitrary obs column, class table built directly from TRAIN labels (also rules out
any drug_vocab mapping bug). Use cell_line_id as a POSITIVE CONTROL: if a known-
strong label (CTD~0.97) decodes well here, the pipeline is correct and drug~chance
is real; if cell_line ALSO fails, there's a bug.

  python verify_probe.py --label_col cell_line_id   # positive control (expect HIGH acc)
  python verify_probe.py --label_col drug           # reproduce (expect ~chance)
"""
import argparse, json, os
import anndata as ad, numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

EMB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/embeddings"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(path, col):
    a = ad.read_h5ad(path)
    X = a.obsm["X_expr"]
    X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)
    return X, a.obs[col].astype(str).values


class MLP2(nn.Module):
    def __init__(self, n_in, n_out, h=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, n_out))
    def forward(self, x): return self.net(x)


def train_eval(kind, Xtr, ytr, Xva, yva, ncls, epochs=60, lr=1e-3, bs=4096, seed=0):
    torch.manual_seed(seed)
    model = (nn.Linear(Xtr.shape[1], ncls) if kind == "linear" else MLP2(Xtr.shape[1], ncls)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed); perm = rng.permutation(len(Xtr)); c = max(1, int(0.1*len(Xtr)))
    iva, itr = perm[:c], perm[c:]
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr[itr]), torch.from_numpy(ytr[itr])), batch_size=bs, shuffle=True)
    Xiv, yiv = torch.from_numpy(Xtr[iva]).to(DEV), torch.from_numpy(ytr[iva]).to(DEV)
    best, bstate, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = lf(model(Xiv), yiv).item()
        if vl < best - 1e-4: best, bstate, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if ep % 5 == 0: print(f"    [{kind}] ep{ep} iv-CE {vl:.3f} (best {best:.3f}, bad {bad})", flush=True)
        if bad >= 8: break
    if bstate: model.load_state_dict(bstate)
    model.eval()
    with torch.no_grad(): pred = model(torch.from_numpy(Xva).to(DEV)).argmax(1).cpu().numpy()
    return {"accuracy": float(accuracy_score(yva, pred)),
            "macro_f1": float(f1_score(yva, pred, average="macro")),
            "weighted_f1": float(f1_score(yva, pred, average="weighted"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_col", default="drug")
    ap.add_argument("--train_emb", default=os.path.join(EMB, "tahoe_vae_train.h5ad"))
    ap.add_argument("--val_emb", default=os.path.join(EMB, "tahoe_vae_val.h5ad"))
    args = ap.parse_args()

    Xtr, ltr = load(args.train_emb, args.label_col)
    Xva, lva = load(args.val_emb, args.label_col)
    classes = sorted(np.unique(ltr).tolist()); c2i = {c: i for i, c in enumerate(classes)}
    ytr = np.array([c2i[c] for c in ltr], dtype=np.int64)
    keep = np.array([c in c2i for c in lva]); Xva, lva = Xva[keep], lva[keep]
    yva = np.array([c2i[c] for c in lva], dtype=np.int64)
    chance = float(np.bincount(yva, minlength=len(classes)).max()/len(yva))
    print(f"label={args.label_col} n_classes={len(classes)} n_train={len(Xtr)} n_val={len(Xva)} "
          f"dropped_unseen_val={int((~keep).sum())} device={DEV}")
    print(f"X_expr: min={Xtr.min():.3f} max={Xtr.max():.3f} mean={Xtr.mean():.3f} nonzero_frac={(Xtr>0).mean():.3f}")
    print(f"chance_acc(majority)={chance:.4f} | label examples: {ltr[:3].tolist()}")
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)+1e-6
    Xtr, Xva = (Xtr-mu)/sd, (Xva-mu)/sd
    out = {"label_col": args.label_col, "n_classes": len(classes), "n_val": int(len(Xva)), "chance": chance}
    for kind in ("mlp", "linear"):
        print(f"=== {kind} ===")
        r = train_eval(kind, Xtr, ytr, Xva, yva, len(classes)); out[kind] = r
        print(f"  {kind}: acc {r['accuracy']:.4f} macroF1 {r['macro_f1']:.4f} weightedF1 {r['weighted_f1']:.4f}")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"verify_{args.label_col}.json"), "w"), indent=2)
    print("VERIFY_DONE", args.label_col)


if __name__ == "__main__":
    main()
