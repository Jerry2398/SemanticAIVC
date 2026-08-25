"""
Experiment A: does PCA-compressing expression to the latent dim keep drug signal?

Fit PCA on TRAIN 5000-HVG expression -> k dims (k=32 matches the VAE latent),
then the SAME DD classifier (2-layer MLP + linear, train->val) on the PCA
features. Comparable to:
  - full 5000-HVG single-cell DD (expr_dd_probe): acc 0.039 (mlp)
  - VAE-latent-32 DD: 0.037 (baseline) / 0.062 (drug-supervised)
If PCA-32 ~ chance too, a 32-d linear compression isn't the reason DD is low.

  python pca_dd_probe.py --dims 32 128
"""
import argparse, json, os
import anndata as ad, numpy as np, torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

EMB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/embeddings"
VOCAB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/drug_vocab.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    a = ad.read_h5ad(path)
    X = a.obsm["X_expr"]
    X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)
    return X, a.obs["drug"].astype(str).values


class MLP2(nn.Module):
    def __init__(self, n_in, n_out, h=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, n_out))
    def forward(self, x): return self.net(x)


def train_eval(kind, Xtr, ytr, Xva, yva, ncls, epochs=60, lr=1e-3, bs=4096, seed=0):
    torch.manual_seed(seed)
    model = (nn.Linear(Xtr.shape[1], ncls) if kind == "linear" else MLP2(Xtr.shape[1], ncls)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); lf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed); perm = rng.permutation(len(Xtr)); c = max(1, int(0.1*len(Xtr)))
    iva, itr = perm[:c], perm[c:]
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr[itr]), torch.from_numpy(ytr[itr])), batch_size=bs, shuffle=True)
    Xiv, yiv = torch.from_numpy(Xtr[iva]).to(DEV), torch.from_numpy(ytr[iva]).to(DEV)
    best, bstate, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV); opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = lf(model(Xiv), yiv).item()
        if vl < best - 1e-4: best, bstate, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= 8: break
    if bstate: model.load_state_dict(bstate)
    model.eval()
    with torch.no_grad(): pred = model(torch.from_numpy(Xva).to(DEV)).argmax(1).cpu().numpy()
    return {"accuracy": float(accuracy_score(yva, pred)), "macro_f1": float(f1_score(yva, pred, average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 128])
    args = ap.parse_args()
    vocab = json.load(open(VOCAB)); c2i = vocab["drug_to_id"]; ncls = vocab["n_drugs"]
    Xtr, ltr = load(os.path.join(EMB, "tahoe_vae_train.h5ad"))
    Xva, lva = load(os.path.join(EMB, "tahoe_vae_val.h5ad"))
    ytr = np.array([c2i[c] for c in ltr], dtype=np.int64)
    yva = np.array([c2i[c] for c in lva], dtype=np.int64)
    chance = float(np.bincount(yva, minlength=ncls).max()/len(yva))
    print(f"n_train={len(Xtr)} n_val={len(Xva)} n_classes={ncls} chance={chance:.4f} device={DEV}")
    out = {"experiment": "PCA_expression_DD", "n_classes": ncls, "chance": chance,
           "reference": {"full5000_mlp": 0.039, "vae_latent32_baseline": 0.037, "vae_latent32_sup": 0.062}}
    for k in args.dims:
        pca = PCA(n_components=k, random_state=0).fit(Xtr)
        Ztr, Zva = pca.transform(Xtr).astype(np.float32), pca.transform(Xva).astype(np.float32)
        evr = float(pca.explained_variance_ratio_.sum())
        mu, sd = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True)+1e-6
        Ztr, Zva = (Ztr-mu)/sd, (Zva-mu)/sd
        res = {"explained_var": evr}
        for kind in ("mlp", "linear"):
            res[kind] = train_eval(kind, Ztr, ytr, Zva, yva, ncls)
        out[f"pca{k}"] = res
        print(f"PCA-{k} (EVR={evr:.3f}): mlp acc {res['mlp']['accuracy']:.4f} (F1 {res['mlp']['macro_f1']:.4f}) | "
              f"linear acc {res['linear']['accuracy']:.4f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pca_results.json"), "w"), indent=2)
    print("PCA_DD_DONE")


if __name__ == "__main__":
    main()
