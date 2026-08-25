"""
Pseudobulk DD vs number of HVGs (expression space, batch-MIXED only).

Same task/protocol as pseudobulk_multi (avg B cells of (cell_line,drug) -> decode
drug, 2-layer MLP + linear, shard split, 380 classes) but sweeps the HVG count
K in {5000, 2000, 1000, 500}. Genes ranked by per-gene variance of log-norm
expression on TRAIN (highly-variable proxy); top-K are a nested subset of the
5000 panel. log-norm always uses the 5000-panel library size; only the classifier
input gene set shrinks. Latent is excluded (the VAE is fixed to 5000-HVG input).

  python pseudobulk_hvg.py --Ks 5000 2000 1000 500 --Bs 1 16 64
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))

import numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import tahoe_common as T

VOCAB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/drug_vocab.json"
OUT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def lognorm(counts, target=1e4):
    lib = counts.sum(-1, keepdims=True).clip(min=1.0)
    return np.log1p(counts / lib * target).astype(np.float32)


def build_pools(split, hvg, n_vocab, kept, cap, shard_budget, seed):
    shards = T.list_shards(split); rng = np.random.default_rng(seed); rng.shuffle(shards)
    raw = defaultdict(list); full = set(); ntarget = len(kept) * 380
    for si, p in enumerate(shards[:shard_budget]):
        X, obs = T.shard_to_csr(p, n_vocab, want_obs=True)
        cl = obs["cell_line_id"].astype(str).values; dr = obs["drug"].astype(str).values
        m = np.isin(cl, list(kept)) & (dr != "DMSO_TF"); idx = np.where(m)[0]
        if len(idx) == 0: continue
        hv = X[idx][:, hvg].toarray().astype(np.float32)
        for j, i in enumerate(idx):
            g = (cl[i], dr[i])
            if g in full: continue
            raw[g].append(hv[j])
            if len(raw[g]) >= cap: full.add(g)
        if len(full) >= ntarget: break
    pools = {g: {"raw": np.stack(v), "drug": g[1]} for g, v in raw.items()}
    print(f"[{split}] {len(pools)} (cl,drug) pools, median cells/pool="
          f"{int(np.median([len(v['raw']) for v in pools.values()]))}", flush=True)
    return pools


def rank_hvg(pools):
    """Per-gene variance of log-norm over all train cells -> descending gene order."""
    d = next(iter(pools.values()))["raw"].shape[1]
    n = 0; s = np.zeros(d, np.float64); ss = np.zeros(d, np.float64)
    for v in pools.values():
        ln = lognorm(v["raw"]); n += len(ln); s += ln.sum(0); ss += (ln * ln).sum(0)
    var = ss / n - (s / n) ** 2
    return np.argsort(-var)          # gene indices, most-variable first


def make_bulks(units, feature, gidx, B, cap, d2i, rng):
    Xs, ys = [], []
    for u in units:
        n = len(u["raw"])
        if u["drug"] not in d2i or n < B: continue
        feat = (u["raw"] if feature == "raw" else lognorm(u["raw"]))[:, gidx]
        order = rng.permutation(n); nb = min(cap, n // B)
        for k in range(nb):
            Xs.append(feat[order[k*B:(k+1)*B]].mean(0)); ys.append(d2i[u["drug"]])
    return np.stack(Xs).astype(np.float32), np.array(ys, np.int64)


class MLP2(nn.Module):
    def __init__(self, ni, no, h=512, p=0.1):
        super().__init__(); self.net = nn.Sequential(nn.Linear(ni, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, no))
    def forward(self, x): return self.net(x)


def train_eval(kind, Xtr, ytr, Xva, yva, ncls, epochs=80, lr=1e-3, bs=2048, seed=0):
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
        if vl < best-1e-4: best, bstate, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= 10: break
    if bstate: model.load_state_dict(bstate)
    model.eval()
    with torch.no_grad(): pred = model(torch.from_numpy(Xva).to(DEV)).argmax(1).cpu().numpy()
    return {"accuracy": float(accuracy_score(yva, pred)), "macro_f1": float(f1_score(yva, pred, average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", type=int, nargs="+", default=[5000, 2000, 1000, 500])
    ap.add_argument("--Bs", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--k_cl", type=int, default=8)
    ap.add_argument("--cap_train", type=int, default=400)
    ap.add_argument("--cap_val", type=int, default=200)
    ap.add_argument("--n_bulk_cap", type=int, default=16)
    ap.add_argument("--train_shards", type=int, default=1000)
    ap.add_argument("--val_shards", type=int, default=170)
    args = ap.parse_args()

    vocab = json.load(open(VOCAB)); d2i = vocab["drug_to_id"]; ncls = vocab["n_drugs"]
    hvg = T.load_hvg()["token_ids"]; n_vocab = T.n_vocab()
    import pyarrow.parquet as pq
    cl0 = pq.read_table(T.list_shards("train")[0], columns=["cell_line_id"]).to_pandas()["cell_line_id"].astype(str)
    kept = set(cl0.value_counts().index[:args.k_cl].tolist())
    print(f"kept {len(kept)} cell lines | Ks={args.Ks} Bs={args.Bs} device={DEV}", flush=True)

    trp = build_pools("train", hvg, n_vocab, kept, args.cap_train, args.train_shards, 0)
    vap = build_pools("val", hvg, n_vocab, kept, args.cap_val, args.val_shards, 1)
    order = rank_hvg(trp)                       # gene ranking from TRAIN only

    rng = np.random.default_rng(0)
    tu, vu = list(trp.values()), list(vap.values())
    out = {"experiment": "pseudobulk_hvg", "grouping": "mixed", "k_cl": args.k_cl, "results": {}}
    for K in args.Ks:
        gidx = order[:K]
        for feature in ("raw", "lognorm"):
            for B in args.Bs:
                Xtr, ytr = make_bulks(tu, feature, gidx, B, args.n_bulk_cap, d2i, rng)
                Xva, yva = make_bulks(vu, feature, gidx, B, args.n_bulk_cap, d2i, rng)
                mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
                Xtr, Xva = (Xtr-mu)/sd, (Xva-mu)/sd
                chance = float(np.bincount(yva, minlength=ncls).max()/len(yva))
                key = f"HVG{K}/{feature}/B{B}"
                res = {"n_train": int(len(Xtr)), "n_val": int(len(Xva)), "chance": chance,
                       "mlp": train_eval("mlp", Xtr, ytr, Xva, yva, ncls),
                       "linear": train_eval("linear", Xtr, ytr, Xva, yva, ncls)}
                out["results"][key] = res
                print(f"{key:22s} n_va={len(Xva):6d} chance={chance:.4f} || "
                      f"mlp {res['mlp']['accuracy']:.4f} (F1 {res['mlp']['macro_f1']:.4f}) lin {res['linear']['accuracy']:.4f}", flush=True)
    json.dump(out, open(os.path.join(OUT, "pseudobulk_hvg_results.json"), "w"), indent=2)
    print("PSEUDOBULK_HVG_DONE")


if __name__ == "__main__":
    main()
