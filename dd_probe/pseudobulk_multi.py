"""
Pseudobulk DD across feature spaces x batch-grouping (extends pseudobulk_dd.py).

Same task/protocol as pseudobulk_dd.py (avg B cells of a group -> decode drug,
2-layer MLP + linear, shard-level train/val, 380 classes) but sweeps:
  feature  in {raw, lognorm, latent}   (raw = mean raw counts [prev run];
             lognorm = mean of per-cell log-norm [= single-cell expr space];
             latent = mean of the gauss-VAE latent embedding)
  grouping in {mixed, clean}           (mixed key=(cell_line,drug) [prev, mixes
             plates]; clean key=(cell_line,drug,plate) [single-plate pseudobulks])

One streaming pass: per cell store raw HVG (5000) + latent (32), keyed by
(cell_line,drug,plate); lognorm derived from raw; mixed units merge plates of the
same (cell_line,drug). Everything else aligned so all 18 configs are comparable.

NOTE (batch shortcut): clean grouping makes each pseudobulk single-plate, but the
shard-level split still shares plates between train/val, so a plate<->drug shortcut
can persist; a plate-HOLDOUT split (future) is needed to fully remove it.
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vae"))

import numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import tahoe_common as T

VOCAB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/drug_vocab.json"
MODELS = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models"
OUT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def lognorm(counts, target=1e4):
    lib = counts.sum(-1, keepdims=True).clip(min=1.0)
    return np.log1p(counts / lib * target).astype(np.float32)


def build_gauss(name="tahoe_vae"):
    from model import SimpleVAE
    ck = torch.load(os.path.join(MODELS, name, "vae.pt"), map_location="cpu")
    a = ck["args"]; m = SimpleVAE(ck["n_genes"], a["latent_dim"], a["hidden"], a["depth"])
    m.load_state_dict(ck["model"]); m.eval()
    return m


def build_pools(split, vae, hvg, n_vocab, kept, cap, shard_budget, seed, bs=8192):
    shards = T.list_shards(split); rng = np.random.default_rng(seed); rng.shuffle(shards)
    raw = defaultdict(list); lat = defaultdict(list); full = set()
    ntarget = len(kept) * 380 * 3
    for si, p in enumerate(shards[:shard_budget]):
        X, obs = T.shard_to_csr(p, n_vocab, want_obs=True)
        cl = obs["cell_line_id"].astype(str).values; dr = obs["drug"].astype(str).values
        pl = obs["plate"].astype(str).values
        m = np.isin(cl, list(kept)) & (dr != "DMSO_TF"); idx = np.where(m)[0]
        if len(idx) == 0: continue
        hv = X[idx][:, hvg].toarray().astype(np.float32)
        with torch.no_grad():
            z = np.concatenate([vae.encode(torch.from_numpy(lognorm(hv[b:b+bs])).to(DEV))[0].cpu().numpy()
                                for b in range(0, len(hv), bs)]).astype(np.float32)
        for j, i in enumerate(idx):
            g = (cl[i], dr[i], pl[i])
            if g in full: continue
            raw[g].append(hv[j]); lat[g].append(z[j])
            if len(raw[g]) >= cap: full.add(g)
        if len(full) >= ntarget: break
    pools = {g: {"raw": np.stack(raw[g]), "lat": np.stack(lat[g]), "drug": g[1]} for g in raw}
    print(f"[{split}] {len(pools)} (cl,drug,plate) pools, median cells/pool="
          f"{int(np.median([len(v['raw']) for v in pools.values()]))}", flush=True)
    return pools


def units_for(pools, grouping):
    if grouping == "clean":
        return list(pools.values())
    merged = defaultdict(lambda: {"raw": [], "lat": [], "drug": None})
    for (cl, dr, pl), v in pools.items():
        merged[(cl, dr)]["raw"].append(v["raw"]); merged[(cl, dr)]["lat"].append(v["lat"]); merged[(cl, dr)]["drug"] = dr
    return [{"raw": np.concatenate(u["raw"]), "lat": np.concatenate(u["lat"]), "drug": u["drug"]} for u in merged.values()]


def make_bulks(units, feature, B, cap, drug_to_id, rng):
    Xs, ys = [], []
    for u in units:
        n = len(u["raw"])
        if u["drug"] not in drug_to_id or n < B: continue
        order = rng.permutation(n); nb = min(cap, n // B)
        src = u["lat"] if feature == "latent" else u["raw"]
        for k in range(nb):
            sel = order[k*B:(k+1)*B]
            if feature == "raw":       v = src[sel].mean(0)
            elif feature == "lognorm": v = lognorm(src[sel]).mean(0)
            else:                      v = src[sel].mean(0)          # latent
            Xs.append(v); ys.append(drug_to_id[u["drug"]])
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
    ap.add_argument("--Bs", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--k_cl", type=int, default=8)
    ap.add_argument("--cap_train", type=int, default=300)
    ap.add_argument("--cap_val", type=int, default=150)
    ap.add_argument("--n_bulk_cap", type=int, default=16)
    ap.add_argument("--train_shards", type=int, default=900)
    ap.add_argument("--val_shards", type=int, default=170)
    args = ap.parse_args()

    vocab = json.load(open(VOCAB)); d2i = vocab["drug_to_id"]; ncls = vocab["n_drugs"]
    hvg = T.load_hvg()["token_ids"]; n_vocab = T.n_vocab()
    vae = build_gauss().to(DEV)
    import pyarrow.parquet as pq
    cl0 = pq.read_table(T.list_shards("train")[0], columns=["cell_line_id"]).to_pandas()["cell_line_id"].astype(str)
    kept = set(cl0.value_counts().index[:args.k_cl].tolist())
    print(f"kept {len(kept)} cell lines | Bs={args.Bs} device={DEV}", flush=True)

    trp = build_pools("train", vae, hvg, n_vocab, kept, args.cap_train, args.train_shards, 0)
    vap = build_pools("val", vae, hvg, n_vocab, kept, args.cap_val, args.val_shards, 1)

    rng = np.random.default_rng(0)
    out = {"experiment": "pseudobulk_multi", "k_cl": args.k_cl, "results": {}}
    for grouping in ("mixed", "clean"):
        tu, vu = units_for(trp, grouping), units_for(vap, grouping)
        for feature in ("raw", "lognorm", "latent"):
            for B in args.Bs:
                Xtr, ytr = make_bulks(tu, feature, B, args.n_bulk_cap, d2i, rng)
                Xva, yva = make_bulks(vu, feature, B, args.n_bulk_cap, d2i, rng)
                mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
                Xtr, Xva = (Xtr-mu)/sd, (Xva-mu)/sd
                chance = float(np.bincount(yva, minlength=ncls).max()/len(yva))
                key = f"{grouping}/{feature}/B{B}"
                res = {"n_train": int(len(Xtr)), "n_val": int(len(Xva)), "chance": chance,
                       "mlp": train_eval("mlp", Xtr, ytr, Xva, yva, ncls),
                       "linear": train_eval("linear", Xtr, ytr, Xva, yva, ncls)}
                out["results"][key] = res
                print(f"{key:26s} n_tr={len(Xtr):6d} n_va={len(Xva):6d} chance={chance:.4f} "
                      f"|| mlp {res['mlp']['accuracy']:.4f} (F1 {res['mlp']['macro_f1']:.4f}) lin {res['linear']['accuracy']:.4f}", flush=True)
    json.dump(out, open(os.path.join(OUT, "pseudobulk_multi_results.json"), "w"), indent=2)
    print("PSEUDOBULK_MULTI_DONE")


if __name__ == "__main__":
    main()
