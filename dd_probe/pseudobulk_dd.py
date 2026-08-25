"""
Experiment B: pseudobulk DD -- how much drug signal recovers with aggregation?

Average B cells of the SAME (cell_line, drug) group into one pseudobulk sample,
then decode drug (380-way) with the SAME classifier (2-layer MLP + linear,
train->val) as the single-cell / PCA / latent DD probes. Sweep B (B=1 == single
cell -> consistency anchor with the 0.039 single-cell result).

Comparability: same 5000-HVG log-norm feature space, same classifier/protocol,
same shard-level train/val split. Restricted to the top-K_CL cell lines (drug
still 380-way) to bound streaming/memory; reports n_samples + chance per B.

Runs as a PBS job (heavy streaming). Writes pseudobulk_results.json.
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))

import numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import tahoe_common as T

VOCAB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/drug_vocab.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def lognorm(counts, target=1e4):
    lib = counts.sum(1, keepdims=True).clip(min=1.0)
    return np.log1p(counts / lib * target).astype(np.float32)


def build_pools(split, hvg, n_vocab, kept_cls, cap, shard_budget, seed):
    """Stream shards -> per-(cell_line,drug) capped cell pools (log-norm HVG)."""
    shards = T.list_shards(split)
    rng = np.random.default_rng(seed); rng.shuffle(shards)
    pools = defaultdict(list); full = set()
    n_groups_target = len(kept_cls) * 380
    for si, p in enumerate(shards[:shard_budget]):
        X, obs = T.shard_to_csr(p, n_vocab, want_obs=True)
        cl = obs["cell_line_id"].astype(str).values
        dr = obs["drug"].astype(str).values
        m = np.isin(cl, list(kept_cls)) & (dr != "DMSO_TF")
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        hv = X[idx][:, hvg].toarray().astype(np.float32)
        for j, i in enumerate(idx):
            g = (cl[i], dr[i])
            if g in full:
                continue
            pools[g].append(hv[j])
            if len(pools[g]) >= cap:
                full.add(g)
        if len(full) >= n_groups_target or si == shard_budget - 1:
            if len(full) >= n_groups_target:
                break
    pools = {g: np.stack(v) for g, v in pools.items() if len(v) >= 1}
    print(f"[{split}] pools: {len(pools)} groups, scanned<= {shard_budget} shards, "
          f"median cells/group={int(np.median([len(v) for v in pools.values()]))}", flush=True)
    return pools


def make_bulks(pools, drug_to_id, B, n_bulk_cap, rng):
    """Disjoint partition each pool into pseudobulks of B cells (mean). Label=drug."""
    Xs, ys = [], []
    for (cl, dr), P in pools.items():
        if dr not in drug_to_id or len(P) < B:
            continue
        order = rng.permutation(len(P))
        nb = min(n_bulk_cap, len(P) // B)
        for k in range(nb):
            sel = order[k * B:(k + 1) * B]
            Xs.append(P[sel].mean(0))
            ys.append(drug_to_id[dr])
    return np.stack(Xs).astype(np.float32), np.array(ys, dtype=np.int64)


class MLP2(nn.Module):
    def __init__(self, n_in, n_out, h=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, n_out))
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
        if vl < best - 1e-4: best, bstate, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= 10: break
    if bstate: model.load_state_dict(bstate)
    model.eval()
    with torch.no_grad(): pred = model(torch.from_numpy(Xva).to(DEV)).argmax(1).cpu().numpy()
    return {"accuracy": float(accuracy_score(yva, pred)), "macro_f1": float(f1_score(yva, pred, average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--k_cl", type=int, default=10)
    ap.add_argument("--cap_train", type=int, default=512)
    ap.add_argument("--cap_val", type=int, default=256)
    ap.add_argument("--n_bulk_cap", type=int, default=16)
    ap.add_argument("--train_shards", type=int, default=800)
    ap.add_argument("--val_shards", type=int, default=170)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vocab = json.load(open(VOCAB)); drug_to_id = vocab["drug_to_id"]; ncls = vocab["n_drugs"]
    hvg = T.load_hvg()["token_ids"]; n_vocab = T.n_vocab()

    # pick top-K cell lines from the first train shard (deterministic, shared train/val)
    import pyarrow.parquet as pq
    s0 = T.list_shards("train")[0]
    cl0 = pq.read_table(s0, columns=["cell_line_id"]).to_pandas()["cell_line_id"].astype(str)
    kept = set(cl0.value_counts().index[:args.k_cl].tolist())
    print(f"kept {len(kept)} cell lines: {sorted(kept)} | Bs={args.Bs} device={DEV}", flush=True)

    tr_pools = build_pools("train", hvg, n_vocab, kept, args.cap_train, args.train_shards, args.seed)
    va_pools = build_pools("val", hvg, n_vocab, kept, args.cap_val, args.val_shards, args.seed + 1)

    out = {"experiment": "pseudobulk_DD", "n_classes": ncls, "k_cl": args.k_cl,
           "reference_singlecell_mlp": 0.039, "results": {}}
    rng = np.random.default_rng(args.seed)
    for B in args.Bs:
        Xtr, ytr = make_bulks(tr_pools, drug_to_id, B, args.n_bulk_cap, rng)
        Xva, yva = make_bulks(va_pools, drug_to_id, B, args.n_bulk_cap, rng)
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        chance = float(np.bincount(yva, minlength=ncls).max() / len(yva))
        res = {"B": B, "n_train_bulk": int(len(Xtr)), "n_val_bulk": int(len(Xva)),
               "n_drugs_val": int(len(np.unique(yva))), "chance": chance}
        for kind in ("mlp", "linear"):
            res[kind] = train_eval(kind, Xtr, ytr, Xva, yva, ncls)
        out["results"][str(B)] = res
        print(f"B={B:3d} | n_train={len(Xtr)} n_val={len(Xva)} chance={chance:.4f} "
              f"|| mlp acc {res['mlp']['accuracy']:.4f} (F1 {res['mlp']['macro_f1']:.4f}) "
              f"linear acc {res['linear']['accuracy']:.4f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pseudobulk_results.json"), "w"), indent=2)
    print("PSEUDOBULK_DD_DONE")


if __name__ == "__main__":
    main()
