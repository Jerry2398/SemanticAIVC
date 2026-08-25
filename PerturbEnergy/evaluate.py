"""
Inference + verification for PerturbEnergy.  Tasks (--tasks):

 recon      : held-out reconstruction fidelity (MSE / R2 / gene & cell Pearson).
 predict    : PERTURBATION PREDICTION (paper "Inference"): for each (cell_line, drug)
              group, take matched DMSO_TF control cells, z_b0 ~ q(z_b|x0,c0), draw M
              candidates z_a ~ q(z_a|z_b0,a), KEEP THE LOWEST-ENERGY one, decode, and
              compare with the real perturbed population. Metrics are identical to
              latent_diffusion/evaluate_expression.py (R2/Pearson of mean and of delta
              vs control, all genes + top-K DEGs, plus an encode-real-cells ceiling),
              so numbers are directly comparable to the LDM baselines.
 retrieval  : energy-based drug decoding (Theorem 2 / Eq.18): rank all candidate drugs
              by E_alpha(z_a(a'),z_b,a'); report top-1/top-5/MRR per cell AND per
              population (energies averaged over a group's cells before ranking).
 probe      : linear/MLP probes for drug identity from z_b and z_a + Energy PP-SNR,
              with collapse diagnostics (within-condition spread, effective rank).
              NOTE: z_a is COMPUTED FROM a, so a probe on z_a is circular by
              construction -- reported only as a diagnostic, never as evidence.

Cells come from the same pre-extracted train/val files used by the LDM evaluation.

  python evaluate.py --tasks recon predict retrieval probe
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, r2_score

import cfg as CFG
from data import load_vocabs
from models import PerturbEnergy

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEV)
    c = CFG.Cfg(ck["cfg"])
    m = PerturbEnergy(ck["n_genes"], ck["n_cond"], ck["n_pert"], dict(c.model)).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, c, ck


def read_cells(path, n_max=None, seed=0):
    A = ad.read_h5ad(path)
    X = A.obsm["X_expr"]
    X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)
    obs = A.obs.reset_index(drop=True)
    if n_max and len(X) > n_max:
        idx = np.random.default_rng(seed).choice(len(X), n_max, replace=False)
        X, obs = X[idx], obs.iloc[idx].reset_index(drop=True)
    return X, obs


def ids_of(obs, cfg, c2i, a2i):
    c = np.array([c2i.get(str(v), -1) for v in obs[cfg.data.cond_col]], np.int64)
    a = np.array([a2i.get(str(v), -1) for v in obs[cfg.data.pert_col]], np.int64)
    return c, a


@torch.no_grad()
def encode_all(model, X, c, a, bs=4096):
    """posterior means (deterministic) -> z_b, z_a"""
    zb, za = [], []
    for s in range(0, len(X), bs):
        o = model(torch.from_numpy(X[s:s+bs]).to(DEV), torch.from_numpy(c[s:s+bs]).to(DEV),
                  torch.from_numpy(a[s:s+bs]).to(DEV), sample=False)
        zb.append(o["z_b"].cpu().numpy()); za.append(o["z_a"].cpu().numpy())
    return np.concatenate(zb), np.concatenate(za)


# ------------------------------- 1. recon ---------------------------------- #
@torch.no_grad()
def task_recon(model, X, c, a, bs=4096):
    Xh = []
    for s in range(0, len(X), bs):
        x = torch.from_numpy(X[s:s+bs]).to(DEV)
        ct = torch.from_numpy(c[s:s+bs]).to(DEV); at = torch.from_numpy(a[s:s+bs]).to(DEV)
        Xh.append(model(x, ct, at, sample=False)["x_hat"].cpu().numpy())
    Xh = np.concatenate(Xh)
    def colr(A, B):
        A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
        return float(np.nanmean((A*B).sum(0) / (np.sqrt((A**2).sum(0)*(B**2).sum(0)) + 1e-8)))
    ss = ((X - Xh) ** 2).sum(); st = ((X - X.mean()) ** 2).sum() + 1e-8
    return {"recon_mse": float(((X - Xh) ** 2).mean()), "r2": float(1 - ss/st),
            "gene_pearson": colr(Xh, X), "cell_pearson": colr(Xh.T, X.T), "n_cells": int(len(X))}


# --------------------- 2. perturbation prediction -------------------------- #
def _const(v, n):
    return torch.full((n,), int(v), dtype=torch.long, device=DEV)


@torch.no_grad()
def predict_response(model, x0, c_id, a_id, M):
    """paper Inference: z_b0 from CONTROL cells, M candidate z_a, energy picks one, decode."""
    x0 = torch.from_numpy(x0).to(DEV)
    c_e, a_e = model.embed(_const(c_id, len(x0)), _const(a_id, len(x0)))
    z_b = model.basal(x0, c_e, sample=False)[0]                 # z_b0 (posterior mean)
    _, mu_a, lv_a = model.pert(z_b, a_e, sample=False)
    z_a, e_sel, _ = model.select_z_a(mu_a, lv_a, z_b, a_e, n=max(1, M))
    return model.decode(z_b, z_a).cpu().numpy(), float(e_sel.mean())


@torch.no_grad()
def encode_decode(model, X, c_id, a_id):
    """ceiling: encode the REAL perturbed cells and decode (decoder-limited upper bound)."""
    n = len(X)
    o = model(torch.from_numpy(X).to(DEV), _const(c_id, n), _const(a_id, n), sample=False)
    return o["x_hat"].cpu().numpy()


def task_predict(model, cfg, X, obs, c, a, ecfg):
    ctl_name = cfg.data.control_label
    drug = obs[cfg.data.pert_col].astype(str).values
    cl = obs[cfg.data.cond_col].astype(str).values
    is_ctl = drug == ctl_name
    base = {k: X[(cl == k) & is_ctl].mean(0) for k in np.unique(cl) if ((cl == k) & is_ctl).sum() > 0}
    keys = np.array([f"{u}|{v}" for u, v in zip(cl, drug)])
    ev = (~is_ctl) & np.array([k in base for k in cl])
    uniq, cnt = np.unique(keys[ev], return_counts=True)
    order = np.argsort(-cnt)
    groups = [u for u, n in zip(uniq[order], cnt[order]) if n >= ecfg["min_cells"]][:ecfg["max_groups"]]
    print(f"[predict] {len(groups)} groups (>= {ecfg['min_cells']} cells), baseline = {ctl_name} per cell line")

    def r2(u, v): return float(r2_score(u, v))
    def pr(u, v):
        s = np.std(u) * np.std(v)
        return float(np.corrcoef(u, v)[0, 1]) if s > 0 else float("nan")

    rows = []
    for gi, g in enumerate(groups):
        m = (keys == g) & ev
        ci = int(c[m][0]); ai = int(a[m][0]); cln = cl[m][0]
        if ci < 0 or ai < 0:
            continue
        b = base[cln]
        mu_real = X[m].mean(0)
        ctl_idx = np.where((cl == cln) & is_ctl)[0]
        take = ctl_idx[:ecfg["n_per_group"]]
        x_pred, e_mean = predict_response(model, X[take], ci, ai, ecfg["m_samples"])
        mu_pred = x_pred.mean(0)
        mu_ceil = encode_decode(model, X[m], ci, ai).mean(0)
        d_r, d_p, d_c = mu_real - b, mu_pred - b, mu_ceil - b
        deg = np.argsort(np.abs(d_r))[-ecfg["top_degs"]:]
        rows.append({"group": g, "n_real": int(m.sum()), "n_ctrl_used": int(len(take)),
                     "energy_sel": e_mean,
                     "r2_mean_all": r2(mu_real, mu_pred), "pearson_mean_all": pr(mu_real, mu_pred),
                     "r2_mean_deg": r2(mu_real[deg], mu_pred[deg]),
                     "pearson_mean_deg": pr(mu_real[deg], mu_pred[deg]),
                     "r2_delta_all": r2(d_r, d_p), "pearson_delta_all": pr(d_r, d_p),
                     "r2_delta_deg": r2(d_r[deg], d_p[deg]),
                     "pearson_delta_deg": pr(d_r[deg], d_p[deg]),
                     "r2_mean_all_vae": r2(mu_real, mu_ceil), "r2_mean_deg_vae": r2(mu_real[deg], mu_ceil[deg]),
                     "r2_delta_all_vae": r2(d_r, d_c), "r2_delta_deg_vae": r2(d_r[deg], d_c[deg])})
        if gi < 5:
            r = rows[-1]
            print(f"  [{gi:3d}] {g[:34]:34s} mean(R2 {r['r2_mean_all']:.3f}/deg {r['r2_mean_deg']:.3f}) "
                  f"delta(R2 {r['r2_delta_all']:.3f}/deg {r['r2_delta_deg']:.3f} r_deg {r['pearson_delta_deg']:.3f})", flush=True)
    ks = [k for k in rows[0] if k not in ("group", "n_real", "n_ctrl_used")] if rows else []
    return {"n_groups": len(rows), "baseline": f"{ctl_name} per cell line",
            **{k: float(np.nanmean([r[k] for r in rows])) for k in ks}}, rows


# --------------------- 3. energy-based drug retrieval ---------------------- #
@torch.no_grad()
def task_retrieval(model, cfg, X, obs, c, a, n_pert, n_cells=4000, bs=256, seed=0):
    """Rank every candidate drug by E_alpha(z_a(a'),z_b,a')  (lower energy = better)."""
    keep = (c >= 0) & (a >= 0) & (obs[cfg.data.pert_col].astype(str).values != cfg.data.control_label)
    idx = np.where(keep)[0]
    if len(idx) > n_cells:
        idx = np.random.default_rng(seed).choice(idx, n_cells, replace=False)
    Xs, cs, as_ = X[idx], c[idx], a[idx]
    cl = obs[cfg.data.cond_col].astype(str).values[idx]
    dr = obs[cfg.data.pert_col].astype(str).values[idx]
    all_a = torch.arange(n_pert, device=DEV)
    E = np.zeros((len(Xs), n_pert), np.float32)
    for s in range(0, len(Xs), bs):
        x = torch.from_numpy(Xs[s:s+bs]).to(DEV)
        ct = torch.from_numpy(cs[s:s+bs]).to(DEV)
        c_e = model.cond_emb(ct)
        z_b = model.basal(x, c_e, sample=False)[0]                # [n, zb]
        n = z_b.shape[0]
        zb_rep = z_b.repeat_interleave(n_pert, 0)                 # [n*P, zb]
        ae_rep = model.pert_emb(all_a.repeat(n))
        mu_a = model.pert(zb_rep, ae_rep, sample=False)[1]        # posterior mean
        E[s:s+bs] = model.energy(mu_a, zb_rep, ae_rep).view(n, n_pert).cpu().numpy()
    rank = np.argsort(E, 1)                                       # ascending energy
    true_rank = np.array([np.where(rank[i] == as_[i])[0][0] for i in range(len(as_))])
    out = {"n_cells": int(len(Xs)), "n_candidates": int(n_pert),
           "cell_top1": float((true_rank == 0).mean()), "cell_top5": float((true_rank < 5).mean()),
           "cell_mrr": float(np.mean(1.0 / (true_rank + 1))),
           "cell_median_rank": float(np.median(true_rank + 1)),
           "chance_top1": 1.0 / n_pert}
    # population-level: average energies over each (cell_line, drug) group, then rank
    grp = np.array([f"{u}|{v}" for u, v in zip(cl, dr)])
    p1 = p5 = tot = 0; mrr = []
    for g in np.unique(grp):
        m = grp == g
        if m.sum() < 5:
            continue
        e = E[m].mean(0)
        r = int(np.where(np.argsort(e) == as_[m][0])[0][0])
        p1 += (r == 0); p5 += (r < 5); mrr.append(1.0/(r+1)); tot += 1
    if tot:
        out.update({"pop_groups": tot, "pop_top1": p1/tot, "pop_top5": p5/tot,
                    "pop_mrr": float(np.mean(mrr))})
    return out


# ------------------- 4. probes + PP-SNR + collapse checks ------------------- #
class MLP2(nn.Module):
    def __init__(self, ni, no, h=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(ni, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, no))
    def forward(self, x): return self.net(x)


def probe(kind, Ztr, ytr, Zva, yva, ncls, epochs=60, lr=1e-3, bs=4096, seed=0):
    torch.manual_seed(seed)
    mu, sd = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True) + 1e-6
    Ztr, Zva = (Ztr-mu)/sd, (Zva-mu)/sd
    model = (nn.Linear(Ztr.shape[1], ncls) if kind == "linear" else MLP2(Ztr.shape[1], ncls)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Ztr), torch.from_numpy(ytr)
    best, bstate, bad = 1e9, None, 0
    rng = np.random.default_rng(seed); perm = rng.permutation(len(Xt)); ncut = max(1, len(Xt)//10)
    iv, it = perm[:ncut], perm[ncut:]
    Xiv, yiv = Xt[iv].to(DEV), yt[iv].to(DEV)
    for ep in range(epochs):
        model.train(); p = rng.permutation(len(it))
        for s in range(0, len(it), bs):
            b = it[p[s:s+bs]]
            xb, yb = Xt[b].to(DEV), yt[b].to(DEV)
            opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = lf(model(Xiv), yiv).item()
        if vl < best-1e-4: best, bstate, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= 8: break
    if bstate: model.load_state_dict(bstate)
    model.eval()
    with torch.no_grad(): pred = model(torch.from_numpy(Zva).to(DEV)).argmax(1).cpu().numpy()
    return {"accuracy": float(accuracy_score(yva, pred)), "macro_f1": float(f1_score(yva, pred, average="macro"))}


def ppsnr(Z, obs, cfg, eps=1e-2):
    """Energy PP-SNR (control-standardized squared Euclidean closed form), matched on
    (cell_line, plate); returns mean/median over perturbations + collapse diagnostics."""
    cl = obs[cfg.data.cond_col].astype(str).values
    dr = obs[cfg.data.pert_col].astype(str).values
    pl = obs["plate"].astype(str).values if "plate" in obs else np.array(["p"] * len(dr))
    ctl = dr == cfg.data.control_label
    vals, within = [], []
    for c_ in np.unique(cl):
        for p_ in np.unique(pl[cl == c_]):
            mC = (cl == c_) & (pl == p_) & ctl
            if mC.sum() < 20:
                continue
            Y = Z[mC]; muY = Y.mean(0); vY = Y.var(0, ddof=1)
            sc2 = (np.sqrt(np.clip(vY, 0, None)) + eps) ** 2
            for d_ in np.unique(dr[(cl == c_) & (pl == p_) & ~ctl]):
                mX = (cl == c_) & (pl == p_) & (dr == d_)
                if mX.sum() < 20:
                    continue
                Xg = Z[mX]; muX = Xg.mean(0); vX = Xg.var(0, ddof=1)
                num = float(np.sum((muX - muY) ** 2 / sc2))
                den = float(np.sum((np.clip(vX, 0, None) + np.clip(vY, 0, None)) / sc2))
                vals.append(num / (den + 1e-8)); within.append(float(np.sqrt(vX.mean())))
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return {"n_perturbations": len(vals),
            "mean_ppsnr": float(np.mean(vals)) if vals else None,
            "median_ppsnr": float(np.median(vals)) if vals else None,
            "within_cond_std": float(np.mean(within)) if within else None,
            "effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum()))}


def task_probe(model, cfg, Xtr, otr, ctr, atr, Xva, ova, cva, ava, n_pert):
    ktr = (ctr >= 0) & (atr >= 0); kva = (cva >= 0) & (ava >= 0)
    zb_tr, za_tr = encode_all(model, Xtr[ktr], ctr[ktr], atr[ktr])
    zb_va, za_va = encode_all(model, Xva[kva], cva[kva], ava[kva])
    ytr, yva = atr[ktr], ava[kva]
    out = {"chance_accuracy": float(np.bincount(yva, minlength=n_pert).max() / len(yva))}
    for name, Ztr, Zva in (("z_b", zb_tr, zb_va), ("z_a", za_tr, za_va),
                           ("z_concat", np.hstack([zb_tr, za_tr]), np.hstack([zb_va, za_va]))):
        out[f"dd_{name}_linear"] = probe("linear", Ztr, ytr, Zva, yva, n_pert)
        out[f"dd_{name}_mlp"] = probe("mlp", Ztr, ytr, Zva, yva, n_pert)
        out[f"ppsnr_{name}"] = ppsnr(Zva, ova[kva].reset_index(drop=True), cfg)
    out["_note"] = ("z_a (and z_concat) are computed FROM the perturbation label a, so drug "
                    "probes/PP-SNR on them are circular by construction -- diagnostics only. "
                    "z_b is the perturbation-invariant control (low DD == good disentanglement); "
                    "the non-circular perturbation-information test is the 'retrieval' task.")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    CFG.add_cfg_args(ap)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tasks", nargs="+", default=["recon", "predict", "retrieval", "probe"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = CFG.load(args.config, args.set)
    ckpt = args.ckpt or os.path.join(base.train.out_dir, "perturbenergy.pt")
    model, cfg, ck = load_model(ckpt)
    cfg.data = base.data; cfg.eval = base.eval                     # allow eval-side overrides
    c2i, a2i, n_cond, n_pert = load_vocabs()
    ecfg = dict(cfg.eval)
    print(f"loaded {ckpt} (step {ck['step']}) | z_b={model.z_b_dim} z_a={model.z_a_dim} dev={DEV}")

    Xva, ova = read_cells(cfg.data.val_emb)
    cva, ava = ids_of(ova, cfg, c2i, a2i)
    res = {"ckpt": ckpt, "step": ck["step"], "tasks": args.tasks}

    if "recon" in args.tasks:
        k = (cva >= 0) & (ava >= 0)
        res["recon"] = task_recon(model, Xva[k], cva[k], ava[k])
        print("[recon]", {k2: round(v, 4) for k2, v in res["recon"].items() if isinstance(v, float)})
    if "predict" in args.tasks:
        s, rows = task_predict(model, cfg, Xva, ova, cva, ava, ecfg)
        res["predict"] = s; res["predict_per_group"] = rows
        print("\n=== PERTURBATION PREDICTION (avg over groups) ===")
        for k2, v in s.items():
            print(f"  {k2:20s}: {v:.4f}" if isinstance(v, float) else f"  {k2:20s}: {v}")
    if "retrieval" in args.tasks:
        res["retrieval"] = task_retrieval(model, cfg, Xva, ova, cva, ava, n_pert,
                                          n_cells=ecfg.get("retrieval_cells", 4000))
        print("\n=== ENERGY-BASED DRUG RETRIEVAL ===")
        for k2, v in res["retrieval"].items():
            print(f"  {k2:20s}: {v:.4f}" if isinstance(v, float) else f"  {k2:20s}: {v}")
    if "probe" in args.tasks:
        Xtr, otr = read_cells(cfg.data.train_emb, ecfg.get("probe_cells"))
        ctr, atr = ids_of(otr, cfg, c2i, a2i)
        res["probe"] = task_probe(model, cfg, Xtr, otr, ctr, atr, Xva, ova, cva, ava, n_pert)
        print("\n=== LATENT PROBES / PP-SNR ===")
        for k2, v in res["probe"].items():
            print(f"  {k2}: {v}")

    out = args.out or os.path.join(base.train.out_dir, "eval_results.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
