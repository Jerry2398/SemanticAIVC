"""
Evaluate a trained LDM in LATENT space (no decoder needed).

Two things:
  1. Distribution match: sample latents conditioned on the val cells' own
     conditions, then compare generated vs real val latents with RBF-MMD (lower
     = better) + mean/covariance error. Reported overall and per cell type.
  2. Condition fidelity: a logistic probe fit on REAL train latents -> cell_type,
     applied to GENERATED latents; accuracy = how well generation respects the
     requested condition (higher = better).

  SEMANTIC_DATASET=sciplex python evaluate.py --vae scvi --method flow --guidance 2.0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import config as C
from data import load_latents
from sample import load_ldm


def rbf_mmd(x, y, gamma=None):
    def sq(a, b):
        return (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T
    if gamma is None:
        d = sq(x[:500], x[:500])
        gamma = 1.0 / (np.median(d[d > 0]) + 1e-8)
    kxx, kyy, kxy = np.exp(-gamma * sq(x, x)), np.exp(-gamma * sq(y, y)), np.exp(-gamma * sq(x, y))
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--method", default="flow")
    ap.add_argument("--ldm_dir", default=None)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n_eval", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = args.ldm_dir or C.ldm_dir(args.vae, tag=args.method)
    net, diff, codec, scaler, meta = load_ldm(d, dev)

    tr_emb, va_emb = C.emb_paths(args.vae)
    z_tr, obs_tr = load_latents(tr_emb)
    z_va, obs_va = load_latents(va_emb)
    rng = np.random.default_rng(args.seed)
    if len(z_va) > args.n_eval:
        sel = rng.choice(len(z_va), args.n_eval, replace=False)
        z_va, obs_va = z_va[sel], obs_va.iloc[sel]
    n = len(z_va)

    # conditions taken from the real val cells
    cond_np = codec.encode(obs_va)
    cond = {k: (torch.as_tensor(v).long() if v.dtype.kind == "i" else torch.as_tensor(v).float()).to(dev)
            for k, v in cond_np.items()}
    cemb_c = net.embed(cond, drop=None)
    cemb_u = net.embedder.null_like(n, dev)
    g = args.guidance

    def predict(z_t, t):
        pc = net.denoise(z_t, t, cemb_c)
        return pc if g == 1.0 else net.denoise(z_t, t, cemb_u) + g * (pc - net.denoise(z_t, t, cemb_u))

    shape = (n, meta["latent_dim"])
    z_gen = (diff.ddim_sample(predict, shape, steps=args.steps) if meta["method"] == "ddpm"
             else diff.sample(predict, shape, steps=args.steps, device=dev))
    z_gen = scaler.inverse(z_gen.cpu().numpy().astype(np.float32))

    # 1. distribution match (native latent scale)
    print(f"=== LDM eval: {C.DATASET}/{args.vae} {meta['method']} guidance={g} n={n} ===")
    print(f"MMD (overall)        : {rbf_mmd(z_gen, z_va):.5f}")
    print(f"mean L2 error        : {np.linalg.norm(z_gen.mean(0) - z_va.mean(0)):.4f}")
    print(f"cov Frobenius error  : {np.linalg.norm(np.cov(z_gen.T) - np.cov(z_va.T)):.4f}")
    ct_col = C.LABELS["cell_type"]
    cts = obs_va[ct_col].astype(str).values
    for ct in sorted(np.unique(cts)):
        m = cts == ct
        if m.sum() >= 50:
            print(f"  MMD[{ct}]: {rbf_mmd(z_gen[m], z_va[m]):.5f}  (n={m.sum()})")

    # 2. condition fidelity (probe fit on REAL train latents -> cell_type)
    clf = LogisticRegression(max_iter=300, n_jobs=-1)
    clf.fit(z_tr, obs_tr[ct_col].astype(str).values)
    acc = accuracy_score(cts, clf.predict(z_gen))
    print(f"cell_type fidelity (probe acc on generated): {acc:.4f}  "
          f"(real-val probe acc {accuracy_score(cts, clf.predict(z_va)):.4f})")


if __name__ == "__main__":
    main()
