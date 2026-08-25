"""
Energy-based Population Perturbation SNR (PP-SNR_E) -- per PDF section 2/7.

For each perturbation p in context c=(cell_line, plate) that has a matched
control (DMSO_TF, same cell_line & plate):
    X = perturbation-population representation,  Y = control-population rep.
    delta_XY = E||x-y||^2 ; sigma_X = E||x-x'||^2 ; sigma_Y = E||y-y'||^2
    PP-SNR_E = [delta_XY - .5(sigma_X+sigma_Y)] / [.5(sigma_X+sigma_Y) + eps]

Distance d = squared Euclidean in CONTROL-standardized space (z~=(z-mu_ctrl)/
(sigma_ctrl+eps), stats from matched control only; each representation
standardized independently -> fair cross-space comparison, PDF 7.1). With
squared Euclidean this has an exact closed form from per-group sufficient
statistics (count, sum z, sum z^2):
    numerator = ||(mu_X-mu_Y)/sigma_ctrl||^2                    (standardized mean shift^2)
    denom     = sum_g (Var_X_g + Var_Y_g)/sigma_ctrl_g^2        (within-condition variance)
so the whole thing streams -- no O(n^2) pairwise, covers every perturbation.

Computed in EXPRESSION (5000-HVG log-norm) and each VAE LATENT, on the SAME cells
(each cell contributes to all representations). Train and val computed separately.

  python energy_ppsnr.py --split train
  python energy_ppsnr.py --split val
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vae"))

import numpy as np
import scipy.sparse as sp
import torch

import tahoe_common as T

MODELS = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models"
OUT = os.path.dirname(os.path.abspath(__file__))
# representation name -> VAE checkpoint dir (None = raw expression)
VAE_SET = [("gauss", "tahoe_vae"), ("nb", "tahoe_vae_nb"),
           ("scgen", "tahoe_vae_scgen"), ("drugsup", "tahoe_vae_auxmlp100")]
STD_EPS = 1e-2       # floor on control per-gene std (guards silent genes)
SNR_EPS = 1e-8


def build_vae(name):
    ck = torch.load(os.path.join(MODELS, name, "vae.pt"), map_location="cpu")
    a, lik, ng = ck["args"], ck.get("likelihood", "gaussian"), ck["n_genes"]
    if lik == "nb":
        from model_nb import NBVAE; m = NBVAE(ng, a["latent_dim"], a["hidden"], a["depth"])
    elif lik == "gaussian_scgen":
        from model_scgen import SCGenLikeVAE; m = SCGenLikeVAE(ng, a["latent_dim"], a["hidden"], a["depth"], a.get("dropout", 0.2))
    else:
        from model import SimpleVAE; m = SimpleVAE(ng, a["latent_dim"], a["hidden"], a["depth"])
    m.load_state_dict(ck["model"]); m.eval()
    return m


def lognorm(counts, target=1e4):
    lib = counts.sum(1, keepdims=True).clip(min=1.0)
    return np.log1p(counts / lib * target).astype(np.float32)


class Accum:
    """Per-group sufficient stats (count, sum, sumsq) for one representation."""
    def __init__(self, dim, maxg):
        self.cnt = np.zeros(maxg, np.float64)
        self.s = np.zeros((maxg, dim), np.float64)
        self.ss = np.zeros((maxg, dim), np.float64)

    def add(self, onehot, Z):
        # onehot: csr [G_local, n]; Z: [n, dim]; gids handled by caller via slot map
        return onehot @ Z, onehot @ (Z * Z), np.asarray(onehot.sum(1)).ravel()


def run(split, vaes, hvg, n_vocab, shard_budget, n_min, dev, bs=8192):
    reps = ["expr"] + [n for n, _ in vaes]
    dims = {"expr": len(hvg), **{n: m.latent_dim for n, m in vaes}}
    MAXP, MAXC = 22000, 1500
    P = {r: Accum(dims[r], MAXP) for r in reps}
    C = {r: Accum(dims[r], MAXC) for r in reps}
    pid, cid, pplate = {}, {}, {}

    shards = T.list_shards(split)
    np.random.default_rng(0).shuffle(shards)          # spread the capped budget across all plates
    shards = shards[:shard_budget]
    for si, path in enumerate(shards):
        X, obs = T.shard_to_csr(path, n_vocab, want_obs=True)
        expr = lognorm(X[:, hvg].toarray())
        cl = obs["cell_line_id"].astype(str).values
        dr = obs["drug"].astype(str).values
        pl = obs["plate"].astype(str).values
        Zr = {"expr": expr}
        with torch.no_grad():
            xt = torch.from_numpy(expr).to(dev)
            for name, m in vaes:
                outs = []
                for b in range(0, len(xt), bs):
                    mu, _ = m.encode(xt[b:b + bs]); outs.append(mu.cpu().numpy())
                Zr[name] = np.concatenate(outs).astype(np.float64)
        Zr["expr"] = expr.astype(np.float64)

        isc = dr == "DMSO_TF"
        for mask, keyf, idmap, store, track in [
                (~isc, lambda i: (cl[i], dr[i]), pid, P, True),
                (isc, lambda i: (cl[i], pl[i]), cid, C, False)]:
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            gid = np.empty(len(idx), np.int64)
            for j, i in enumerate(idx):
                k = keyf(i)
                if k not in idmap:
                    idmap[k] = len(idmap)
                gid[j] = idmap[k]
                if track and k not in pplate:
                    pplate[k] = pl[i]
            # local factorize -> sparse one-hot -> scatter-add
            uniq, inv = np.unique(gid, return_inverse=True)
            oh = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                               shape=(len(uniq), len(idx)))
            cnt = np.asarray(oh.sum(1)).ravel()
            for r in reps:
                Zsub = Zr[r][idx]
                store[r].s[uniq] += oh @ Zsub
                store[r].ss[uniq] += oh @ (Zsub * Zsub)
                store[r].cnt[uniq] += cnt
        if si % 200 == 0:
            print(f"[{split}] shard {si}/{len(shards)} | perts={len(pid)} ctrls={len(cid)}", flush=True)

    # --- compute PP-SNR per perturbation with matched control ---
    per_rep = {r: [] for r in reps}
    meta = []
    for k, p in pid.items():
        cl_, drug = k
        ck = (cl_, pplate[k])
        if ck not in cid:
            continue
        c = cid[ck]
        nX, nY = P["expr"].cnt[p], C["expr"].cnt[c]
        if nX < n_min or nY < n_min:
            continue
        row = {"cell_line": cl_, "drug": drug, "plate": pplate[k], "n_pert": int(nX), "n_ctrl": int(nY)}
        for r in reps:
            muX = P[r].s[p] / nX; muY = C[r].s[c] / nY
            varX = (P[r].ss[p] - nX * muX ** 2) / max(nX - 1, 1)
            varY = (C[r].ss[c] - nY * muY ** 2) / max(nY - 1, 1)
            sc2 = (np.sqrt(np.clip(varY, 0, None)) + STD_EPS) ** 2      # (sigma_ctrl+eps)^2
            num = float(np.sum((muX - muY) ** 2 / sc2))                # ||Delta mu / sigma_ctrl||^2
            den = float(np.sum((np.clip(varX, 0, None) + np.clip(varY, 0, None)) / sc2))
            snr = num / (den + SNR_EPS)
            per_rep[r].append(snr)
            row[f"snr_{r}"] = snr
            row[f"energy_{r}"] = 2.0 * num                              # squared-euclidean energy distance
        meta.append(row)

    summary = {"split": split, "n_perturbations": len(meta), "n_min": n_min,
               "shards_used": len(shards), "std_eps": STD_EPS,
               "mean_ppsnr": {r: float(np.mean(per_rep[r])) if per_rep[r] else None for r in reps},
               "median_ppsnr": {r: float(np.median(per_rep[r])) if per_rep[r] else None for r in reps}}
    json.dump({"summary": summary, "per_perturbation": meta},
              open(os.path.join(OUT, f"ppsnr_{split}.json"), "w"), indent=2)
    print(f"\n=== PP-SNR ({split}) over {len(meta)} perturbations ===")
    for r in reps:
        print(f"  {r:8s}: mean {summary['mean_ppsnr'][r]:.4f}  median {summary['median_ppsnr'][r]:.4f}")
    print(f"wrote ppsnr_{split}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--shard_budget", type=int, default=100000)
    ap.add_argument("--n_min", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hvg = T.load_hvg()["token_ids"]; n_vocab = T.n_vocab()
    vaes = [(n, build_vae(ck).to(dev)) for n, ck in VAE_SET]
    print(f"split={args.split} reps=expr+{[n for n,_ in vaes]} device={dev} n_min={args.n_min}")
    run(args.split, vaes, hvg, n_vocab, args.shard_budget, args.n_min, dev)


if __name__ == "__main__":
    main()
