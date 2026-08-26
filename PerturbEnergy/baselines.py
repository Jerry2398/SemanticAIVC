"""
Single-latent baselines for the SAME perturbation-prediction protocol.

Both baselines use ONE unified latent z (no z_b / z_a split) with the same MLP
encoder/decoder shape as PerturbEnergy (hidden, depth, fusion), so only the latent
structure and the objective differ:

  --style vae    q(z|x,c) -> z -> p(x|z); recon = mse+mmd, KL weight = ours (5e-5).
                 "plain Gaussian VAE", architecture-matched to our encoder/decoder.
  --style scgen  faithful scGen: BatchNorm + LeakyReLU + dropout 0.2 and scGen's own
                 objective 0.5*SSE + 0.5*kl_weight*KL (no MMD term).

Neither has a perturbation latent, so prediction uses scGen's LATENT VECTOR ARITHMETIC:
    delta_a = mean_TRAIN(z | drug = a) - mean_TRAIN(z | drug = control)
    z_pred  = z(control cell) + delta_a   ->  decode
The deltas come from TRAIN cells only (no leakage). Evaluation reuses the exact groups,
control baseline, DEG definition and metrics of PerturbEnergy/evaluate.py.

  python baselines.py --style vae   --set train.max_steps=100000
  python baselines.py --style scgen --set train.max_steps=100000
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

import cfg as CFG
from data import load_vocabs, lognorm, make_stream, to_ids
from evaluate import _const, read_cells, ids_of
from losses import kl_standard, mse_recon, recon_loss
from models import Fuse, GaussianHead, mlp, reparameterize

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STYLE = {"vae": dict(norm="layer", act="silu", dropout=0.1),
         "scgen": dict(norm="batch", act="leakyrelu", dropout=0.2)}


class SingleZVAE(nn.Module):
    def __init__(self, n_genes, n_cond, z_dim, hidden, depth, style, fusion, fuse_dim, cond_dim):
        super().__init__()
        s = STYLE[style]
        self.z_dim = z_dim
        self.cond_emb = nn.Embedding(n_cond, cond_dim)
        self.fuse_e = Fuse([n_genes, cond_dim], fusion, fuse_dim)
        self.enc = GaussianHead(self.fuse_e.out_dim, z_dim, hidden, depth, s["dropout"], s["norm"])
        self.dec = mlp(z_dim, hidden, depth, s["dropout"], s["norm"], out_dim=n_genes, act=s["act"])

    def encode(self, x, c, sample=True):
        mu, lv = self.enc(self.fuse_e(x, self.cond_emb(c)))
        return (reparameterize(mu, lv) if sample else mu), mu, lv

    def forward(self, x, c, sample=True):
        z, mu, lv = self.encode(x, c, sample)
        return self.dec(z), mu, lv


def train(model, cfg, style, c2i, a2i, out):
    L, O = cfg.loss, cfg.optim
    tr = make_stream("train", cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=O.vae_lr, weight_decay=O.weight_decay)
    cf = open(os.path.join(out, "metrics.csv"), "w"); cw = csv.writer(cf)
    cw.writerow(["step", "loss", "recon_mse", "kl"]); cf.flush()
    acc = dict(loss=0.0, mse=0.0, kl=0.0, n=0)
    step, epoch = 0, 0
    while step < cfg.train.max_steps:
        tr.set_epoch(epoch)
        for counts, cond in DataLoader(tr, batch_size=None, num_workers=cfg.data.num_workers):
            model.train()
            x = lognorm(counts.to(DEV))
            c, _, keep = to_ids(cond, cfg, c2i, a2i, DEV)
            if keep.sum() < 8:
                continue
            x, c = x[keep], c[keep]
            xh, mu, lv = model(x, c)
            kl = kl_standard(mu, lv)
            if style == "scgen":                       # scGen's own objective
                loss = 0.5 * mse_recon(x, xh) + 0.5 * L.beta_kl_b * kl
                mse_v = mse_recon(x, xh).detach()
            else:                                      # ours: same recon as PerturbEnergy
                rec, mse_v, _ = recon_loss(x, xh, L.recon, L.mmd_weight, tuple(L.mmd_scales))
                loss = rec + L.beta_kl_b * kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), O.grad_clip); opt.step()
            acc["loss"] += float(loss); acc["mse"] += float(mse_v); acc["kl"] += float(kl); acc["n"] += 1
            step += 1
            if step % cfg.train.eval_every == 0 or step >= cfg.train.max_steps:
                A = {k: acc[k] / max(1, acc["n"]) for k in ("loss", "mse", "kl")}
                print(f"step {step:6d} | loss {A['loss']:.1f} mse {A['mse']:.1f} kl {A['kl']:.1f}", flush=True)
                cw.writerow([step, A["loss"], A["mse"], A["kl"]]); cf.flush()
                torch.save({"model": model.state_dict(), "cfg": dict(cfg), "style": style,
                            "step": step}, os.path.join(out, "baseline.pt"))
                acc = dict(loss=0.0, mse=0.0, kl=0.0, n=0)
            if step >= cfg.train.max_steps:
                break
        epoch += 1


@torch.no_grad()
def encode_np(model, X, c, bs=4096):
    return np.concatenate([model.encode(torch.from_numpy(X[s:s+bs]).to(DEV),
                                        torch.from_numpy(c[s:s+bs]).to(DEV), sample=False)[0].cpu().numpy()
                           for s in range(0, len(X), bs)])


def evaluate(model, cfg, c2i, a2i, out):
    """scGen-style latent arithmetic on the same 60 groups / baseline / DEGs as our models."""
    model.eval()
    ctl = cfg.data.control_label
    Xtr, otr = read_cells(cfg.data.train_emb)
    ctr, _ = ids_of(otr, cfg, c2i, a2i)
    ztr = encode_np(model, Xtr, ctr)
    dtr = otr[cfg.data.pert_col].astype(str).values
    z_ctl_mean = ztr[dtr == ctl].mean(0)
    delta = {d: ztr[dtr == d].mean(0) - z_ctl_mean for d in np.unique(dtr) if d != ctl}
    print(f"[latent arithmetic] deltas for {len(delta)} drugs from {len(ztr)} TRAIN cells")

    X, obs = read_cells(cfg.data.val_emb)
    c, _ = ids_of(obs, cfg, c2i, a2i)
    cl = obs[cfg.data.cond_col].astype(str).values
    dr = obs[cfg.data.pert_col].astype(str).values
    is_ctl = dr == ctl
    base = {k: X[(cl == k) & is_ctl].mean(0) for k in np.unique(cl) if ((cl == k) & is_ctl).sum() > 0}
    keys = np.array([f"{u}|{v}" for u, v in zip(cl, dr)])
    ev = (~is_ctl) & np.array([k in base for k in cl])
    uniq, cnt = np.unique(keys[ev], return_counts=True); o = np.argsort(-cnt)
    E = dict(cfg.eval)
    groups = [u for u, n in zip(uniq[o], cnt[o]) if n >= E["min_cells"]][:E["max_groups"]]

    def pr(u, v):
        return float(np.corrcoef(u, v)[0, 1]) if np.std(u) * np.std(v) > 0 else float("nan")

    rows = []
    for g in groups:
        m = (keys == g) & ev
        cln, drug = cl[m][0], dr[m][0]
        if drug not in delta:
            continue
        b, mu_real = base[cln], X[m].mean(0)
        idx = np.where((cl == cln) & is_ctl)[0][:E["n_per_group"]]
        with torch.no_grad():
            z0 = encode_np(model, X[idx], c[idx])
            zp = torch.from_numpy(z0 + delta[drug][None]).to(DEV)
            mu_pred = model.dec(zp).cpu().numpy().mean(0)
            mu_ceil = model.dec(torch.from_numpy(encode_np(model, X[m], c[m])).to(DEV)).cpu().numpy().mean(0)
        d_r, d_p, d_c = mu_real - b, mu_pred - b, mu_ceil - b
        deg = np.argsort(np.abs(d_r))[-E["top_degs"]:]
        rows.append(dict(group=g, n_real=int(m.sum()),
                         r2_mean_all=r2_score(mu_real, mu_pred), pearson_mean_all=pr(mu_real, mu_pred),
                         r2_mean_deg=r2_score(mu_real[deg], mu_pred[deg]),
                         pearson_mean_deg=pr(mu_real[deg], mu_pred[deg]),
                         r2_delta_all=r2_score(d_r, d_p), pearson_delta_all=pr(d_r, d_p),
                         r2_delta_deg=r2_score(d_r[deg], d_p[deg]),
                         pearson_delta_deg=pr(d_r[deg], d_p[deg]),
                         r2_mean_all_vae=r2_score(mu_real, mu_ceil),
                         r2_delta_deg_vae=r2_score(d_r[deg], d_c[deg])))
    ks = [k for k in rows[0] if k not in ("group", "n_real")]
    summary = {"n_groups": len(rows), "method": "latent arithmetic (scGen)",
               **{k: float(np.nanmean([r[k] for r in rows])) for k in ks}}
    print("\n=== PERTURBATION PREDICTION (avg over groups) ===")
    for k in ks:
        print(f"  {k:20s}: {summary[k]:.4f}")
    json.dump({"predict": summary, "per_group": rows}, open(os.path.join(out, "eval_results.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    CFG.add_cfg_args(ap)
    ap.add_argument("--style", choices=["vae", "scgen"], required=True)
    ap.add_argument("--z_dim", type=int, default=None, help="default: z_b_dim + z_a_dim")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    cfg = CFG.load(args.config, args.set)
    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)
    out = cfg.train.out_dir; os.makedirs(out, exist_ok=True)

    c2i, a2i, n_cond, _ = load_vocabs()
    z_dim = args.z_dim or (cfg.model.z_b_dim + cfg.model.z_a_dim)
    n_genes = len(make_stream("val", cfg).hvg_tokens)
    model = SingleZVAE(n_genes, n_cond, z_dim, cfg.model.hidden, cfg.model.depth, args.style,
                       cfg.model.fusion, cfg.model.fuse_dim, cfg.model.cond_emb_dim).to(DEV)
    print(f"{args.style} baseline | z_dim={z_dim} hidden={cfg.model.hidden} depth={cfg.model.depth} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M dev={DEV}", flush=True)
    if args.eval_only:
        model.load_state_dict(torch.load(os.path.join(out, "baseline.pt"), map_location=DEV)["model"])
    else:
        train(model, cfg, args.style, c2i, a2i, out)
    evaluate(model, cfg, c2i, a2i, out)
    print("BASELINE_DONE ->", out)


if __name__ == "__main__":
    main()
