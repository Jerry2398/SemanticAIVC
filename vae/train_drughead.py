"""
Gaussian VAE + a DRUG-supervised auxiliary head on the latent mu. Unified across
head types so every experiment shares an IDENTICAL setup (only --aux_mode and
--aux_weight differ) -> maximally comparable.

  L = recon + beta*KL + lambda * (L_aux / c_mode)

--aux_mode:
  ce_linear : Linear(32->380), CrossEntropy on drug identity          (c=1)   [baseline aux]
  ce_mlp    : Linear(32->256)->ReLU->Dropout->Linear(256->380), CE    (c=1)   [dir.2 nonlinear]
  ecfp      : Linear(32->2048), cosine loss to the drug's ECFP fp     (c=1)   [dir.1 regression]
  infonce   : f(mu)&g(ECFP)->128, in-batch InfoNCE (drug negatives)   (c=lnU) [dir.1 contrastive]

ce_* keep raw CE (comparable to the existing linear-identity runs). ecfp/infonce
are normalized to ~O(1) init so lambda is comparable BETWEEN the two SMILES
experiments. Everything else (SimpleVAE, recon=SSE, beta*KL, AdamW lr1e-3, 532k
steps, warmup5000, latent 32) is identical to train.py. The saved vae.pt is a
PURE SimpleVAE (head saved separately) -> extractor/LDM/decoder work unchanged.
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import SimpleVAE
from common import lognorm, collect_recon, reconstruction_metrics, generative_mean_match
from tahoe_dataset import TahoeStream
from train import plot_curve

TD = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe"
IGNORE = -100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux_mode", required=True, choices=["ce_linear", "ce_mlp", "ecfp", "infonce"])
    ap.add_argument("--aux_weight", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kl_warmup", type=int, default=5000)
    ap.add_argument("--proj", type=int, default=128, help="InfoNCE projection dim")
    ap.add_argument("--tau", type=float, default=0.1, help="InfoNCE temperature")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=532000)
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    vocab = json.load(open(f"{TD}/drug_vocab.json"))
    drug_to_id, n_drugs = vocab["drug_to_id"], vocab["n_drugs"]
    ecfp = torch.tensor(np.load(f"{TD}/drug_ecfp2048.npy"), device=dev)          # [380,2048]
    ecfp_valid = torch.tensor(np.load(f"{TD}/drug_ecfp2048_valid.npy"), device=dev)
    print(f"[{args.aux_mode}] lambda={args.aux_weight} n_drugs={n_drugs} dev={dev} steps={args.max_steps}")

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    csv_f = open(csv_path, "w"); cw = csv.writer(csv_f)
    cw.writerow(["step", "train_loss", "recon", "kl", "aux_raw", "aux_metric", "beta",
                 "val_recon_mse", "val_gene_pearson", "val_cell_pearson", "val_r2", "val_gen_mean_pearson"])
    csv_f.flush()

    tr = TahoeStream(split="train", batch_size=args.batch_size, cond_cols=("drug",))
    va = TahoeStream(split="val", batch_size=args.batch_size, shuffle=False, cond_cols=())
    n_genes = len(tr.hvg_tokens)
    model = SimpleVAE(n_genes, args.latent_dim, args.hidden, args.depth).to(dev)

    # --- head(s) per mode ---
    ld = args.latent_dim
    if args.aux_mode == "ce_linear":
        head = nn.Linear(ld, n_drugs).to(dev); params = list(head.parameters())
    elif args.aux_mode == "ce_mlp":
        head = nn.Sequential(nn.Linear(ld, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, n_drugs)).to(dev)
        params = list(head.parameters())
    elif args.aux_mode == "ecfp":
        head = nn.Linear(ld, 2048).to(dev); params = list(head.parameters())
    else:  # infonce
        f_cell = nn.Linear(ld, args.proj).to(dev)
        g_drug = nn.Linear(2048, args.proj).to(dev)
        params = list(f_cell.parameters()) + list(g_drug.parameters())

    opt = torch.optim.AdamW(list(model.parameters()) + params, lr=args.lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    def drug_ids(cond):
        return torch.tensor([drug_to_id.get(str(d), IGNORE) for d in cond["drug"]],
                            dtype=torch.long, device=dev)

    def aux_step(mu, y):
        """returns (aux_loss_normalized, raw_loss_value, metric_value)."""
        if args.aux_mode in ("ce_linear", "ce_mlp"):
            logits = head(mu)
            raw = ce(logits, y)
            valid = y != IGNORE
            acc = (logits.argmax(1)[valid] == y[valid]).float().mean().item() if valid.any() else 0.0
            return raw, raw.item(), acc                                   # c=1
        valid = (y != IGNORE) & ecfp_valid[y.clamp(min=0)]
        if valid.sum() < 2:
            z = mu.sum() * 0.0
            return z, 0.0, 0.0
        muv, yv = mu[valid], y[valid]
        if args.aux_mode == "ecfp":
            pred = head(muv)
            tgt = ecfp[yv]
            cos = F.cosine_similarity(pred, tgt, dim=1)
            raw = (1.0 - cos).mean()
            return raw, raw.item(), cos.mean().item()                     # c=1
        # infonce: in-batch drug negatives
        uniq, inv = torch.unique(yv, return_inverse=True)                 # inv -> target index
        zc = F.normalize(f_cell(muv), dim=1)                              # [b,proj]
        zd = F.normalize(g_drug(ecfp[uniq]), dim=1)                       # [U,proj]
        logits = zc @ zd.t() / args.tau                                   # [b,U]
        raw = ce(logits, inv)
        c = math.log(len(uniq)) if len(uniq) > 1 else 1.0
        acc = (logits.argmax(1) == inv).float().mean().item()
        return raw / c, raw.item(), acc

    step, epoch, done = 0, 0, False
    run = {k: 0.0 for k in ("loss", "recon", "kl", "aux", "met", "n")}
    while not done:
        tr.set_epoch(epoch)
        loader = DataLoader(tr, batch_size=None, num_workers=args.num_workers)
        for counts, cond in loader:
            model.train()
            x = lognorm(counts.to(dev)); y = drug_ids(cond)
            beta = args.beta * min(1.0, step / max(1, args.kl_warmup))
            mu, lv = model.encode(x)
            z = SimpleVAE.reparameterize(mu, lv)
            xh = model.decode(z)
            vae_loss, recon, kl = SimpleVAE.loss(x, xh, mu, lv, beta)
            aux_norm, aux_raw, metric = aux_step(mu, y)
            loss = vae_loss + args.aux_weight * aux_norm
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + params, 5.0)
            opt.step()
            run["loss"] += loss.item(); run["recon"] += recon.item(); run["kl"] += kl.item()
            run["aux"] += aux_raw; run["met"] += metric; run["n"] += 1
            step += 1

            if step % args.eval_every == 0 or step >= args.max_steps:
                X, Xh = collect_recon(model, DataLoader(va, batch_size=None, num_workers=0), dev, max_batches=20)
                m = reconstruction_metrics(X, Xh); g = generative_mean_match(model, X.mean(0), dev)
                a = {k: run[k] / max(1, run["n"]) for k in ("loss", "recon", "kl", "aux", "met")}
                print(f"step {step:6d} ep {epoch} | loss {a['loss']:.2f} recon {a['recon']:.2f} kl {a['kl']:.2f} "
                      f"aux_raw {a['aux']:.3f} aux_metric {a['met']:.3f} beta {beta:.2f} "
                      f"|| val R2 {m['r2']:.4f} cell_r {m['cell_pearson']:.4f}", flush=True)
                cw.writerow([step, a["loss"], a["recon"], a["kl"], a["aux"], a["met"], beta,
                             m["recon_mse"], m["gene_pearson"], m["cell_pearson"], m["r2"], g["gen_mean_pearson"]])
                csv_f.flush()
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "n_genes": n_genes, "step": step}, os.path.join(args.out_dir, "vae.pt"))
                torch.save({"params": [p.detach().cpu() for p in params], "aux_mode": args.aux_mode,
                            "aux_weight": args.aux_weight}, os.path.join(args.out_dir, "head.pt"))
                plot_curve(csv_path, os.path.join(args.out_dir, "loss_curve.png"))
                run = {k: 0.0 for k in run}
            if step >= args.max_steps:
                done = True; break
        epoch += 1
    print("TAHOE_DRUGHEAD_DONE ->", args.out_dir)


if __name__ == "__main__":
    main()
