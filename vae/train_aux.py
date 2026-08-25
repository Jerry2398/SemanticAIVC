"""
Gaussian VAE + supervised auxiliary DRUG head (constraint on the latent).

Identical to train.py (SimpleVAE, recon=SSE, beta*KL, AdamW lr 1e-3, 532k steps,
KL warmup 5000) EXCEPT a linear head predicts drug identity from the posterior
mean mu, adding lambda * CE to the loss:

    L = recon + beta*KL + lambda * CE(Linear(mu), drug_id)

The head is EXTERNAL to the VAE (kept in the train loop), so the saved vae.pt is
a pure SimpleVAE -> the extractor / LDM / decoder adapters work unchanged; the
head is a train-time regularizer only (saved separately for the record).

  python train_aux.py --aux_weight 20  --out_dir .../models/tahoe_vae_aux20
  python train_aux.py --aux_weight 50  --out_dir .../models/tahoe_vae_aux50
  python train_aux.py --aux_weight 100 --out_dir .../models/tahoe_vae_aux100
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SimpleVAE
from common import lognorm, collect_recon, reconstruction_metrics, generative_mean_match
from tahoe_dataset import TahoeStream
from train import plot_curve

OUT_DIR = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models/tahoe_vae_aux"
DRUG_VOCAB = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/tahoe/drug_vocab.json"
IGNORE = -100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kl_warmup", type=int, default=5000)
    ap.add_argument("--aux_weight", type=float, required=True, help="lambda on the drug-CE term")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=532000)
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    vocab = json.load(open(DRUG_VOCAB))
    drug_to_id, n_drugs = vocab["drug_to_id"], vocab["n_drugs"]
    print(f"[aux] lambda={args.aux_weight} n_drugs={n_drugs} device={dev} max_steps={args.max_steps}")

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    csv_f = open(csv_path, "w"); cw = csv.writer(csv_f)
    cw.writerow(["step", "train_loss", "recon", "kl", "aux_ce", "aux_acc", "beta",
                 "val_recon_mse", "val_gene_pearson", "val_cell_pearson", "val_r2", "val_gen_mean_pearson"])
    csv_f.flush()

    # drug labels come from the stream -> cond_cols=("drug",)
    tr = TahoeStream(split="train", batch_size=args.batch_size, cond_cols=("drug",))
    va = TahoeStream(split="val", batch_size=args.batch_size, shuffle=False, cond_cols=())
    n_genes = len(tr.hvg_tokens)

    model = SimpleVAE(n_genes, args.latent_dim, args.hidden, args.depth).to(dev)
    aux_head = nn.Linear(args.latent_dim, n_drugs).to(dev)          # linear head on mu
    opt = torch.optim.AdamW(list(model.parameters()) + list(aux_head.parameters()),
                            lr=args.lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    def drug_ids(cond):
        return torch.tensor([drug_to_id.get(str(d), IGNORE) for d in cond["drug"]],
                            dtype=torch.long, device=dev)

    step, epoch, done = 0, 0, False
    run = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "aux": 0.0, "acc": 0.0, "n": 0}
    while not done:
        tr.set_epoch(epoch)
        loader = DataLoader(tr, batch_size=None, num_workers=args.num_workers)
        for counts, cond in loader:
            model.train(); aux_head.train()
            x = lognorm(counts.to(dev))
            y = drug_ids(cond)
            beta = args.beta * min(1.0, step / max(1, args.kl_warmup))
            mu, lv = model.encode(x)
            z = SimpleVAE.reparameterize(mu, lv)
            xh = model.decode(z)
            vae_loss, recon, kl = SimpleVAE.loss(x, xh, mu, lv, beta)
            logits = aux_head(mu)                                   # predict drug from mu
            aux_ce = ce(logits, y)
            loss = vae_loss + args.aux_weight * aux_ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(aux_head.parameters()), 5.0)
            opt.step()

            valid = y != IGNORE
            acc = (logits.argmax(1)[valid] == y[valid]).float().mean().item() if valid.any() else 0.0
            run["loss"] += loss.item(); run["recon"] += recon.item(); run["kl"] += kl.item()
            run["aux"] += aux_ce.item(); run["acc"] += acc; run["n"] += 1
            step += 1

            if step % args.eval_every == 0 or step >= args.max_steps:
                X, Xh = collect_recon(model, DataLoader(va, batch_size=None, num_workers=0), dev, max_batches=20)
                m = reconstruction_metrics(X, Xh)
                g = generative_mean_match(model, X.mean(0), dev)
                a = {k: run[k] / max(1, run["n"]) for k in ("loss", "recon", "kl", "aux", "acc")}
                print(f"step {step:6d} ep {epoch} | loss {a['loss']:.2f} recon {a['recon']:.2f} kl {a['kl']:.2f} "
                      f"aux_ce {a['aux']:.3f} aux_acc {a['acc']:.3f} beta {beta:.2f} "
                      f"|| val R2 {m['r2']:.4f} cell_r {m['cell_pearson']:.4f} gen {g['gen_mean_pearson']:.4f}", flush=True)
                cw.writerow([step, a["loss"], a["recon"], a["kl"], a["aux"], a["acc"], beta,
                             m["recon_mse"], m["gene_pearson"], m["cell_pearson"], m["r2"], g["gen_mean_pearson"]])
                csv_f.flush()
                # vae.pt is a PURE SimpleVAE (downstream-compatible); head saved separately
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "n_genes": n_genes, "step": step}, os.path.join(args.out_dir, "vae.pt"))
                torch.save({"aux_head": aux_head.state_dict(), "aux_weight": args.aux_weight,
                            "n_drugs": n_drugs}, os.path.join(args.out_dir, "aux_head.pt"))
                plot_curve(csv_path, os.path.join(args.out_dir, "loss_curve.png"))
                run = {k: 0.0 for k in run}
            if step >= args.max_steps:
                done = True; break
        epoch += 1
    print("TAHOE_VAE_AUX_DONE ->", args.out_dir)


if __name__ == "__main__":
    main()
