"""
Train the simple scGen-style VAE on Tahoe-100M (streaming, out-of-core).

Input = 5000-HVG raw counts from TahoeStream, log-normalized on the fly. Trains
by mini-batch SGD over shards (never loads all cells). Logs the loss curve
(metrics.csv + loss_curve.png) and periodically evaluates VAL reconstruction
(gene/cell Pearson, R2, MSE) + a generative mean-profile check.

  python train.py --latent_dim 32 --max_steps 30000 --eval_every 2000

All outputs -> /scratch/.../models/tahoe_vae/ (NOT the project dir).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SimpleVAE
from common import lognorm, collect_recon, reconstruction_metrics, generative_mean_match
from tahoe_dataset import TahoeStream

OUT_DIR = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models/tahoe_vae"


def plot_curve(csv_path, png_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = list(csv.DictReader(open(csv_path)))
        st = [int(r["step"]) for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(st, [float(r["train_loss"]) for r in rows]); ax[0].set_title("train loss"); ax[0].set_xlabel("step")
        ax[0].plot(st, [float(r["recon"]) for r in rows], label="recon"); ax[0].plot(st, [float(r["kl"]) for r in rows], label="kl"); ax[0].legend()
        vr = [(int(r["step"]), float(r["val_r2"])) for r in rows if r.get("val_r2")]
        vp = [(int(r["step"]), float(r["val_gene_pearson"])) for r in rows if r.get("val_gene_pearson")]
        if vr: ax[1].plot(*zip(*vr), marker="o"); ax[1].set_title("val R2"); ax[1].set_xlabel("step")
        if vp: ax[2].plot(*zip(*vp), marker="o"); ax[2].set_title("val gene-Pearson"); ax[2].set_xlabel("step")
        fig.tight_layout(); fig.savefig(png_path, dpi=110); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight (after warmup)")
    ap.add_argument("--kl_warmup", type=int, default=5000, help="steps to ramp beta 0->beta")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    csv_f = open(csv_path, "w")
    cw = csv.writer(csv_f)
    cw.writerow(["step", "train_loss", "recon", "kl", "beta",
                 "val_recon_mse", "val_gene_pearson", "val_cell_pearson", "val_r2", "val_gen_mean_pearson"])
    csv_f.flush()

    tr = TahoeStream(split="train", batch_size=args.batch_size, cond_cols=())
    va = TahoeStream(split="val", batch_size=args.batch_size, shuffle=False, cond_cols=())
    n_genes = len(tr.hvg_tokens)
    print(f"n_genes(HVG)={n_genes} latent={args.latent_dim} device={dev} max_steps={args.max_steps}")

    model = SimpleVAE(n_genes, args.latent_dim, args.hidden, args.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    step = 0
    epoch = 0
    run = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    done = False
    while not done:                       # outer epoch loop: an IterableDataset
        tr.set_epoch(epoch)               # exhausts after 1 pass -> re-iterate & reshuffle
        loader = DataLoader(tr, batch_size=None, num_workers=args.num_workers)
        for counts, _ in loader:
            model.train()
            x = lognorm(counts.to(dev))
            beta = args.beta * min(1.0, step / max(1, args.kl_warmup))
            xh, mu, lv = model(x)
            loss, recon, kl = SimpleVAE.loss(x, xh, mu, lv, beta)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run["loss"] += loss.item(); run["recon"] += recon.item(); run["kl"] += kl.item(); run["n"] += 1
            step += 1

            if step % args.eval_every == 0 or step >= args.max_steps:
                X, Xh = collect_recon(model, DataLoader(va, batch_size=None, num_workers=0), dev, max_batches=20)
                m = reconstruction_metrics(X, Xh)
                g = generative_mean_match(model, X.mean(0), dev)
                avg = {k: run[k] / max(1, run["n"]) for k in ("loss", "recon", "kl")}
                print(f"step {step:6d} ep {epoch} | loss {avg['loss']:.3f} recon {avg['recon']:.3f} kl {avg['kl']:.3f} beta {beta:.3f} "
                      f"|| val R2 {m['r2']:.4f} gene_r {m['gene_pearson']:.4f} cell_r {m['cell_pearson']:.4f} "
                      f"mse {m['recon_mse']:.4f} gen_meanR {g['gen_mean_pearson']:.4f}", flush=True)
                cw.writerow([step, avg["loss"], avg["recon"], avg["kl"], beta,
                             m["recon_mse"], m["gene_pearson"], m["cell_pearson"], m["r2"], g["gen_mean_pearson"]])
                csv_f.flush()
                torch.save({"model": model.state_dict(),
                            "args": vars(args), "n_genes": n_genes, "step": step},
                           os.path.join(args.out_dir, "vae.pt"))
                plot_curve(csv_path, os.path.join(args.out_dir, "loss_curve.png"))
                run = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
            if step >= args.max_steps:
                done = True
                break
        epoch += 1
    print("TAHOE_VAE_DONE ->", args.out_dir)


if __name__ == "__main__":
    main()
