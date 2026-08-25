"""
Train the scGen-ALIGNED Gaussian VAE on Tahoe-100M (streaming, out-of-core).

Same data / step budget / optimizer / lr / eval as train.py; only the scGen
settings differ (BatchNorm, LeakyReLU, dropout 0.2, loss = 0.5*SSE +
0.5*kl_weight*KL with kl_weight=5e-5). NN dimensions kept (latent 32/hidden
512/depth 2). Metrics computed in log-norm space -> comparable to tahoe_vae.

  python train_scgen.py --latent_dim 32 --max_steps 532000 --eval_every 10000

All outputs -> /scratch/.../models/tahoe_vae_scgen/ (NOT the project dir).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))

import torch
from torch.utils.data import DataLoader

from model_scgen import SCGenLikeVAE
from common import lognorm, collect_recon, reconstruction_metrics, generative_mean_match
from tahoe_dataset import TahoeStream
from train import plot_curve                      # identical loss-curve plotter

OUT_DIR = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models/tahoe_vae_scgen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2, help="scGen wrapper default")
    ap.add_argument("--kl_weight", type=float, default=5e-5, help="scGen SCGENVAE default")
    ap.add_argument("--kl_warmup", type=int, default=5000, help="steps to ramp kl_weight 0->kl_weight")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=532000)
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--num_workers", type=int, default=8)
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
    print(f"[scGen-aligned] n_genes(HVG)={n_genes} latent={args.latent_dim} device={dev} "
          f"max_steps={args.max_steps} kl_weight={args.kl_weight}")

    model = SCGenLikeVAE(n_genes, args.latent_dim, args.hidden, args.depth, args.dropout).to(dev)
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
            kl_w = args.kl_weight * min(1.0, step / max(1, args.kl_warmup))
            xh, mu, lv = model(x)
            loss, recon, kl = SCGenLikeVAE.loss(x, xh, mu, lv, kl_w)
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
                print(f"step {step:6d} ep {epoch} | loss {avg['loss']:.3f} recon {avg['recon']:.3f} kl {avg['kl']:.3f} kl_w {kl_w:.2e} "
                      f"|| val R2 {m['r2']:.4f} gene_r {m['gene_pearson']:.4f} cell_r {m['cell_pearson']:.4f} "
                      f"mse {m['recon_mse']:.4f} gen_meanR {g['gen_mean_pearson']:.4f}", flush=True)
                cw.writerow([step, avg["loss"], avg["recon"], avg["kl"], kl_w,
                             m["recon_mse"], m["gene_pearson"], m["cell_pearson"], m["r2"], g["gen_mean_pearson"]])
                csv_f.flush()
                torch.save({"model": model.state_dict(),
                            "args": vars(args), "n_genes": n_genes, "step": step, "likelihood": "gaussian_scgen"},
                           os.path.join(args.out_dir, "vae.pt"))
                plot_curve(csv_path, os.path.join(args.out_dir, "loss_curve.png"))
                run = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
            if step >= args.max_steps:
                done = True
                break
        epoch += 1
    print("TAHOE_VAE_SCGEN_DONE ->", args.out_dir)


if __name__ == "__main__":
    main()
