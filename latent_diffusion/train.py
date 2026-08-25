"""
Train a conditional latent-diffusion model on a VAE's latent space.

  SEMANTIC_DATASET=sciplex python train.py --vae scvi --method flow --save
  SEMANTIC_DATASET=pbmc    python train.py --vae scgen --method ddpm --save

Trains ONLY on the saved latent embeddings + obs conditions (no VAE needed).
Latent dim is read from the embedding file. Classifier-free guidance via joint
condition dropout (--p_uncond). Checkpoint (net + latent scaler + condition
codec + config) is written to config.ldm_dir(vae).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import config as C
from data import load_latents, LatentScaler, CondCodec
from models.net import LDMNet
from diffusion.ddpm import GaussianDiffusion
from diffusion.flow import RectifiedFlow


def to_tensors(cond_np, device):
    out = {}
    for k, v in cond_np.items():
        t = torch.as_tensor(v)
        out[k] = t.long().to(device) if v.dtype.kind == "i" else t.float().to(device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True, help="which encoder's latent space (scvi/scgen/...)")
    ap.add_argument("--method", choices=["ddpm", "flow"], default="flow")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of condition names to use (default: all in config.CONDITION_SPEC)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p_uncond", type=float, default=0.1, help="CFG condition-drop prob")
    ap.add_argument("--num_steps", type=int, default=1000, help="DDPM timesteps")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    spec = C.CONDITION_SPEC
    if args.conditions:
        spec = [s for s in spec if s[0] in args.conditions]
    print(f"dataset={C.DATASET} vae={args.vae} method={args.method} "
          f"conditions={[s[0] for s in spec]} device={dev}")

    # --- data ---
    tr_emb, _ = C.emb_paths(args.vae)
    z, obs = load_latents(tr_emb)
    latent_dim = z.shape[1]
    scaler = LatentScaler.fit(z)
    z = scaler.transform(z)
    codec = CondCodec(spec, C.CONTINUOUS_LOG10).fit(obs)
    cond_np = codec.encode(obs)
    print(f"latent_dim={latent_dim}  n_train={len(z)}  "
          f"cats={{ {', '.join(f'{n}:{codec.cardinality(n)}' for n,k,_ in spec if k=='categorical')} }}")

    z_t = torch.as_tensor(z, dtype=torch.float32)
    cond_t = to_tensors(cond_np, "cpu")

    # --- model + diffusion ---
    net = LDMNet(latent_dim, codec, hidden=args.hidden, depth=args.depth).to(dev)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    diff = (GaussianDiffusion(args.num_steps, device=dev) if args.method == "ddpm"
            else RectifiedFlow(device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    n = len(z_t)
    steps_per_epoch = max(1, n // args.batch_size)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            z0 = z_t[idx].to(dev)
            cond = {k: v[idx].to(dev) for k, v in cond_t.items()}
            drop = (torch.rand(len(idx), device=dev) < args.p_uncond)
            cemb = net.embed(cond, drop)
            predict = lambda zt, t: net.denoise(zt, t, cemb)  # noqa: E731
            loss = diff.training_loss(predict, z0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(args.ema).add_(v, alpha=1 - args.ema)
            tot += loss.item()
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}  loss {tot/steps_per_epoch:.4f}")

    if args.save:
        d = C.ldm_dir(args.vae, tag=args.method)
        os.makedirs(d, exist_ok=True)
        torch.save({"ema": ema, "state": net.state_dict()}, os.path.join(d, "net.pt"))
        json.dump({"method": args.method, "hidden": args.hidden, "depth": args.depth,
                   "latent_dim": latent_dim, "num_steps": args.num_steps,
                   "vae": args.vae, "dataset": C.DATASET,
                   "scaler": scaler.to_dict(), "codec": codec.to_dict()},
                  open(os.path.join(d, "meta.json"), "w"))
        print("saved LDM ->", d)


if __name__ == "__main__":
    main()
