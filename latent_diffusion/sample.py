"""
Conditional sampling from a trained LDM -> latent vectors (+ optional decode).

  SEMANTIC_DATASET=sciplex python sample.py --vae scvi \
      --set cell_type=A549 --set drug=Vorinostat --set dose=10000 \
      --n 2000 --guidance 2.0 --out samples.h5ad

Runs classifier-free guidance (conditional vs joint-null) at scale --guidance,
de-standardizes to the VAE's native latent scale, and writes an .h5ad whose X is
the generated latent and obs records the requested conditions. Pass --decode
<scvi> to additionally reconstruct gene expression via a decoder adapter.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anndata as ad
import numpy as np
import torch

import config as C
from data import LatentScaler, CondCodec
from models.net import LDMNet
from diffusion.ddpm import GaussianDiffusion
from diffusion.flow import RectifiedFlow


def load_ldm(d, device):
    meta = json.load(open(os.path.join(d, "meta.json")))
    codec = CondCodec.from_dict(meta["codec"])
    scaler = LatentScaler.from_dict(meta["scaler"])
    net = LDMNet(meta["latent_dim"], codec, hidden=meta["hidden"], depth=meta["depth"]).to(device)
    ckpt = torch.load(os.path.join(d, "net.pt"), map_location=device)
    net.load_state_dict(ckpt.get("ema", ckpt["state"]))
    net.eval()
    diff = (GaussianDiffusion(meta["num_steps"], device=device) if meta["method"] == "ddpm"
            else RectifiedFlow(device=device))
    return net, diff, codec, scaler, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--ldm_dir", default=None, help="default: config.ldm_dir(vae, method)")
    ap.add_argument("--method", default="flow", help="only used to locate default ldm_dir")
    ap.add_argument("--set", action="append", default=[], metavar="name=value",
                    help="fix a condition, e.g. --set cell_type=A549 --set dose=10000")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--guidance", type=float, default=1.0, help="CFG scale (1.0 = no guidance)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--decode", default=None, help="decoder adapter name (e.g. 'scvi') to also make expression")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = args.ldm_dir or C.ldm_dir(args.vae, tag=args.method)
    net, diff, codec, scaler, meta = load_ldm(d, dev)

    requested = dict(kv.split("=", 1) for kv in args.set)
    n = args.n
    cond = {}
    for name, kind, _ in codec.spec:
        val = requested.get(name)
        if val is None:                      # unspecified -> null (unconditional for this var)
            arr = np.full(n, codec.cardinality(name) if kind == "categorical" else 0.0,
                          dtype=np.int64 if kind == "categorical" else np.float32)
        else:
            arr = codec.encode_value(name, val, n)
        t = torch.as_tensor(arr)
        cond[name] = (t.long() if kind == "categorical" else t.float()).to(dev)
    print(f"sampling n={n} from {d} | conditions={requested} | guidance={args.guidance}")

    cemb_c = net.embed(cond, drop=None)
    cemb_u = net.embedder.null_like(n, dev)
    g = args.guidance

    def predict(z_t, t):
        pc = net.denoise(z_t, t, cemb_c)
        if g == 1.0:
            return pc
        pu = net.denoise(z_t, t, cemb_u)
        return pu + g * (pc - pu)

    shape = (n, meta["latent_dim"])
    if meta["method"] == "ddpm":
        z = diff.ddim_sample(predict, shape, steps=args.steps)
    else:
        z = diff.sample(predict, shape, steps=args.steps, device=dev)
    z = scaler.inverse(z.cpu().numpy().astype(np.float32))    # back to VAE latent scale

    out = ad.AnnData(X=z)
    for k, v in requested.items():
        out.obs[k] = v
    out.uns["ldm"] = {"vae": args.vae, "method": meta["method"], "guidance": g}
    out_path = args.out or os.path.join(C.samples_dir(), f"gen_{args.vae}_{meta['method']}.h5ad")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.write_h5ad(out_path)
    print("wrote generated latents:", out_path, out.shape)

    if args.decode:
        from decoders import get_decoder
        dec = get_decoder(args.decode)
        expr = dec.decode(z)
        e = ad.AnnData(X=expr, obs=out.obs.copy())
        ep = out_path.replace(".h5ad", "_expr.h5ad")
        e.write_h5ad(ep)
        print("wrote decoded expression:", ep, e.shape)


if __name__ == "__main__":
    main()
