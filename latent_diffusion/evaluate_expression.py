"""
Expression-space evaluation of the LDM = cell-state / perturbation PREDICTION.

For each condition group (defined by the categorical+continuous conditions, e.g.
sciplex (cell_type, drug, dose) / PBMC (cell_type, ctrl|stim)):
  1. generate n latents conditioned on the group (CFG),
  2. DECODE them to gene expression via the VAE decoder,
  3. compare predicted vs real (measured) expression.

Metrics (per condition group, then averaged over groups):
  MEAN expression (predicted vs real mean profile):
    * R2  / Pearson  -- all genes            (overall sanity; inflated by stable genes)
    * R2  / Pearson  -- top-K DEGs           (perturbation-specific genes)
  DELTA = (group mean - control baseline)    (isolates the predicted CHANGE):
    * R2  / Pearson  -- all genes
    * R2  / Pearson  -- top-K DEGs           (**most stringent** perturbation metric)
  Ceilings (decode REAL latents -> separates VAE-decoder error from LDM error):
    * R2 mean-all, R2 delta-all

Control baseline for DELTA / DEG:
  - if the data has an explicit control (condition in {control,ctrl,vehicle,dmso,
    none,untreated}, e.g. PBMC 'ctrl'): baseline = mean of control cells per cell
    type, and control groups are excluded from evaluation.
  - else (sciplex is treated-only): baseline = per-cell-type mean profile
    (a self-contained proxy). Reported as `baseline_kind` in the summary.
  DEGs = top-K genes by |delta_real| (genes this condition changes most vs baseline).

Runs in the DECODER's env (scGen -> sem_scgen_py39, scVI -> sem_scvi, ...), on
CPU if the env's torch can't use the GPU. LDM ckpt is plain tensors -> loads
across torch versions.

  python evaluate_expression.py --vae scgen --decoder scgen --method flow \
      --guidance 1.5 --n_per_group 300 --max_groups 60 --top_degs 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anndata as ad
import numpy as np
import torch
from sklearn.metrics import r2_score

import config as C
from sample import load_ldm


def guided_sample(net, diff, meta, cond, n, guidance, steps, dev):
    cemb_c = net.embed(cond, drop=None)
    cemb_u = net.embedder.null_like(n, dev)

    def predict(z_t, t):
        pc = net.denoise(z_t, t, cemb_c)
        if guidance == 1.0:
            return pc
        pu = net.denoise(z_t, t, cemb_u)
        return pu + guidance * (pc - pu)

    shape = (n, meta["latent_dim"])
    if meta["method"] == "ddpm":
        return diff.ddim_sample(predict, shape, steps=steps)
    return diff.sample(predict, shape, steps=steps, device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--decoder", required=True, help="decoder adapter (scvi/scgen/trvae)")
    ap.add_argument("--method", default="flow")
    ap.add_argument("--ldm_dir", default=None)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n_per_group", type=int, default=300)
    ap.add_argument("--max_groups", type=int, default=60, help="eval the N largest groups")
    ap.add_argument("--min_cells", type=int, default=50)
    ap.add_argument("--top_degs", type=int, default=50)
    ap.add_argument("--control_value", default="auto",
                    help="'auto' (detect control condition), 'celltype_mean' (force "
                         "per-cell-type baseline), or an explicit control label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = args.ldm_dir or C.ldm_dir(args.vae, tag=args.method)
    net, diff, codec, scaler, meta = load_ldm(d, dev)

    # real val: latents (embedding X) + expression (obsm['X_expr']) + conditions
    _, va_emb = C.emb_paths(args.vae)
    A = ad.read_h5ad(va_emb)
    z_real = (A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)).astype(np.float32)
    Xr = A.obsm["X_expr"]
    Xr = (Xr.toarray() if hasattr(Xr, "toarray") else np.asarray(Xr)).astype(np.float32)
    obs = A.obs
    ct_col, drug_col = C.LABELS["cell_type"], C.LABELS["drug"]
    cts = obs[ct_col].astype(str).values

    # --- control baseline per cell type (for DELTA + DEG) ---
    CONTROL_NAMES = {"control", "ctrl", "vehicle", "dmso", "none", "untreated"}
    cond_lower = obs[drug_col].astype(str).str.lower().values
    if args.control_value == "celltype_mean":
        is_ctrl = np.zeros(len(obs), bool)
    elif args.control_value == "auto":
        is_ctrl = np.isin(cond_lower, list(CONTROL_NAMES))
    else:
        is_ctrl = cond_lower == args.control_value.lower()
    if is_ctrl.any():
        baseline_kind = f"control cells ({sorted(set(obs[drug_col].values[is_ctrl]))})"
        baseline = {ct: Xr[(cts == ct) & is_ctrl].mean(0) for ct in np.unique(cts)
                    if ((cts == ct) & is_ctrl).sum() > 0}
        eval_mask = ~is_ctrl                                    # don't score control as a group
    else:
        baseline_kind = "per-cell-type mean profile (no explicit control in data)"
        baseline = {ct: Xr[cts == ct].mean(0) for ct in np.unique(cts)}
        eval_mask = np.ones(len(obs), bool)

    # groups from the conditioning obs columns
    gcols = [col for _, _, col in codec.spec]
    keys = obs[gcols].astype(str).agg("|".join, axis=1).values
    ev = eval_mask & np.array([cts[i] in baseline for i in range(len(cts))])
    uniq, counts = np.unique(keys[ev], return_counts=True)
    groups = [u for u in uniq[np.argsort(-counts)]
              if counts[list(uniq).index(u)] >= args.min_cells][:args.max_groups]
    print(f"dataset={C.DATASET} vae={args.vae} decoder={args.decoder} groups={len(groups)} "
          f"(by {gcols}) n_per_group={args.n_per_group} device={dev}")
    print(f"baseline: {baseline_kind}")

    from decoders import get_decoder
    decoder = get_decoder(args.decoder)

    def r2(a, b):
        return float(r2_score(a, b))

    def pr(a, b):
        s = np.std(a) * np.std(b)
        return float(np.corrcoef(a, b)[0, 1]) if s > 0 else float("nan")

    rows = []
    for gi, gkey in enumerate(groups):
        m = (keys == gkey) & ev
        g_obs = obs[m].iloc[0]
        ct = str(g_obs[ct_col])
        base = baseline[ct]
        mu_real = Xr[m].mean(0)
        # generate + decode
        cond = {}
        for name, kind, col in codec.spec:
            arr = codec.encode_value(name, g_obs[col], args.n_per_group)
            t = torch.as_tensor(arr)
            cond[name] = (t.long() if kind == "categorical" else t.float()).to(dev)
        z_gen = guided_sample(net, diff, meta, cond, args.n_per_group, args.guidance, args.steps, dev)
        z_gen = scaler.inverse(z_gen.cpu().numpy().astype(np.float32))
        mu_pred = decoder.decode(z_gen).mean(0)
        mu_vae = decoder.decode(z_real[m]).mean(0)           # VAE-recon ceiling

        d_real, d_pred, d_vae = mu_real - base, mu_pred - base, mu_vae - base
        deg = np.argsort(np.abs(d_real))[-args.top_degs:]    # DEGs = biggest change vs baseline
        rows.append({
            "group": gkey, "n_real": int(m.sum()),
            "r2_mean_all": r2(mu_real, mu_pred),   "pearson_mean_all": pr(mu_real, mu_pred),
            "r2_mean_deg": r2(mu_real[deg], mu_pred[deg]), "pearson_mean_deg": pr(mu_real[deg], mu_pred[deg]),
            "r2_delta_all": r2(d_real, d_pred),    "pearson_delta_all": pr(d_real, d_pred),
            "r2_delta_deg": r2(d_real[deg], d_pred[deg]),  "pearson_delta_deg": pr(d_real[deg], d_pred[deg]),
            "r2_mean_all_vae": r2(mu_real, mu_vae), "r2_delta_all_vae": r2(d_real, d_vae),
            "r2_mean_deg_vae": r2(mu_real[deg], mu_vae[deg]),
            "r2_delta_deg_vae": r2(d_real[deg], d_vae[deg]),
        })
        if gi < 6 or gi % 20 == 0:
            r = rows[-1]
            print(f"  [{gi:3d}] {gkey[:34]:34s} mean(R2 {r['r2_mean_all']:.3f}/deg {r['r2_mean_deg']:.3f}) "
                  f"delta(R2 {r['r2_delta_all']:.3f}/deg {r['r2_delta_deg']:.3f}, r_deg {r['pearson_delta_deg']:.3f})")

    def avg(k):
        return float(np.nanmean([r[k] for r in rows]))
    keys_out = ["r2_mean_all", "pearson_mean_all", "r2_mean_deg", "pearson_mean_deg",
                "r2_delta_all", "pearson_delta_all", "r2_delta_deg", "pearson_delta_deg",
                "r2_mean_all_vae", "r2_mean_deg_vae", "r2_delta_all_vae", "r2_delta_deg_vae"]
    summary = {"dataset": C.DATASET, "vae": args.vae, "decoder": args.decoder,
               "method": meta["method"], "guidance": args.guidance, "n_groups": len(rows),
               "baseline_kind": baseline_kind, "top_degs": args.top_degs}
    summary.update({k: avg(k) for k in keys_out})
    print("\n=== PERTURBATION-PREDICTION SUMMARY (avg over groups) ===")
    for k in keys_out:
        print(f"  {k:20s}: {summary[k]:.4f}")

    out_dir = os.path.join(C.samples_dir(), "eval")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"pred_{args.vae}_{meta['method']}.json")
    json.dump({"summary": summary, "per_group": rows}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
