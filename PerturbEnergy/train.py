"""
Train PerturbEnergy on Tahoe-100M (streaming, out-of-core) -- paper Algorithm 1.

One step on a batch (x, c, a):
  forward   z_b ~ q(z_b|x,c); M candidates z_a ~ q(z_a|z_b,a); the ENERGY picks one;
            x_hat = p_theta(x|z_b, z_a_selected)
  phase A   alpha <- alpha - zeta * grad J_E   (Eq.12; VAE inputs detached)
            positives  = the selected z_a
            negatives  = Langevin chain started from ANOTHER cell's z_a in the batch
  phase B   {theta, phi_b, phi_a, emb} <- + xi * grad L_ELBO   (Eq.16; energy frozen)
            L = recon + b_b*KL(q_b||N(0,I)) + b_a*KL(q_a||N(0,I))
                + w_E * E[E_alpha] + alpha_con * ||z_b(x) - z_b(x_control)||^2

  python train.py --set train.max_steps=100000
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
from torch.utils.data import DataLoader

import cfg as CFG
from data import ControlBank, load_vocabs, lognorm, make_stream, to_ids
from langevin import sample_negative
from losses import contrastive_align, energy_objective, kl_standard, recon_loss
from models import PerturbEnergy


def set_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


@torch.no_grad()
def val_metrics(model, loader, cfg, c2i, a2i, dev, max_batches=10):
    """Held-out reconstruction (deterministic path) + energy gap + collapse diagnostics."""
    model.eval()
    X, Xh, gaps, spreads = [], [], [], []
    for i, (counts, cond) in enumerate(loader):
        if i >= max_batches:
            break
        x = lognorm(counts.to(dev))
        c, a, keep = to_ids(cond, cfg, c2i, a2i, dev)
        if keep.sum() == 0:
            continue
        x, c, a = x[keep], c[keep], a[keep]
        o = model(x, c, a, sample=False)
        X.append(x.cpu().numpy()); Xh.append(o["x_hat"].cpu().numpy())
        spreads.append(o["z_a"].std(0).mean().item())
        if not model.use_energy:
            gaps.append(0.0)
            continue
        with torch.enable_grad():
            init = o["z_a"][torch.randperm(len(o["z_a"]), device=dev)]
            zn = sample_negative(model, o["z_b"], o["a_e"], init, cfg.langevin.steps,
                                 cfg.langevin.step_size, cfg.langevin.noise_scale,
                                 cfg.langevin.clamp)
        gaps.append((model.energy(zn, o["z_b"], o["a_e"]).mean() - o["e_sel"].mean()).item())
    X, Xh = np.concatenate(X), np.concatenate(Xh)

    def colr(A, B):
        A, B = A - A.mean(0, keepdims=True), B - B.mean(0, keepdims=True)
        return float(np.nanmean((A * B).sum(0) / (np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0)) + 1e-8)))

    ss_res, ss_tot = ((X - Xh) ** 2).sum(), ((X - X.mean()) ** 2).sum() + 1e-8
    model.train()
    return {"val_mse": float(((X - Xh) ** 2).mean()), "val_r2": float(1 - ss_res / ss_tot),
            "val_gene_pearson": colr(Xh, X), "val_cell_pearson": colr(Xh.T, X.T),
            "energy_gap": float(np.mean(gaps)), "z_a_std": float(np.mean(spreads))}


def main():
    ap = argparse.ArgumentParser()
    CFG.add_cfg_args(ap)
    args = ap.parse_args()
    cfg = CFG.load(args.config, args.set)
    L, O, LG = cfg.loss, cfg.optim, cfg.langevin

    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = cfg.train.out_dir
    os.makedirs(out, exist_ok=True)
    json.dump(dict(cfg), open(os.path.join(out, "config_used.json"), "w"), indent=2)

    c2i, a2i, n_cond, n_pert = load_vocabs()
    tr, va = make_stream("train", cfg), make_stream("val", cfg)
    n_genes = len(tr.hvg_tokens)
    model = PerturbEnergy(n_genes, n_cond, n_pert, dict(cfg.model)).to(dev)
    print(f"PerturbEnergy | genes={n_genes} n_cond={n_cond} n_pert={n_pert} "
          f"z_b={model.z_b_dim} z_a={model.z_a_dim} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M dev={dev}\n"
          f"  use_energy={model.use_energy} candidates={cfg.model.n_candidates} "
          f"select={cfg.model.select_mode} "
          f"recon={L.recon} kl=({L.beta_kl_b},{L.beta_kl_a}) margin={O.margin if O.use_margin else 'off'}\n"
          f"  aux_ce_weight={L.aux_ce_weight} "
          f"  langevin(K={LG.steps}, eta={LG.step_size}, init=batch-negatives) "
          f"z_b-invariance(w={L.contrast_weight}, {L.contrast_mode})", flush=True)

    use_contrast = float(L.contrast_weight) > 0
    bank = ControlBank(cfg, dev, c2i) if use_contrast else None

    # auxiliary drug classifier on z_a: a train-time regulariser only. It is optimised
    # together with the VAE and DISCARDED afterwards -- the checkpoint stays a pure model.
    use_aux = float(L.aux_ce_weight) > 0
    aux_head = (nn.Sequential(nn.Linear(model.z_a_dim, L.aux_hidden), nn.ReLU(),
                              nn.Dropout(L.aux_dropout), nn.Linear(L.aux_hidden, n_pert)).to(dev)
                if use_aux else None)
    aux_ce_fn = nn.CrossEntropyLoss()
    vae_params = model.vae_parameters() + (list(aux_head.parameters()) if use_aux else [])
    vae_opt = torch.optim.AdamW(vae_params, lr=O.vae_lr, weight_decay=O.weight_decay)
    e_opt = (torch.optim.AdamW(model.energy_parameters(), lr=O.energy_lr,
                               weight_decay=O.weight_decay) if model.use_energy else None)

    cf = open(os.path.join(out, "metrics.csv"), "w"); cw = csv.writer(cf)
    cols = ["step", "loss", "recon_mse", "mmd", "kl_b", "kl_a", "e_pos", "e_neg", "energy_loss",
            "contrast", "zb_shift", "zb_std", "aux_ce", "aux_acc", "val_mse", "val_r2",
            "val_gene_pearson", "val_cell_pearson", "energy_gap", "z_a_std"]
    cw.writerow(cols); cf.flush()
    acc = {k: 0.0 for k in ("loss", "mse", "mmd", "kl_b", "kl_a", "e_pos", "e_neg",
                            "eloss", "con", "shift", "zbstd", "auxce", "auxacc", "n")}

    ep_v = en_v = el_v = 0.0
    step, epoch, done = 0, 0, False
    while not done:
        tr.set_epoch(epoch)
        for counts, cond in DataLoader(tr, batch_size=None, num_workers=cfg.data.num_workers):
            model.train()
            x = lognorm(counts.to(dev))
            c, a, keep = to_ids(cond, cfg, c2i, a2i, dev)
            if keep.sum() < 8:
                continue
            x, c, a = x[keep], c[keep], a[keep]
            w_con = L.contrast_weight * min(1.0, step / max(1, L.contrast_warmup))

            # ---------- forward (energy is read-only here: it only SELECTS z_a) ----------
            if model.use_energy:
                set_grad(model.energy_net, False)
            o = model(x, c, a, sample=True)

            # ---------- phase A: energy update (Eq.12), everything else detached ---------
            if model.use_energy and step % max(1, O.energy_update_every) == 0:
                set_grad(model.energy_net, True)
                z_b_d, a_e_d = o["z_b"].detach(), o["a_e"].detach()
                z_pos = o["z_a"].detach()
                init = z_pos[torch.randperm(len(z_pos), device=dev)]   # another cell's z_a
                z_neg = sample_negative(model, z_b_d, a_e_d, init, LG.steps, LG.step_size,
                                        LG.noise_scale, LG.clamp)
                e_pos = model.energy(z_pos, z_b_d, a_e_d)
                e_neg = model.energy(z_neg, z_b_d, a_e_d)
                e_loss = energy_objective(e_pos, e_neg, O.margin, O.use_margin, O.energy_l2_reg)
                e_opt.zero_grad(set_to_none=True)
                e_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.energy_parameters(), O.grad_clip)
                e_opt.step()
                ep_v, en_v, el_v = e_pos.mean().item(), e_neg.mean().item(), e_loss.item()
                set_grad(model.energy_net, False)

            # ---------- phase B: VAE update (Eq.16), energy frozen ----------------------
            # (ablation use_energy=false: identical objective minus the E[E_alpha] term)
            rec, mse_v, mmd_v = recon_loss(x, o["x_hat"], L.recon, L.mmd_weight, tuple(L.mmd_scales))
            kl_b, kl_a = kl_standard(o["mu_b"], o["lv_b"]), kl_standard(o["mu_a"], o["lv_a"])
            e_in_elbo = (model.energy(o["z_a"], o["z_b"], o["a_e"]).mean()
                         if model.use_energy else torch.zeros((), device=dev))
            if use_contrast and w_con > 0:
                x_ctl, pmask = bank.sample(c, cond.get("plate"))
                is_ctl = torch.from_numpy(
                    np.array([str(d) for d, k in zip(cond[cfg.data.pert_col], keep.tolist()) if k])
                    == cfg.data.control_label).to(dev)
                mu_bc = model.basal(x_ctl, o["c_e"], sample=False)[1]
                l_con, shift = contrastive_align(o["mu_b"], mu_bc, pmask & ~is_ctl, c, L.contrast_mode)
            else:
                l_con = shift = torch.zeros((), device=dev)
            if use_aux:                                   # CE(drug | z_a), on the posterior mean
                logits = aux_head(o["mu_a"])
                aux_ce = aux_ce_fn(logits, a)
                aux_acc = (logits.argmax(1) == a).float().mean()
            else:
                aux_ce = aux_acc = torch.zeros((), device=dev)
            loss = (rec + L.beta_kl_b * kl_b + L.beta_kl_a * kl_a
                    + L.energy_elbo_weight * e_in_elbo + w_con * l_con
                    + L.aux_ce_weight * aux_ce)
            vae_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vae_parameters(), O.grad_clip)
            vae_opt.step()
            if model.use_energy:
                set_grad(model.energy_net, True)

            for k, v in (("loss", loss), ("mse", mse_v), ("mmd", mmd_v), ("kl_b", kl_b),
                         ("kl_a", kl_a), ("con", l_con), ("shift", shift),
                         ("auxce", aux_ce), ("auxacc", aux_acc)):
                acc[k] += float(v)
            acc["e_pos"] += ep_v; acc["e_neg"] += en_v; acc["eloss"] += el_v
            acc["zbstd"] += float(o["mu_b"].std(0).mean()); acc["n"] += 1
            step += 1

            if step % cfg.train.eval_every == 0 or step >= cfg.train.max_steps:
                m = val_metrics(model, DataLoader(va, batch_size=None, num_workers=0),
                                cfg, c2i, a2i, dev)
                A = {k: acc[k] / max(1, acc["n"]) for k in acc if k != "n"}
                print(f"step {step:6d} | loss {A['loss']:.1f} mse {A['mse']:.1f} "
                      f"kl_b {A['kl_b']:.1f} kl_a {A['kl_a']:.1f} | E+ {A['e_pos']:.3f} "
                      f"E- {A['e_neg']:.3f} gap {m['energy_gap']:.3f} | con {A['con']:.3f} "
                      f"zb_shift {A['shift']:.3f} zb_std {A['zbstd']:.3f} "
                      f"aux_ce {A['auxce']:.3f} aux_acc {A['auxacc']:.3f} | "
                      f"val R2 {m['val_r2']:.4f} cell_r {m['val_cell_pearson']:.4f} "
                      f"z_a_std {m['z_a_std']:.4f}", flush=True)
                cw.writerow([step, A["loss"], A["mse"], A["mmd"], A["kl_b"], A["kl_a"], A["e_pos"],
                             A["e_neg"], A["eloss"], A["con"], A["shift"], A["zbstd"],
                             A["auxce"], A["auxacc"], m["val_mse"],
                             m["val_r2"], m["val_gene_pearson"], m["val_cell_pearson"],
                             m["energy_gap"], m["z_a_std"]]); cf.flush()
                torch.save({"model": model.state_dict(), "cfg": dict(cfg), "step": step,
                            "n_genes": n_genes, "n_cond": n_cond, "n_pert": n_pert},
                           os.path.join(out, "perturbenergy.pt"))
                if use_aux:     # saved separately; never needed at inference
                    torch.save({"aux_head": aux_head.state_dict(), "weight": L.aux_ce_weight},
                               os.path.join(out, "aux_head.pt"))
                acc = {k: 0.0 for k in acc}
            if step >= cfg.train.max_steps:
                done = True
                break
        epoch += 1
    print("PERTURBENERGY_TRAIN_DONE ->", out)


if __name__ == "__main__":
    main()
