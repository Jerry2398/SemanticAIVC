# Latent Diffusion for conditional cell-state generation

Two-stage generation, building on the trained encoders in `../semantic_study`:

1. **Stage 1 (VAE, already trained)** — an encoder-decoder (scVI / scGen / trVAE /
   scDisInFact / Squidiff) maps expression `x → z` and back `z → x`. Their latent
   embeddings + condition labels are already saved by `semantic_study/extract`.
2. **Stage 2 (this package)** — a conditional diffusion / flow-matching model
   learns `p(z | condition)` in that latent space. Sampling gives `z ~ p(z|cond)`,
   which the VAE decoder turns back into gene expression.

## Design highlights (per the constraints)

- **Trains only on latent embeddings** (no VAE loaded at train time) — reads the
  `.h5ad` produced by semantic_study.
- **Latent-dim agnostic**: the dim is read from the embedding file (`z.shape[1]`),
  never hardcoded — so it aligns to whatever dim each VAE used (and to the dim-32
  retrain automatically).
- **Multi-condition**: per-dataset condition spec (`config.CONDITION_SPEC`);
  categorical (cell_type, drug, ctrl/stim) via embedding tables, continuous (dose)
  via an MLP. Any subset selectable with `--conditions`.
- **Two training paradigms**, hyperparameter-selected: `--method ddpm` (Gaussian
  diffusion, ε-prediction, DDIM sampling) or `--method flow` (rectified flow /
  conditional flow matching, ODE sampling). `flow` is the recommended default.
- **Classifier-free guidance**: joint condition-dropout in training (`--p_uncond`),
  guided sampling (`--guidance`).
- **Isolated** from semantic_study (imports its config read-only); checkpoints on
  `/scratch/.../ldm_models/` (not the project dir).

## Layout

```
latent_diffusion/
├── config.py            # dataset/paths (reuses semantic_study), condition spec
├── data.py              # load latents (infers dim), LatentScaler, CondCodec
├── conditioning.py      # ConditionEmbedder (+ CFG null)
├── models/
│   ├── denoiser.py      # adaLN-Zero conditional MLP denoiser
│   └── net.py           # embedder + denoiser bundle (LDMNet)
├── diffusion/
│   ├── ddpm.py          # Gaussian diffusion + DDIM
│   └── flow.py          # rectified flow / CFM + Euler ODE
├── train.py             # train (method=ddpm|flow), EMA, CFG dropout -> ckpt
├── sample.py            # conditional sampling -> latent (+ optional decode)
├── evaluate.py          # latent-space eval: MMD + condition-fidelity probe
├── decoders/            # z -> expression adapters (scvi provided)
└── submit_ldm.sh        # PBS GPU job (train + evaluate)
```

## Run

```bash
ENV=/scratch/yuchen.yan/envs/squidiff/bin/python     # torch2.5 + scanpy + sklearn
export HDF5_USE_FILE_LOCKING=FALSE

# train (flow) on the scVI latent space of sciplex
SEMANTIC_DATASET=sciplex $ENV train.py --vae scvi --method flow --save
# conditional generation
SEMANTIC_DATASET=sciplex $ENV sample.py --vae scvi \
    --set cell_type=A549 --set drug=Vorinostat --set dose=10000 \
    --n 2000 --guidance 2.0
# latent-space evaluation
SEMANTIC_DATASET=sciplex $ENV evaluate.py --vae scvi --method flow --guidance 1.5

# or as a GPU job:
qsub -v VAE=scvi,METHOD=flow,SEMANTIC_DATASET=sciplex submit_ldm.sh
```

## Evaluation

Two complementary evals:

- **Latent space** (`evaluate.py`, runs in the training env, no decoder): RBF-MMD
  + mean/cov error between generated and real val latents, and a condition-fidelity
  probe.
- **Expression space = perturbation / cell-state PREDICTION** (`evaluate_expression.py`,
  runs in the DECODER's env): per condition-group, generate → **decode to gene
  expression** → compare to real measured expression. Reports the standard
  scGen/CPA suite — **R² of the mean (all genes + top-K DEGs)** and Pearson, plus a
  VAE-reconstruction R² ceiling that separates decoder error from LDM error.

  ```bash
  # in the decoder's env (scGen -> sem_scgen_py39, scVI -> sem_scvi, ...)
  qsub -v VAE=scgen,DECODER=scgen,METHOD=flow,SEMANTIC_DATASET=sciplex submit_eval_expr.sh
  ```

## Decoding latents → expression

`sample.py --decode scvi` reconstructs expression via `decoders/scvi_decoder.py`
(scVI generative decoder, returns the library-normalized profile). **Must run in
the scVI env** (`/scratch/yuchen.yan/envs/sem_scvi`), since it loads the saved
scVI model. Other VAEs (scGen/trVAE) can get an adapter the same way; Squidiff is
itself a diffusion decoder and is not a natural fit for this stage.

## Notes

- Choose a VAE with a usable decoder for the full generate→expression pipeline:
  **scVI** is the cleanest (modern env, standard decoder). scGen/trVAE also work
  with an adapter.
- The diffusion is a small MLP (latent vectors, not images) — training is fast;
  GPU is optional for small datasets.
- Verified end-to-end (train → sample → evaluate) for both `flow` and `ddpm` on
  the PBMC scVI latent space.
