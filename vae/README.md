# Simple VAE on Tahoe-100M (streaming)

A minimal scGen-style Gaussian VAE trained out-of-core on Tahoe-100M's 5000-HVG
expression (data prep in `../tahoe`). Reconstruction (MSE) + KL only.

- **Input**: 5000-HVG raw counts (from `TahoeStream`), log-normalized on the fly
  (`normalize_total(1e4)+log1p`, like scGen).
- **Model** (`model.py`): encoder MLP → (mu, logvar) → z → decoder MLP → x̂.
  Loss = per-cell SSE reconstruction + `beta`·KL(q(z|x)‖N(0,I)); `beta` warms up.
  Latent dim is a hyperparameter (`--latent_dim`, default 32).
- **Streaming**: mini-batch SGD over shards; never loads all ~95.6M cells.
- **Monitoring**:
  - loss curve → `metrics.csv` + `loss_curve.png` (train loss / recon / kl, and val curves).
  - periodic **val reconstruction** every `--eval_every` steps: gene-wise & cell-wise
    Pearson, global R², MSE (encode→decode val cells vs their real HVG), plus a
    **generative** check (decode z~N(0,I), Pearson of mean profile vs real val mean).

All outputs → `/scratch/.../models/tahoe_vae/` (checkpoints/logs never in the project dir).

## Run
```bash
ENV=/scratch/yuchen.yan/envs/squidiff/bin/python     # torch2.5 GPU + pyarrow + scanpy
export HDF5_USE_FILE_LOCKING=FALSE
# GPU job (recommended):
qsub -v LATENT=32,MAX_STEPS=30000 submit_vae.sh
# or directly:
$ENV train.py --latent_dim 32 --max_steps 30000 --eval_every 2000 --num_workers 8
$ENV evaluate.py --max_batches 200
```

## Notes
- Requires `../tahoe/hvg5000.json` (built by `tahoe/build_hvg.py`) and the downloaded shards.
- This is intentionally the simplest VAE (unconditional, Gaussian). Conditioning
  (cell_line/drug/MOA) or an NB likelihood can be added later; `TahoeStream` already
  yields the conditions.
- ~30k steps × 512 ≈ 15M cells (~0.16 epoch) — enough to see convergence; raise
  `MAX_STEPS` for more.
