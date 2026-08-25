# Tahoe-100M — data prep for a general (out-of-core) VAE

~95.6M cells (3388 parquet shards × ~28k cells), 50 cell lines × ~379 drugs
(Vevo/Arc). Each cell is sparse `(genes=token_ids, expressions=raw counts)` over
a **62,710-gene vocabulary**; obs has `drug`, `cell_line_id` (CVCL), `moa-fine`,
`canonical_smiles`, `plate`, `sample`. token_ids 0–2 are special tokens (first
per-cell entry is a CLS-like token with a −2 sentinel); real genes have
token_id ≥ 3.

Too large to load in memory → **stream mini-batches from the shards on disk**.
Everything lives on `/scratch/.../datasets/tahoe/` (never the project dir).

## Files
```
tahoe/
├── tahoe_common.py    # paths, gene metadata, fast shard->CSR, HVG save/load, shard split
├── download_tahoe.py  # snapshot_download all shards+metadata (xet disabled -> HTTPS)
├── build_hvg.py       # pick 5000 HVG from a subsample of shards -> hvg5000.json
├── tahoe_dataset.py   # torch IterableDataset: streaming 5000-HVG count minibatches + conditions
└── README.md
```

## Pipeline
```bash
ENV=/scratch/yuchen.yan/envs/squidiff/bin/python
export HF_HOME=/scratch/yuchen.yan/hf_cache HF_HUB_DISABLE_XET=1 HDF5_USE_FILE_LOCKING=FALSE

# 1. download all shards + metadata (~241 GB, resumable; run on an internet node)
nohup $ENV download_tahoe.py > /scratch/yuchen.yan/tahoe_download.log 2>&1 &

# 2. select 5000 HVG from a subsample of TRAIN shards -> tahoe/hvg5000.json
$ENV build_hvg.py --n_shards 40 --cells_per_shard 5000 --n_top 5000

# 3. use the streaming dataset in a VAE trainer (next step)
#    from tahoe_dataset import TahoeStream
#    ds = TahoeStream(split="train", batch_size=512, cond_cols=("cell_line_id","drug","moa-fine"))
#    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=4)
#    for counts, cond in loader:  # counts [B,5000] raw counts; cond dict of str lists
```

## Design notes
- **Latent-dim / gene-space agnostic downstream**: fixed 5000-HVG panel (token_ids
  saved in `hvg5000.json`); the loader subsets every cell to these genes.
- **Fast reconstruction**: CSR built directly from parquet list offsets (no
  per-cell Python loop) — ~0.5 s/shard; a 1024×5000 minibatch densifies in ~0.03 s.
- **Train/val split** is by shard (every 20th shard = val) so streaming keeps whole
  shards intact and there is no cell overlap.
- **Raw counts** are yielded (for an NB/scVI-style VAE); normalize inside the model,
  or switch to log-norm in the trainer if using a Gaussian decoder.
- **Conditions** (cell line / drug / MOA / SMILES) are yielded as string lists per
  batch; the VAE trainer encodes them (for a conditional VAE) or ignores them.
- Verified end-to-end on one shard (build_hvg + streaming) before the full download.
