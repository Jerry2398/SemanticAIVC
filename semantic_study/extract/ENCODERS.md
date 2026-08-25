# Adding encoders to the semantic study

The 9 metrics never change. To evaluate a new encoder you only write an
**extractor** that produces two files in the shared format:

```
config.emb_paths("<name>")  ->  <EMB_DIR>/<name>_train.h5ad , <name>_val.h5ad
  .X              = encoder latent (n_cells, latent_dim)
  .obsm["X_expr"] = 200-HVG log-norm expression   (attached by _base.write_embedding)
  .obs            = sciplex labels                 (attached by _base.write_embedding)
```

Then:

```bash
python run_all.py --encoder <name>       # results -> results/<name>/summary.json
```

`extract/_base.py` does the bookkeeping: your extractor just encodes
`load_split("train")` / `load_split("val")` (in order) and calls
`write_embedding(name, split, latent)`. Because `X_expr` and `obs` come from the
canonical split files, **every encoder is compared against an identical
reference** and results are directly comparable.

---

## Data: canonical v2 dataset (real counts)

`prep_sciplex_counts.py` builds the canonical dataset from scPerturb raw counts:
5000-HVG train/val files with **`.X` = log-norm** and **`.layers['counts']` = raw
integer counts**, plus a full-gene version (`--emit_full`) for FMs. So every
model gets the input it wants from ONE dataset:

| Model | Reads | Setup |
|---|---|---|
| scGen, trVAE (`mse`), Squidiff | `.X` (log-norm) | ✅ |
| scVI, scDisInFact | `.layers['counts']` (raw) | ✅ `--counts_source layer` (default) |
| Geneformer / scFoundation | raw counts | ✅ full-gene version |
| scGPT / UCE / SCimilarity | log-norm + symbols | ✅ full-gene version |

`_base.reconstruct_counts` is only a fallback for count-less data; the canonical
dataset has real counts, so leave scVI at `--counts_source layer`.

## Checkpoints

Pass `--save_model` to persist a trained model under
`config.models_dir(<name>)` = `/scratch/.../models/<name>/`, so you can
re-extract later without retraining.

---

## Environment strategy

These frameworks have **conflicting dependencies** — do NOT put them in the
`squidiff` env. One small conda env per model family; each extractor imports
only its own package. The metrics run in any env with `scanpy/torch/sklearn`
(the `squidiff` env is fine for `run_all.py`).

```bash
conda create -n scvi   python=3.10 -y && conda activate scvi   && pip install scvi-tools scanpy
conda create -n scgen  python=3.10 -y && conda activate scgen  && pip install scgen scanpy
conda create -n trvae  python=3.9  -y && conda activate trvae  && pip install scarches scanpy
conda create -n scdi   python=3.9  -y && conda activate scdi   && pip install scdisinfact scanpy
```

Typical loop per model:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
conda activate scvi     && python extract/extract_scvi.py --save_model   # trains + extracts (+ saves ckpt)
conda activate squidiff && python run_all.py --encoder scvi              # metrics in any env
```

The provided VAE extractors (`extract_scvi.py`, `extract_scgen.py`,
`extract_trvae.py`, `extract_scdisinfact.py`) follow each package's standard API
but are **untested here** (their packages aren't installed in this env). scVI /
scGen / trVAE use stable APIs; scDisInFact has version-dependent function names —
its script marks the two lines to verify against your install.

---

## Foundation-model embeddings (recommended)

All produce a per-cell embedding; feed it to `write_embedding` and evaluate
exactly the same way. sciplex is human with gene **symbols** present (good), but
FMs usually want the **full 5000-gene panel** (`config.FULL_DATA_PATH`), not the
200 HVGs — subset it to the split cells, encode, and pass the latent (rows in
canonical split order) to `write_embedding`.

Priority for this dataset (no counts, symbols available):

| Model | Why / notes | Counts? |
|---|---|---|
| **UCE** (snap-stanford/UCE) | Truly zero-shot, no fine-tuning, protein-embedding based; easiest FM baseline | log-norm ok |
| **scGPT** (bowang-lab/scGPT) | 33M-cell transformer; strong general cell embeddings, zero-shot or fine-tune | prefers counts, handles norm |
| **SCimilarity** (Genentech) | Human cell-embedding FM for similarity/annotation; log-norm to its gene set | log-norm ok |
| **Geneformer** (HF: ctheodoris/Geneformer) | Rank-value encoding; cell emb = mean gene-token emb | **needs counts** (ranking) |
| **scFoundation / xTrimoGene** (biomap-research) | 100M-param; reads counts + total; cell embeddings | **needs counts** |
| **CellPLM** (OmicsML/CellPLM) | Cell language model; per-cell latent | log-norm ok |

Suggested order given the counts constraint: **UCE → scGPT → SCimilarity**
first; add Geneformer / scFoundation only if you reconstruct or fetch counts.

Write an `extract_<fm>.py` that: loads `FULL_DATA_PATH`, subsets to the split
cells (match by re-deriving the split, or by an index you save in
`prep_sciplex.py`), runs the FM's embedding call, then `write_embedding(...)`.
