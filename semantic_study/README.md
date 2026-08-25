# Semantic Study of Cell-Representation Encoders

Probes the **semantic content** of an encoder's latent space on single-cell
perturbation data (sciplex-3), following two axes:

- **Semantic Decodability** — how much biological information can be *recovered*
  from the latent (linear probes only, so we measure the latent and not a strong
  decoder).
- **Semantic Organization** — whether the latent *preserves the geometry* of the
  real biological world (expression manifold, cell ontology, drug/MOA structure).

## Design: encoder-agnostic

Only one step is encoder-specific. `extract/` loads a trained encoder and writes
two embedding files (train / val). **Every metric reads that generic format**, so
to study a different encoder you only re-run extraction:

```
<embedding>.h5ad
  .X                = encoder embedding        (n_cells, latent_dim)
  .obsm["X_expr"]   = log-norm gene expression (n_cells, n_genes)
  .obs              = labels (cell_type, condition, dose, pathway, target, ...)
```

Probes are **trained on the train split and evaluated on the val split**.
Geometry metrics run directly on the val embeddings (no training).

## Layout

```
semantic_study/
├── config.py            # paths, label columns, model config (edit for a new dataset)
├── common.py            # IO + correlation/geometry helpers, embedding format
├── probes.py            # linear / MLP probe models + train loops (standardized, early stop)
├── extract/
│   └── extract_squidiff_embeddings.py   # ONLY encoder-specific file
├── decodability/
│   ├── ger.py           # Gene Expression Recoverability  (linear regression -> expr)
│   ├── ctd.py           # Cell Type Decodability          (softmax -> cell_type)
│   ├── dd.py            # Drug Decodability               (softmax -> drug)
│   ├── dose_d.py        # Dose Decodability               (regression -> log10 dose)
│   ├── moa_d.py         # MOA Decodability                (softmax -> pathway)
│   └── _classify.py     # shared classification runner
├── organization/
│   ├── egc.py           # Expression Geometry Consistency (Mantel / distance corr)
│   ├── lnp.py           # Local Neighborhood Preservation (kNN Jaccard)
│   ├── cog.py           # Cell Ontology Geometry          (centroid vs ontology tree)
│   └── pgc.py           # Perturbation Geometry Consistency (drug/MOA silhouette)
├── run_all.py           # run everything -> results/summary.json + table
└── results/             # per-metric json + summary
```

## Metrics

| Group | Metric | Input → target | Probe / computation | Reported |
|---|---|---|---|---|
| Decodability | **GER** | latent → expression | `nn.Linear`, MSE | gene/cell-wise Pearson & R² |
| Decodability | **CTD** | latent → cell type | linear softmax | Acc, macro-F1 |
| Decodability | **DD**  | latent → drug | linear softmax | Acc, macro-F1 |
| Decodability | **DoseD** | latent → dose | linear regression | MAE, R² |
| Decodability | **MOA-D** | latent → pathway(MOA) | linear softmax | Acc, macro-F1 |
| Organization | **EGC** | expr vs latent geometry | Mantel / distance corr | r, p, dcor |
| Organization | **LNP** | expr vs latent kNN | Jaccard overlap | Jaccard@k |
| Organization | **COG** | cell-type centroids vs ontology tree | Mantel | r, p |
| Organization | **PGC** | drug centroids vs MOA | silhouette, within/between | silhouette, ratio, p |

## How to run

```bash
ENV=/scratch/yuchen.yan/envs/squidiff/bin/python
export HDF5_USE_FILE_LOCKING=FALSE          # required on /scratch (hpfs)
cd semantic_study

# 1. extract embeddings from the trained Squidiff checkpoint (train + val)
$ENV extract/extract_squidiff_embeddings.py --split both

# 2. run the whole study
$ENV run_all.py
#    or a single metric:
$ENV decodability/ger.py
$ENV organization/pgc.py --moa_col pathway_level_1
```

Or submit both steps as one cluster job: `qsub submit_semantic_study.sh`.

## Comparing multiple encoders

The metrics are encoder-agnostic. To evaluate another model (scVI, scGen, trVAE,
scDisInFact, a foundation model, ...), write an extractor that produces
`config.emb_paths("<name>")` in the shared format (helpers in
[extract/_base.py](extract/_base.py)), then:

```bash
python run_all.py --encoder <name>     # results -> results/<name>/summary.json
```

Ready-made VAE extractors live in `extract/` (`extract_scvi.py`,
`extract_scgen.py`, `extract_trvae.py`, `extract_scdisinfact.py`). Environment
setup, the raw-counts caveat, and foundation-model recommendations are in
**[extract/ENCODERS.md](extract/ENCODERS.md)**. Each model needs its own conda
env; the metrics run in any env with scanpy/torch/sklearn.

## Notes specific to this checkpoint

- Squidiff's drug-structure encoder computes `z_sem = encoder(control_profile,
  drug_fingerprint × log10(dose+1))`. The latent therefore encodes **(control
  state, drug, dose)** and does **not** see the perturbed cell's own
  transcriptome. Expect strong DD / DoseD / MOA-D / CTD but weaker GER and
  weaker EGC/LNP against measured expression — that contrast is a real,
  interpretable finding about the encoder, not a bug in the metric.
- **COG is skipped for sciplex**: only 3 cancer cell lines (A549/MCF7/K562),
  which are not an ontology tree. Point it at a PBMC-style dataset with a
  `--ontology_csv` of Cell-Ontology tree distances to enable it.
- **MOA** is taken from the `pathway` column (20 classes). Use `--label_col
  pathway_level_1` / `target` to probe other levels of the Drug→MOA→Target→
  Pathway hierarchy.
