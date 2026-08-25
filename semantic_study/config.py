"""
Central configuration for the semantic study of cell-representation encoders.

The study is intentionally *encoder-agnostic*: only the extract/ step knows about
a specific model. Every metric consumes a pair of embedding .h5ad files (train /
val) written in the format documented in `common.py`.

Canonical dataset (v2): sci-Plex-3 with REAL raw counts, sourced from scPerturb
(SrivatsanTrapnell2020_sciplex3.h5ad, Zenodo 13350497) and rebuilt by
prep_sciplex_counts.py into 5000-HVG train/val files that carry BOTH raw counts
(layers['counts']) and log-normalized expression (X). All encoders are trained
and evaluated on this single dataset so comparisons isolate the encoder.
"""
import os

# --------------------------------------------------------------------------- #
# Dataset selector: `SEMANTIC_DATASET=sciplex` (default) or `pbmc`.
# All paths / labels / metric applicability below switch on this, so the whole
# pipeline (prep excepted) is dataset-agnostic. sciplex values are unchanged.
# --------------------------------------------------------------------------- #
DATASET = os.environ.get("SEMANTIC_DATASET", "sciplex")

# Unified latent dimensionality for ALL encoders, so geometry/probe metrics are
# compared apples-to-apples (no model wins just by having a bigger latent).
# Every extractor reads this; Squidiff's diffusion encoder reads it as sem_dim.
LATENT_DIM = int(os.environ.get("SEMANTIC_LATENT_DIM", 32))

SQUIDIFF_ROOT = "/home/svu/yuchen.yan/yuchen_workspace/Squidiff"
MODELS_DIR = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models"
_REPO_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_DATASETS_ROOT = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets"

if DATASET == "sciplex":
    COUNTS_DIR      = os.path.join(_DATASETS_ROOT, "sciplex_counts")
    RAW_COUNTS_FILE = os.path.join(COUNTS_DIR, "SrivatsanTrapnell2020_sciplex3.h5ad")
    GENE_SIZE       = 5000
    _PREFIX         = f"sciplex3_hvg{GENE_SIZE}"
    RESULTS_DIR     = os.path.join(_REPO_RESULTS, "sciplex")
    LABELS = dict(cell_type="cell_type", drug="condition", dose="dose",
                  moa="pathway", smiles="SMILES")
    COG_SUPPORTED = False       # only 3 cell lines
    ONTOLOGY_CSV  = None
    METRICS = ["ger", "ctd", "dd", "dose_d", "moa_d", "egc", "lnp", "cog", "pgc"]
    BATCH_KEY = "plate"                              # trVAE / scDisInFact batch axis
    SCDISINFACT_CONDITION_COLS = ["condition", "dose"]

elif DATASET == "pbmc":
    # Kang 2018 IFN-beta PBMC (figshare kang.h5ad): raw counts, 8 cell types,
    # label=ctrl/stim (single perturbation), replicate=patient (batch).
    COUNTS_DIR      = os.path.join(_DATASETS_ROOT, "pbmc_kang")
    RAW_COUNTS_FILE = os.path.join(COUNTS_DIR, "kang.h5ad")
    GENE_SIZE       = 2000
    _PREFIX         = f"pbmc_hvg{GENE_SIZE}"
    RESULTS_DIR     = os.path.join(_REPO_RESULTS, "pbmc")
    LABELS = dict(cell_type="cell_type", drug="condition", dose=None,
                  moa=None, smiles=None)
    COG_SUPPORTED = True                     # 8 cell types -> Cell-Ontology tree
    ONTOLOGY_CSV  = os.path.join(COUNTS_DIR, "cog_ontology_distances.csv")
    # single perturbation, no dose/MOA/multiple drugs -> DoseD/MOA-D/PGC N/A
    METRICS = ["ger", "ctd", "dd", "egc", "lnp", "cog"]
    BATCH_KEY = "batch"                       # = patient/replicate
    SCDISINFACT_CONDITION_COLS = ["condition"]   # ctrl/stim only (no dose)
elif DATASET == "tahoe":
    # Tahoe-100M drug screen (native domain of the three Tahoe-trained VAEs).
    # Embeddings are written directly by extract/extract_tahoe_vae.py (encoding
    # subsampled Tahoe train/val cells), so no *_train.h5ad expression file is
    # built here -- write_embedding gets explicit obs+expr and metrics read the
    # embedding .h5ad only. Labels come from Tahoe obs columns.
    COUNTS_DIR      = os.path.join(_DATASETS_ROOT, "tahoe")
    RAW_COUNTS_FILE = None
    GENE_SIZE       = 5000                            # Tahoe HVG panel
    _PREFIX         = "tahoe"                         # (no on-disk expr file; kept for path fmt)
    RESULTS_DIR     = os.path.join(_REPO_RESULTS, "tahoe")
    # Tahoe has drug / cell_line / MOA but NO dose and NO cell-ontology.
    LABELS = dict(cell_type="cell_line_id", drug="drug", dose=None,
                  moa="moa-fine", smiles="canonical_smiles")
    COG_SUPPORTED = False                             # cancer cell lines, no Cell-Ontology tree
    ONTOLOGY_CSV  = None
    # DoseD skipped (no dose); COG skipped (no ontology). The rest apply.
    METRICS = ["ger", "ctd", "dd", "moa_d", "egc", "lnp", "pgc"]
    BATCH_KEY = "plate"
    SCDISINFACT_CONDITION_COLS = ["drug"]

else:
    raise ValueError(f"unknown SEMANTIC_DATASET={DATASET!r}")

TRAIN_DATA_PATH = os.path.join(COUNTS_DIR, f"{_PREFIX}_train.h5ad")
VAL_DATA_PATH   = os.path.join(COUNTS_DIR, f"{_PREFIX}_val.h5ad")
EMB_DIR         = os.path.join(COUNTS_DIR, "embeddings")

# --- deprecated v1 (PRnet log-norm, 200 HVG, no counts) -- kept for reference --
_V1_DATASETS = os.path.join(_DATASETS_ROOT, "sciplex")


def emb_paths(encoder="squidiff"):
    """(train_emb, val_emb) for an encoder. Extractors write here; metrics read here."""
    return (os.path.join(EMB_DIR, f"{encoder}_train.h5ad"),
            os.path.join(EMB_DIR, f"{encoder}_val.h5ad"))


def models_dir(name):
    """Checkpoint dir for a model, namespaced by dataset (sciplex kept flat for
    back-compat with already-trained checkpoints)."""
    return os.path.join(MODELS_DIR, name if DATASET == "sciplex" else f"{DATASET}_{name}")


# squidiff default embedding paths (kept as attributes for back-compat with metrics)
TRAIN_EMB_PATH, VAL_EMB_PATH = emb_paths("squidiff")

# --------------------------------------------------------------------------- #
# Squidiff (v2, 5000 genes). Two variants are trained, identical to the official
# config except for the data-driven changes (gene_size=5000, v2 data paths):
#   * nodrug : use_drug_structure=False -> z_sem = encoder(x_expr)  (expression-only)
#   * drug   : use_drug_structure=True  -> z_sem = encoder(control, drug x dose)
# Checkpoints go to separate folders under MODELS_DIR.
# --------------------------------------------------------------------------- #
_SQUIDIFF_COMMON = dict(gene_size=GENE_SIZE, output_dim=GENE_SIZE,
                        num_layers=3, use_encoder=True, class_cond=False,
                        sem_dim=LATENT_DIM)
SQUIDIFF_NODRUG_CFG = dict(_SQUIDIFF_COMMON, use_drug_structure=False)
SQUIDIFF_DRUG_CFG   = dict(_SQUIDIFF_COMMON, use_drug_structure=True, drug_dimension=1024)

SQUIDIFF_NODRUG_MODEL_PATH = os.path.join(models_dir("squidiff_nodrug"), "model.pt")
SQUIDIFF_DRUG_MODEL_PATH   = os.path.join(models_dir("squidiff_drug"), "model.pt")

# drug-mode training data (SMILES + row-aligned control), built by
# prep_squidiff_drug.py from the canonical treated files.
TRAIN_DRUG_PATH         = os.path.join(COUNTS_DIR, f"sciplex3_hvg{GENE_SIZE}_train_drug.h5ad")
TRAIN_DRUG_CONTROL_PATH = os.path.join(COUNTS_DIR, f"sciplex3_hvg{GENE_SIZE}_train_drug_control.h5ad")
VAL_DRUG_PATH           = os.path.join(COUNTS_DIR, f"sciplex3_hvg{GENE_SIZE}_val_drug.h5ad")
VAL_DRUG_CONTROL_PATH   = os.path.join(COUNTS_DIR, f"sciplex3_hvg{GENE_SIZE}_val_drug_control.h5ad")

# PRnet file used as the drug-name -> SMILES source (scPerturb has no SMILES).
PRNET_SMILES_FILE = os.path.join(_V1_DATASETS, "sci_plex_random_split_0_train.h5ad")

DOSE_LOG10 = True       # dose is log10-scaled before regression (DoseD)
