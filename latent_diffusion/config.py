"""
Config for the latent-diffusion (LDM) stage.

Two-stage cell-state generation:
  Stage 1 (done in ../semantic_study): a VAE/encoder-decoder maps expression x
    -> latent z (encoder) and z -> x (decoder). Trained encoders already exist
    for scVI / scGen / trVAE / scDisInFact / Squidiff on sciplex + PBMC.
  Stage 2 (this package): a conditional diffusion/flow model learns p(z | cond)
    in that latent space; sampling gives z ~ p(z|cond) which the VAE decoder
    turns back into expression.

Design choices tied to the constraints:
  * LATENT-DIM AGNOSTIC. The latent dim is read from the embedding .h5ad at
    runtime (z.shape[1]), never hardcoded -- so it aligns to whatever dim each
    VAE was trained with (and auto-adapts after the dim-32 retrain).
  * VAE-agnostic training. Training reads only the saved latent embeddings +
    obs conditions produced by semantic_study/extract; no VAE needed to TRAIN.
    A decoder adapter (decoders/) is needed only to turn generated z back into
    expression (scVI adapter provided).
  * Checkpoints on /scratch (NOT the project dir; space).

Paths/labels are imported (read-only) from semantic_study.config so the LDM
stays aligned with the embeddings and never edits the semantic_study package.
"""
import importlib.util
import os
import sys

# Load semantic_study/config.py under a UNIQUE module name (both files are named
# config.py -> a plain `import config` would re-import THIS file). importlib
# avoids the collision and keeps semantic_study untouched.
_SEM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "semantic_study")
sys.path.append(_SEM_DIR)      # appended (not front) so it can't shadow LDM's own modules
_spec = importlib.util.spec_from_file_location("semantic_study_config", os.path.join(_SEM_DIR, "config.py"))
SEM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEM)

DATASET = SEM.DATASET          # sciplex | pbmc (via SEMANTIC_DATASET env)
EMB_DIR = SEM.EMB_DIR          # where extract/ wrote the embeddings
LABELS  = SEM.LABELS

# EVERYTHING that gets written lives on /scratch, NEVER in the project dir
# (limited space). LDM checkpoints go under the shared models dir, one folder per
# model, 'ldm_' prefixed to distinguish from the VAE checkpoints; generated
# samples go under a dedicated /scratch dir.
LDM_MODELS_DIR = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models"
SAMPLES_DIR    = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/ldm_generated"


def emb_paths(vae):
    """(train, val) latent embedding files for a given VAE (from semantic_study)."""
    return SEM.emb_paths(vae)


def ldm_dir(vae, tag=""):
    """Checkpoint dir: /scratch/.../models/ldm_<dataset>_<vae>[_<tag>]/ (on /scratch)."""
    name = f"ldm_{DATASET}_{vae}" + (f"_{tag}" if tag else "")
    return os.path.join(LDM_MODELS_DIR, name)


def samples_dir():
    """Where generated .h5ad go: /scratch/.../ldm_generated/<dataset>/ (on /scratch)."""
    return os.path.join(SAMPLES_DIR, DATASET)


# --------------------------------------------------------------------------- #
# Conditioning spec per dataset: list of (name, kind, obs_column).
#   kind = "categorical" (embedding table) | "continuous" (scalar -> MLP).
# Any subset can be selected at train time via --conditions.
# --------------------------------------------------------------------------- #
_COND_SPECS = {
    "sciplex": [
        ("cell_type", "categorical", LABELS["cell_type"]),   # A549/MCF7/K562
        ("drug",      "categorical", LABELS["drug"]),        # ~187 drugs
        ("dose",      "continuous",  LABELS["dose"]),        # nM, log10-scaled
    ],
    "pbmc": [
        ("cell_type", "categorical", LABELS["cell_type"]),   # 8 PBMC types
        ("condition", "categorical", LABELS["drug"]),        # ctrl / stim
    ],
    "tahoe": [
        ("cell_type", "categorical", LABELS["cell_type"]),   # 50 cancer cell lines
        ("drug",      "categorical", LABELS["drug"]),        # ~380 drugs (+ DMSO_TF control)
    ],                                                       # no dose column in Tahoe
}
CONDITION_SPEC = _COND_SPECS[DATASET]
CONTINUOUS_LOG10 = {"dose"}   # continuous vars to log10(x+1) before feeding
