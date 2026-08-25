#!/bin/bash
# Dedicated Python 3.9 venv for scGen (2022 legacy stack). scGen 2.1.0 needs
# scvi-tools 0.16.4, which drags old anndata + old torch/lightning + a specific
# jax/flax/chex set. We DON'T pin jax/flax ourselves (that caused a deadlock on
# py3.10); we let scvi-tools 0.16.4 resolve them, on py3.9 where the old wheels
# exist. The jax-releases index is available in case an old jaxlib is needed.
set -x
BASE=/scratch/yuchen.yan/envs/squidiff/bin/python   # Python 3.9.25
ENV=/scratch/yuchen.yan/envs/sem_scgen_py39
export PIP_CACHE_DIR=/scratch/yuchen.yan/pip_cache_semantic
export TMPDIR=/scratch/yuchen.yan/tmp_semantic TMP=/scratch/yuchen.yan/tmp_semantic TEMP=/scratch/yuchen.yan/tmp_semantic
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

rm -rf "$ENV"
"$BASE" -m venv "$ENV"
P="$ENV/bin/pip"
$P install --no-input -U "pip<24" setuptools wheel

# ONE resolved command with hard pins on the whole 2022 stack, so nothing gets
# silently upgraded: old CUDA torch (for lightning 1.5) + a coherent old jax set
# (jax 0.3.25 still has jax.core.Shape; flax 0.5.3 predates jax.typing).
$P install --no-input \
    "torch==1.12.1+cu116" \
    "scvi-tools==0.16.4" "scgen==2.1.0" "anndata==0.8.0" "scanpy==1.9.3" \
    "numpy==1.23.5" "pandas<2" \
    "jax==0.3.25" "jaxlib==0.3.25" "chex==0.1.5" "flax==0.5.3" "optax==0.1.3" "ml_dtypes<0.3" \
    "pytorch-lightning==1.5.10" \
    --extra-index-url https://download.pytorch.org/whl/cu116 \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html

echo "=== import test ==="
"$ENV/bin/python" - <<'PYEOF'
for m in ["torch","scvi","scgen","anndata","scanpy"]:
    try:
        x=__import__(m); print("OK ",m,getattr(x,"__version__",""))
    except Exception as e:
        print("FAIL",m,"->",type(e).__name__,str(e)[:160])
import torch; print("torch cuda?", torch.version.cuda)
PYEOF
echo "=== SCGEN_PY39_DONE ==="
