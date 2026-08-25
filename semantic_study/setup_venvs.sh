#!/bin/bash
# Build one venv per model, using an existing Python 3.10 interpreter (conda
# env-creation hangs on repodata download here, so we bypass conda entirely).
# Each venv: py3.10 + CUDA torch + that model's package + scanpy/sklearn.
# pip cache is shared -> torch (~2.5GB) downloads once.
set -x
BASE=/scratch/yuchen.yan/envs/apd_yyc/bin/python   # a stable Python 3.10.20
ENVROOT=/scratch/yuchen.yan/envs
export PIP_CACHE_DIR=/scratch/yuchen.yan/pip_cache_semantic
# /tmp is only ~1GB and HOME is ~94% full; the torch wheel unpacks to several GB,
# so send pip's cache AND temp/unpack dir to /scratch (895G free) -> avoids Errno 28.
export TMPDIR=/scratch/yuchen.yan/tmp_semantic
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

build () {   # $1 = env name ; $2... = model pip package(s)
  local NAME=$1; shift
  local ENV="$ENVROOT/$NAME"
  echo "########## BUILD $ENV ##########"
  rm -rf "$ENV"
  "$BASE" -m venv "$ENV" || { echo "VENV_FAIL $NAME"; return; }
  "$ENV/bin/pip" install --no-input -U pip setuptools wheel
  "$ENV/bin/pip" install --no-input torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  "$ENV/bin/pip" install --no-input "$@" scanpy scikit-learn scikit-misc scipy
  echo "PIP_EXIT_${NAME}=$?"
}

build sem_scvi        scvi-tools
build sem_scgen       scgen
build sem_trvae       scarches
build sem_scdisinfact "git+https://github.com/ZhangLabGT/scDisInFact.git"

echo "=== IMPORT TESTS ==="
test_env () {
  echo "--- $1 : import $2 ---"
  "$ENVROOT/$1/bin/python" -c "
import importlib
for mod in ['$2','scanpy','sklearn','torch']:
    try:
        m=importlib.import_module(mod); print('OK ',mod,getattr(m,'__version__',''))
    except Exception as e:
        print('FAIL',mod,'->',type(e).__name__,str(e)[:150])
try:
    import torch; print('   cuda build:',torch.version.cuda,'| cuda avail (CPU node -> False ok):',torch.cuda.is_available())
except Exception: pass
"
}
test_env sem_scvi        scvi
test_env sem_scgen       scgen
test_env sem_trvae       scarches
test_env sem_scdisinfact scdisinfact
echo "=== VENVS_DONE ==="
