#!/bin/bash
# Build one conda env per model (Python 3.10 + CUDA torch + that model's package).
# pip cache is shared across envs, so torch (~2.5GB) downloads once.
# Run:  nohup bash setup_envs.sh > results/env_permodel.log 2>&1 &
set -x
CONDA=/app1/ebapps/arches/flat-avx2/software/Miniconda3/24.7.1-0/bin/conda
ENVROOT=/scratch/yuchen.yan/envs

# The shared conda package cache is corrupted (SafetyError: size/hash mismatch on
# extraction). Use a FRESH, private package cache on /scratch so packages are
# re-downloaded clean. Also keep pip's cache on /scratch.
export CONDA_PKGS_DIRS=/scratch/yuchen.yan/conda_pkgs_semantic
export PIP_CACHE_DIR=/scratch/yuchen.yan/pip_cache_semantic
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

rm -rf "$ENVROOT/semantic_encoders"   # remove abandoned unified env

build () {   # $1 = env name ; $2... = extra pip packages (the model)
  local NAME=$1; shift
  local ENV="$ENVROOT/$NAME"
  echo "########## BUILD $ENV ##########"
  rm -rf "$ENV"
  # Use the 'defaults' channel only: conda just needs to provide python+pip (the
  # model packages come from pip). conda-forge's repodata is ~200MB and makes
  # metadata collection take 20+ min here.
  "$CONDA" create -y -p "$ENV" --override-channels -c defaults python=3.10 pip || { echo "CREATE_FAIL $NAME"; return; }
  "$ENV/bin/pip" install --no-input torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  "$ENV/bin/pip" install --no-input "$@" scanpy scikit-learn scikit-misc scipy
  echo "PIP_EXIT_${NAME}=$?"
}

build sem_scvi        scvi-tools
build sem_scgen       scgen
build sem_trvae       scarches
build sem_scdisinfact "git+https://github.com/ZhangLabGT/scDisInFact.git"

echo "=== IMPORT TESTS ==="
test_env () {   # $1 = env name ; $2 = module to import
  echo "--- $1 : import $2 ---"
  "$ENVROOT/$1/bin/python" -c "
import importlib
for mod in ['$2','scanpy','sklearn','torch']:
    try:
        m=importlib.import_module(mod); print('OK ',mod,getattr(m,'__version__',''))
    except Exception as e:
        print('FAIL',mod,'->',type(e).__name__,str(e)[:150])
try:
    import torch; print('   cuda build:',torch.version.cuda)
except Exception: pass
"
}
test_env sem_scvi        scvi
test_env sem_scgen       scgen
test_env sem_trvae       scarches
test_env sem_scdisinfact scdisinfact
echo "=== PERMODEL_DONE ==="
