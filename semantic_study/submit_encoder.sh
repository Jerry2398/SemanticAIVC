#!/bin/bash
#PBS -N sem_encoder
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=64gb:ngpus=1
#PBS -j oe
#PBS -o logs/sem_encoder.out

# Train ONE encoder, extract its embeddings, and compute all semantic metrics.
# Pick the model at submit time:
#   qsub -v MODEL=scvi        submit_encoder.sh
#   qsub -v MODEL=scgen       submit_encoder.sh
#   qsub -v MODEL=trvae       submit_encoder.sh
#   qsub -v MODEL=scdisinfact submit_encoder.sh
# (Submit all four together -> they run on separate GPU nodes in parallel.)

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/semantic_study
mkdir -p logs results

# 1. Per-model venv (each has its model package + scanpy/sklearn, so both
#    extraction and metrics run in it). These are venvs, not conda envs.
source /app1/ebapps/ebenv_hopper.sh
module load CUDA/12.1.0   # CUDA runtime for the cu121 torch wheel

# dataset selector (sciplex default; pass SEMANTIC_DATASET=pbmc via -v). Every
# extractor/metric reads this through config.py.
export SEMANTIC_DATASET="${SEMANTIC_DATASET:-sciplex}"

: "${MODEL:?set MODEL=scvi|scgen|trvae|scdisinfact via 'qsub -v MODEL=...[,SEMANTIC_DATASET=pbmc]'}"
# METRICS_PY runs run_all.py; default = the model env's own python. scGen's env
# is a legacy py3.9 stack with old anndata that can't read the modern embedding
# files, so its metrics run in the modern squidiff env instead.
METRICS_PY=python
case "$MODEL" in
  scvi)        SCRIPT=extract/extract_scvi.py;        ENVP=/scratch/yuchen.yan/envs/sem_scvi ;;
  scgen)       SCRIPT=extract/extract_scgen.py;       ENVP=/scratch/yuchen.yan/envs/sem_scgen_py39
               METRICS_PY=/scratch/yuchen.yan/envs/squidiff/bin/python ;;
  trvae)       SCRIPT=extract/extract_trvae.py;       ENVP=/scratch/yuchen.yan/envs/sem_trvae ;;
  scdisinfact) SCRIPT=extract/extract_scdisinfact.py; ENVP=/scratch/yuchen.yan/envs/sem_scdisinfact ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
source "$ENVP/bin/activate"

# per-(dataset,model) log (the static PBS -o would otherwise be clobbered)
exec > "logs/sem_${SEMANTIC_DATASET}_${MODEL}.out" 2>&1

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=16
# keep temp off the tiny /tmp and near-full HOME
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
# reduce CUDA fragmentation (torch>=2 only; scGen's torch 1.12 rejects this option)
if [ "$MODEL" != "scgen" ]; then export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; fi
nvidia-smi

echo "=== [$MODEL] train + extract embeddings (+ save checkpoint) ==="
python -u "$SCRIPT" --save_model

echo "=== [$MODEL] compute semantic metrics (via $METRICS_PY) ==="
HDF5_USE_FILE_LOCKING=FALSE "$METRICS_PY" -u run_all.py --encoder "$MODEL"

echo "done: embeddings in the sciplex_counts/embeddings dir, metrics in results/$MODEL/"
