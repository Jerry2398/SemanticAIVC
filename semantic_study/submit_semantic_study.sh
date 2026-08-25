#!/bin/bash
#PBS -N semantic_study
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=64gb:ngpus=1
#PBS -j oe
#PBS -o logs/semantic_study.out

# Full semantic study: extract Squidiff embeddings, then run every metric.
cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/semantic_study
mkdir -p logs results

# 1. Environment (same env used to train Squidiff)
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3
conda init bash
source ~/.bashrc
module load CUDA/12.1.0
module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

# 2. Critical on /scratch (hpfs): HDF5 file locking hangs otherwise.
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=16

PY=$(which python)
nvidia-smi || echo "no GPU (encoder + probes run fine on CPU too)"

# 3. Extract embeddings (train + val) from the trained checkpoint.
$PY extract/extract_squidiff_embeddings.py --split both

# 4. Run all semantic metrics -> results/summary.json
$PY run_all.py

echo "semantic study done. see results/summary.json"
