#!/bin/bash
#PBS -N sciplex_prep
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:mem=96gb
#PBS -j oe
#PBS -o logs/sciplex_prep.out

# Build the canonical v2 dataset (5000-HVG, counts + log-norm) from scPerturb
# raw counts. CPU only; ~96 GB is comfortable for the two-pass loader.
cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/semantic_study
mkdir -p logs

source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3
conda init bash
source ~/.bashrc
module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=8

# For seurat_v3 HVG (recommended for scVI). Harmless if it fails -> falls back to 'seurat'.
pip install --quiet scikit-misc 2>/dev/null || echo "scikit-misc not installed; using seurat flavor"

python prep_sciplex_counts.py \
    --in_path  /scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/sciplex_counts/SrivatsanTrapnell2020_sciplex3.h5ad \
    --out_dir  /scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/datasets/sciplex_counts \
    --n_top_genes 5000 --val_frac 0.25 --seed 0

echo "prep done. canonical files in the sciplex_counts folder."
