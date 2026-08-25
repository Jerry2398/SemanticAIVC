#!/bin/bash
#PBS -N pbulk_multi
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/pseudobulk_multi.out

# Pseudobulk DD across feature{raw,lognorm,latent} x grouping{mixed,clean} x B{1,16,64}.
cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/dd_probe
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
nvidia-smi

python -u pseudobulk_multi.py --Bs 1 16 64 --k_cl 8 --cap_train 300 --cap_val 150 \
    --train_shards 1000 --val_shards 170 --n_bulk_cap 16
echo "done: pseudobulk_multi -> dd_probe/pseudobulk_multi_results.json"
