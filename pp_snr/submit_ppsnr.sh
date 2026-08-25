#!/bin/bash
#PBS -N ppsnr
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/ppsnr.out

# Energy-based PP-SNR in expression + 4 latent spaces, per perturbation with a
# matched (cell_line, plate) DMSO_TF control. Train and val computed separately.

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/pp_snr
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
nvidia-smi

# cap shards (shuffled -> covers all plates); ~500 gives each (cl,drug) plenty of
# cells for N_min=50 via sufficient statistics. Avoids the all-shard I/O timeout.
python -u energy_ppsnr.py --split train --n_min 50 --shard_budget 500
python -u energy_ppsnr.py --split val   --n_min 50 --shard_budget 500
echo "done: PP-SNR -> pp_snr/ppsnr_train.json + ppsnr_val.json"
