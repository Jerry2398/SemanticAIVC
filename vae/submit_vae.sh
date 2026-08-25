#!/bin/bash
#PBS -N tahoe_vae
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/tahoe_vae.out

# Train the simple VAE on Tahoe-100M (streaming from parquet shards) + validate.
#   qsub -v LATENT=32,MAX_STEPS=30000 submit_vae.sh

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/vae
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff     # torch2.5 GPU + pyarrow + scanpy

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
# full-data default: ~3 epochs over the 90.8M-cell train split (177.4k steps/epoch @ bs512)
: "${LATENT:=32}"; : "${MAX_STEPS:=532000}"; : "${EVAL_EVERY:=10000}"
nvidia-smi

python -u train.py --latent_dim "$LATENT" --max_steps "$MAX_STEPS" \
    --batch_size 512 --num_workers 8 --eval_every "$EVAL_EVERY"
python -u evaluate.py --max_batches 200
echo "done: VAE ckpt + metrics.csv + loss_curve.png in /scratch/.../models/tahoe_vae"
