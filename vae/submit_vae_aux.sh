#!/bin/bash
#PBS -N tahoe_vae_aux
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/tahoe_vae_aux.out

# Gaussian VAE + linear drug aux-head. Config identical to the baseline Gaussian
# VAE (532k steps / 3 epochs, bs512, AdamW lr1e-3, beta1 warmup5000) EXCEPT the
# lambda*CE(drug) term. Sweep lambda via -v LAMBDA.
#   qsub -v LAMBDA=20  submit_vae_aux.sh
#   qsub -v LAMBDA=50  submit_vae_aux.sh
#   qsub -v LAMBDA=100 submit_vae_aux.sh

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/vae
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
: "${LAMBDA:=20}"
OUT=/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models/tahoe_vae_aux${LAMBDA}
nvidia-smi

python -u train_aux.py --aux_weight "$LAMBDA" --out_dir "$OUT" \
    --max_steps 532000 --batch_size 512 --num_workers 8 --eval_every 10000
echo "done: aux-head VAE (lambda=$LAMBDA) -> $OUT"
