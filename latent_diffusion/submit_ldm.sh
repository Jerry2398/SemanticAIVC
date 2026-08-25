#!/bin/bash
#PBS -N ldm
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:mem=48gb:ngpus=1
#PBS -j oe
#PBS -o logs/ldm.out

# Train + evaluate a latent-diffusion model on a VAE's latent space.
#   qsub -v VAE=scvi,METHOD=flow,SEMANTIC_DATASET=sciplex submit_ldm.sh
#   qsub -v VAE=scgen,METHOD=ddpm,SEMANTIC_DATASET=pbmc   submit_ldm.sh
# Trains only on the saved latent embeddings (no VAE needed at train time).

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/latent_diffusion
mkdir -p logs

source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff     # torch2.5 + scanpy + sklearn (GPU-capable)

export SEMANTIC_DATASET="${SEMANTIC_DATASET:-sciplex}"
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
: "${VAE:?set VAE=scvi|scgen|trvae|scdisinfact|squidiff_nodrug}"
: "${METHOD:=flow}"

exec > "logs/ldm_${SEMANTIC_DATASET}_${VAE}_${METHOD}.out" 2>&1
nvidia-smi

python -u train.py --vae "$VAE" --method "$METHOD" --save
python -u evaluate.py --vae "$VAE" --method "$METHOD" --guidance 1.5

echo "done: LDM ckpt in /scratch/.../models/ldm_${SEMANTIC_DATASET}_${VAE}_${METHOD}"
