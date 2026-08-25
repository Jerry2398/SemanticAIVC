#!/bin/bash
#PBS -N tahoe_drughead
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/tahoe_drughead.out

# Gaussian VAE + drug-supervised head. Config identical to the baseline Gaussian
# VAE (532k steps, bs512, AdamW lr1e-3, beta1 warmup5000) except the lambda*aux term.
#   qsub -v MODE=ce_mlp,LAMBDA=20 submit_drughead.sh   # dir.2 nonlinear identity head
#   qsub -v MODE=ecfp,LAMBDA=50   submit_drughead.sh   # dir.1 ECFP regression
#   qsub -v MODE=infonce,LAMBDA=100 submit_drughead.sh # dir.1 InfoNCE contrastive

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/vae
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
: "${MODE:=ce_mlp}"; : "${LAMBDA:=50}"
# tag: ce_mlp->auxmlp, ecfp->auxecfp, infonce->auxnce
case "$MODE" in
  ce_mlp)  TAG=auxmlp ;;
  ecfp)    TAG=auxecfp ;;
  infonce) TAG=auxnce ;;
  ce_linear) TAG=aux ;;
  *) echo "unknown MODE=$MODE"; exit 1 ;;
esac
OUT=/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models/tahoe_vae_${TAG}${LAMBDA}
nvidia-smi

python -u train_drughead.py --aux_mode "$MODE" --aux_weight "$LAMBDA" --out_dir "$OUT" \
    --max_steps 532000 --batch_size 512 --num_workers 8 --eval_every 10000
echo "done: drughead MODE=$MODE lambda=$LAMBDA -> $OUT"
