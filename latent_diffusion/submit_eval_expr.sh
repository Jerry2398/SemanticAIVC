#!/bin/bash
#PBS -N ldm_expr_eval
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=64gb
#PBS -j oe
#PBS -o logs/ldm_expr_eval.out

# Expression-space (perturbation-prediction) evaluation of a trained LDM.
# Runs in the DECODER's env (loads the VAE decoder); CPU is enough.
#   qsub -v VAE=scgen,DECODER=scgen,METHOD=flow,SEMANTIC_DATASET=sciplex submit_eval_expr.sh

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/latent_diffusion
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module unload gcc/13.1.0

: "${VAE:?set VAE=...}"; : "${DECODER:?set DECODER=scvi|scgen|trvae}"; : "${METHOD:=flow}"
export SEMANTIC_DATASET="${SEMANTIC_DATASET:-sciplex}"
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
# Force CPU: this eval is light, no GPU is allocated, and scGen's legacy torch
# 1.12 can't run on the Hopper GPU (sm_90) even if it detects one.
export CUDA_VISIBLE_DEVICES=""

# pick the decoder's environment (VAE package lives there)
case "$DECODER" in
  scvi)  ENVP=/scratch/yuchen.yan/envs/sem_scvi ;;
  scgen) ENVP=/scratch/yuchen.yan/envs/sem_scgen_py39 ;;
  trvae) ENVP=/scratch/yuchen.yan/envs/sem_trvae ;;
  *) echo "unknown DECODER=$DECODER"; exit 2 ;;
esac
# these envs are venvs (not conda) -> call their python directly
PY="$ENVP/bin/python"

exec > "logs/ldm_expr_${SEMANTIC_DATASET}_${VAE}_${METHOD}.out" 2>&1
"$PY" -u evaluate_expression.py --vae "$VAE" --decoder "$DECODER" --method "$METHOD" \
    --guidance 1.5 --n_per_group 300 --max_groups 60 --top_degs 50
echo "done: expression-eval json in /scratch/.../ldm_generated/${SEMANTIC_DATASET}/eval/"
