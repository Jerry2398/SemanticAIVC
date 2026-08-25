#!/bin/bash
#PBS -N tahoe_pipe_heads
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/tahoe_pipeline_heads.out

# Full Tahoe-native comparison for the 3 VAEs (Gaussian / NB / scGen-aligned):
#   1. extract embeddings (frozen VAE encodes Tahoe train+val subsamples)
#   2. semantic study on Tahoe val (GER/CTD/DD/MOA-D/EGC/LNP/PGC; DoseD+COG skipped)
#   3. train latent diffusion (IDENTICAL config across VAEs) on the train latents
#   4. perturbation-prediction eval on val (DMSO_TF baseline)
# Everything runs in the squidiff env (our own VAE decoders -> no legacy env).

# NOTE: no `set -u` -- sourcing ~/.bashrc references unbound vars (BASHRCSOURCED).
ROOT=/nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff
cd "$ROOT/latent_diffusion"; mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export SEMANTIC_DATASET=tahoe
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
PY=python
nvidia-smi

VAES="tahoe_vae_auxmlp20 tahoe_vae_auxmlp50 tahoe_vae_auxmlp100 tahoe_vae_auxecfp20 tahoe_vae_auxecfp50 tahoe_vae_auxecfp100 tahoe_vae_auxnce20 tahoe_vae_auxnce50 tahoe_vae_auxnce100"

# ---- identical LDM config for ALL VAEs (fair comparison) ----
LDM_ARGS="--method flow --hidden 256 --depth 6 --epochs 400 --batch_size 1024 --lr 1e-3 --p_uncond 0.1 --save"
EVAL_ARGS="--method flow --guidance 1.5 --steps 100 --n_per_group 300 --max_groups 60 --min_cells 50 --top_degs 50 --control_value DMSO_TF"

for VAE in $VAES; do
  echo "############################################################"
  echo "### $VAE : 1) EXTRACT"
  echo "############################################################"
  $PY -u "$ROOT/semantic_study/extract/extract_tahoe_vae.py" --vae "$VAE" \
      --n_train 120000 --train_cells_per_shard 500 \
      --n_val_broad 45000 --val_broad_per_shard 350 \
      --n_powered_cls 6 --max_powered 150 --cap_grp 80 --cap_ctrl 250 || echo "!!! extract $VAE FAILED"

  echo "### $VAE : 2) SEMANTIC STUDY"
  $PY -u "$ROOT/semantic_study/run_all.py" --encoder "$VAE" || echo "!!! semantic $VAE FAILED"

  echo "### $VAE : 3) TRAIN LDM"
  $PY -u train.py --vae "$VAE" $LDM_ARGS || echo "!!! ldm-train $VAE FAILED"

  echo "### $VAE : 4) PERTURBATION EVAL"
  $PY -u evaluate_expression.py --vae "$VAE" --decoder "$VAE" $EVAL_ARGS || echo "!!! ldm-eval $VAE FAILED"
done

echo "TAHOE_PIPELINE_DONE"
