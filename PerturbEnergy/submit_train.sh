#!/bin/bash
#PBS -N pertenergy
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/train.out

# Train PerturbEnergy on Tahoe-100M, then run the full verification pipeline.
#   qsub submit_train.sh
#   qsub -v STEPS=200000,OVERRIDES="langevin.steps=40 optim.energy_lr=5e-5" submit_train.sh

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/PerturbEnergy
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
: "${STEPS:=100000}"; : "${OVERRIDES:=}"
nvidia-smi

python -u build_vocab.py                       # no-op if the vocab already exists (cheap)
python -u train.py --set train.max_steps="$STEPS" $OVERRIDES
python -u evaluate.py --tasks recon predict retrieval probe --set $OVERRIDES
echo "done: PerturbEnergy (OVERRIDES=$OVERRIDES)"
