#!/bin/bash
#PBS -N pertenergy_eval
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/eval.out

# Verification only, on an existing checkpoint.
#   qsub submit_eval.sh
#   qsub -v TASKS="predict retrieval" submit_eval.sh

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/PerturbEnergy
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
: "${TASKS:=recon predict retrieval probe}"
python -u evaluate.py --tasks $TASKS
