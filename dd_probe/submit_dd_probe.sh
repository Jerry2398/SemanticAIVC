#!/bin/bash
#PBS -N dd_probe
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/dd_probe.out

# Ceiling check: decode drug identity directly from raw expression (2-layer MLP),
# same cells/protocol as the semantic-study DD -> is the VAE the bottleneck?
# (Cloned from the known-good submit_vae_aux.sh spec: 12h walltime, ncpus16/mem96gb.)

cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/dd_probe
mkdir -p logs
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3; conda init bash; source ~/.bashrc
module load CUDA/12.1.0; module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/squidiff

export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/yuchen.yan/tmp_semantic; mkdir -p "$TMPDIR"
nvidia-smi

python -u expr_dd_probe.py --probe both
echo "done: expression->drug ceiling probe -> dd_probe/results.json"
