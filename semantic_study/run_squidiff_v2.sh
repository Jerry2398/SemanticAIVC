#!/bin/bash
# Extract embeddings + compute all semantic metrics for both Squidiff v2 variants.
# CPU is fine (tiny encoder + linear probes). Results -> results/squidiff_nodrug/,
# results/squidiff_drug/.
set -e
cd /nfs/home/svu/yuchen.yan/yuchen_workspace/SemanticDiff/semantic_study
PY=/scratch/yuchen.yan/envs/squidiff/bin/python
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=16

echo "########## squidiff_nodrug ##########"
$PY -u extract/extract_squidiff_embeddings.py --variant nodrug --split both
$PY -u run_all.py --encoder squidiff_nodrug

echo "########## squidiff_drug ##########"
$PY -u extract/extract_squidiff_embeddings.py --variant drug --split both
$PY -u run_all.py --encoder squidiff_drug

echo "########## SQUIDIFF_V2_DONE ##########"
