#!/usr/bin/env bash
#SBATCH --job-name=revisit-pixels
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/revisit-pixels-%j.out
set -euo pipefail
scratch="/oscar/scratch/$USER/zef-binding"
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python="${PROBE_PYTHON:-$scratch/envs/predictive-map/bin/python}"
"$python" -m mc_binding.revisit_retrieval "${1:?Pass feature cache}" \
 "$scratch/runs/revisit-pixels-$SLURM_JOB_ID" --pixel-dataset "${2:?Pass revisit recordings}"
