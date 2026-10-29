#!/usr/bin/env bash
#SBATCH --job-name=author-distance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/author-distance-%j.out
set -euo pipefail
exec 2>&1
scratch_root="/oscar/scratch/$USER/zef-binding"
reference_python="${REFERENCE_PYTHON:-$scratch_root/envs/predictive-map/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MPLBACKEND=Agg
unset LD_LIBRARY_PATH LD_PRELOAD
"$reference_python" scripts/run_predictive_map_author_distance.py \
  --repo "$scratch_root/vendor/predictive-coding-recovers-maps" \
  --weights "$scratch_root/reference-weights" \
  --output "$scratch_root/runs/predictive-map-author-distance-${SLURM_JOB_ID}"
