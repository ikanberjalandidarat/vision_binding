#!/usr/bin/env bash
#SBATCH --job-name=predictive-map-png
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/predictive-map-png-%j.out
set -euo pipefail

scratch_root="/oscar/scratch/$USER/zef-binding"
reference_python="${REFERENCE_PYTHON:-$scratch_root/envs/predictive-map/bin/python}"
reference_repo="${REFERENCE_REPO:-$scratch_root/vendor/predictive-coding-recovers-maps}"
reference_checkpoint="${REFERENCE_CHECKPOINT:-$scratch_root/reference-weights/predictive_coding.ckpt}"
episodes="${1:-8}"
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || { echo 'Episode count must be positive'; exit 2; }
[[ -x "$reference_python" ]] || { echo "Missing Python: $reference_python"; exit 1; }
mkdir -p logs
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$reference_python" scripts/run_predictive_map_original_pngs.py \
  --repo "$reference_repo" --checkpoint "$reference_checkpoint" \
  --episodes "$episodes" \
  --output "$scratch_root/runs/predictive-map-original-pngs-${SLURM_JOB_ID}"
