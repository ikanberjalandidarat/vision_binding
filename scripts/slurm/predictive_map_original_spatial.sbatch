#!/usr/bin/env bash
#SBATCH --job-name=predictive-map-spatial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/predictive-map-spatial-%j.out
set -euo pipefail

manifest="${1:?Pass the completed path manifest JSON}"
scratch_root="/oscar/scratch/$USER/zef-binding"
reference_python="${REFERENCE_PYTHON:-$scratch_root/envs/predictive-map/bin/python}"
reference_repo="${REFERENCE_REPO:-$scratch_root/vendor/predictive-coding-recovers-maps}"
reference_checkpoint="${REFERENCE_CHECKPOINT:-$scratch_root/reference-weights/predictive_coding.ckpt}"
[[ -x "$reference_python" && -s "$manifest" && -s "$reference_checkpoint" ]] || {
  echo 'Missing Python, completed manifest, or checkpoint'; exit 1;
}
mkdir -p logs
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$reference_python" scripts/run_predictive_map_original_spatial.py \
  --repo "$reference_repo" --checkpoint "$reference_checkpoint" \
  --manifest "$manifest" \
  --output "$scratch_root/runs/predictive-map-original-spatial-${SLURM_JOB_ID}"
