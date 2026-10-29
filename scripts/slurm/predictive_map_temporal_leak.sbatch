#!/usr/bin/env bash
#SBATCH --job-name=predictive-map-leak
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/predictive-map-leak-%j.out
set -euo pipefail

scratch_root="/oscar/scratch/$USER/zef-binding"
reference_python="${REFERENCE_PYTHON:-$scratch_root/envs/predictive-map/bin/python}"
reference_repo="${REFERENCE_REPO:-$scratch_root/vendor/predictive-coding-recovers-maps}"
reference_checkpoint="${REFERENCE_CHECKPOINT:-$scratch_root/reference-weights/predictive_coding.ckpt}"
[[ -x "$reference_python" && -s "$reference_checkpoint" ]] || {
  echo 'Missing reference Python or checkpoint'; exit 1;
}
mkdir -p logs
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$reference_python" scripts/check_predictive_map_temporal_leak.py \
  --repo "$reference_repo" --checkpoint "$reference_checkpoint" \
  --output "$scratch_root/runs/predictive-map-temporal-leak-${SLURM_JOB_ID}/report.json"
