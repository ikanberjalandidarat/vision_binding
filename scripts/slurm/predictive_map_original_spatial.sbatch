#!/usr/bin/env bash
#SBATCH --job-name=predictive-map-spatial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/predictive-map-spatial-%j.out
set -euo pipefail
# Merge Python, Git, and native-library errors into the advertised job log.
exec 2>&1
trap 'code=$?; printf "Spatial job exiting: code=%s time=%s\n" "$code" "$(date -Is)"' EXIT

manifest="${1:?Pass the completed path manifest JSON}"
scratch_root="/oscar/scratch/$USER/zef-binding"
reference_python="${REFERENCE_PYTHON:-$scratch_root/envs/predictive-map/bin/python}"
reference_repo="${REFERENCE_REPO:-$scratch_root/vendor/predictive-coding-recovers-maps}"
reference_checkpoint="${REFERENCE_CHECKPOINT:-$scratch_root/reference-weights/predictive_coding.ckpt}"
output="${2:-$scratch_root/runs/predictive-map-original-spatial-${SLURM_JOB_ID}}"
resume_args=()
if [[ $# -ge 2 ]]; then
  resume_args=(--resume)
fi
[[ -x "$reference_python" && -s "$manifest" && -s "$reference_checkpoint" ]] || {
  echo 'Missing Python, completed manifest, or checkpoint'; exit 1;
}
mkdir -p logs
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$reference_python" scripts/run_predictive_map_original_spatial.py \
  --repo "$reference_repo" --checkpoint "$reference_checkpoint" \
  --manifest "$manifest" \
  --output "$output" "${resume_args[@]}"
