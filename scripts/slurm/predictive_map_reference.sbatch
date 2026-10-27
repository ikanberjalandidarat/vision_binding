#!/usr/bin/env bash
#SBATCH --job-name=predictive-map-ref
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/predictive-map-ref-%j.out
set -euo pipefail

# A separate preinstalled environment; this job never installs packages or
# edits the author's notebooks, existing Minecraft runs, or active envs.
scratch_root="/oscar/scratch/$USER/zef-binding"
REFERENCE_PYTHON="${REFERENCE_PYTHON:-$scratch_root/envs/predictive-map/bin/python}"
REFERENCE_REPO="${REFERENCE_REPO:-$scratch_root/vendor/predictive-coding-recovers-maps}"
REFERENCE_WEIGHTS="${REFERENCE_WEIGHTS:-$scratch_root/reference-weights}"
case_name="${1:?Use environment, path, circle, autoencoder, or circle_autoencoder}"
[[ "$case_name" == environment || "$case_name" == path || "$case_name" == circle || "$case_name" == autoencoder || "$case_name" == circle_autoencoder ]] || exit 2
[[ -x "$REFERENCE_PYTHON" ]] || { echo "Separate reference Python missing: $REFERENCE_PYTHON"; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD

"$REFERENCE_PYTHON" scripts/run_predictive_map_reference.py \
  --repo "$REFERENCE_REPO" --weights "$REFERENCE_WEIGHTS" --case "$case_name" --check-only
extra=()
if [[ "${PAPER_DECODER:-0}" == 1 ]]; then extra+=(--paper-decoder); fi
"$REFERENCE_PYTHON" scripts/run_predictive_map_reference.py \
  --repo "$REFERENCE_REPO" --weights "$REFERENCE_WEIGHTS" --case "$case_name" \
  --output "/oscar/scratch/$USER/zef-binding/runs/predictive-map-reference-${case_name}-${SLURM_JOB_ID}" \
  "${extra[@]}"
