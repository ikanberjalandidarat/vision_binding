#!/usr/bin/env bash
#SBATCH --job-name=predictive-map-state
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/predictive-map-state-%j.out
set -euo pipefail

scratch_root="/oscar/scratch/$USER/zef-binding"
reference_python="${REFERENCE_PYTHON:-$scratch_root/envs/predictive-map/bin/python}"
reference_repo="${REFERENCE_REPO:-$scratch_root/vendor/predictive-coding-recovers-maps}"
[[ -x "$reference_python" ]] || { echo "Missing Python: $reference_python"; exit 1; }
mkdir -p logs
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$reference_python" scripts/audit_predictive_map_original_states.py \
  --repo "$reference_repo" --episodes 128 \
  --output "$scratch_root/runs/predictive-map-state-audit-${SLURM_JOB_ID}/report.json"
