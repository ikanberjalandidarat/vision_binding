#!/usr/bin/env bash
#SBATCH --job-name=vision-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
if [[ "${1:-}" != "--reviewed-captures" ]]; then
  echo 'Inspect all 24 captures first; submit with --reviewed-captures.' >&2
  exit 2
fi
"$MC_BINDING_ENV/bin/python" -m mc_binding vision-pilot \
  --dataset "$MC_BINDING_SCRATCH/runs/vision-captures-04" \
  --config configs/vision_sweep.json \
  --output "$MC_BINDING_SCRATCH/runs/vision-sweep-01" \
  --reviewed-captures
