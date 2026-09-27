#!/usr/bin/env bash
#SBATCH --job-name=vision-late
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
kind="${1:-residual}"
dataset="${2:-vision-captures-depth-01}"
case "$kind" in residual|q|k|v) ;; *) echo 'Choose residual, q, k or v' >&2; exit 2;; esac
[[ "$dataset" =~ ^vision-captures-[A-Za-z0-9_-]+$ ]] || exit 2
[[ "${3:-}" == '--reviewed-captures' ]] || { echo 'Inspect captures then pass --reviewed-captures as third argument' >&2; exit 2; }
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
"$MC_BINDING_ENV/bin/python" -m mc_binding vision-pilot \
 --dataset "$MC_BINDING_SCRATCH/runs/$dataset" \
 --config "configs/vision_late_${kind}.json" \
 --output "$MC_BINDING_SCRATCH/runs/vision-late-${kind}-${dataset}" \
 --reviewed-captures
