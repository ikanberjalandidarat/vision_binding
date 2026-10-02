#!/usr/bin/env bash
#SBATCH --job-name=binding-model
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
mode="${1:?Choose baseline, strength, or heads}"
dataset="${2:?Supply binding-captures directory name}"
case "$mode" in baseline) task=binding-baseline;; strength|heads) task=binding-patch;; *) exit 2;; esac
[[ "$dataset" =~ ^binding-captures-[A-Za-z0-9_-]+$ ]] || exit 2
[[ "${3:-}" == '--reviewed-captures' ]] || { echo 'Review captures first and pass --reviewed-captures' >&2; exit 2; }
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
"$MC_BINDING_ENV/bin/python" -m mc_binding "$task" \
 --dataset "$MC_BINDING_SCRATCH/runs/$dataset" \
 --config "configs/binding_${mode}.json" \
 --output "$MC_BINDING_SCRATCH/runs/binding-${mode}-${SLURM_JOB_ID}" --reviewed-captures
