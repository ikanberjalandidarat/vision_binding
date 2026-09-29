#!/usr/bin/env bash
#SBATCH --job-name=replication-model
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
task="${1:?Choose vision-pilot or destination}"
dataset="${2:?Supply capture directory name}"
case "$task" in vision-pilot|destination) ;; *) exit 2;; esac
[[ "$dataset" =~ ^vision-captures-[A-Za-z0-9_-]+$ ]] || exit 2
[[ "${3:-}" == '--reviewed-captures' ]] || { echo 'Review images first and pass --reviewed-captures' >&2; exit 2; }
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
"$MC_BINDING_ENV/bin/python" -m mc_binding "$task" \
 --dataset "$MC_BINDING_SCRATCH/runs/$dataset" \
 --config configs/vision_replication_v.json \
 --output "$MC_BINDING_SCRATCH/runs/${task}-replication-${SLURM_JOB_ID}" \
 --reviewed-captures
