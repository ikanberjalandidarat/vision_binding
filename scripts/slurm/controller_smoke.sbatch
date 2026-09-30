#!/usr/bin/env bash
#SBATCH --job-name=controller-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_RENDER_ENV="${MC_RENDER_ENV:-$MC_BINDING_SCRATCH/envs/render}"
export JAVA_HOME="$MC_RENDER_ENV"
export PATH="$MC_RENDER_ENV/bin:$PATH"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
goal="${1:-yellow}"
case "$goal" in yellow|blue) ;; *) echo 'Use yellow or blue for f0000' >&2; exit 2;; esac
xvfb-run -a "$MC_RENDER_ENV/bin/python" -m mc_binding controller-smoke \
 --dataset "$MC_BINDING_SCRATCH/runs/vision-captures-replication-24-6852730" \
 --family f0000 --goal-color "$goal" \
 --output "$MC_BINDING_SCRATCH/runs/controller-smoke-${goal}-${SLURM_JOB_ID}"
