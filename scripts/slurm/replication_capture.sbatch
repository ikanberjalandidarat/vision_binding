#!/usr/bin/env bash
#SBATCH --job-name=replication-capture
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
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
n="${1:-1}"
case "$n" in 1|24) ;; *) echo 'Use 1 for render smoke, then 24 after review' >&2; exit 2;; esac
mkdir -p "$MINESTUDIO_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
xvfb-run -a "$MC_RENDER_ENV/bin/python" -m mc_binding capture-vision \
 --scene-set replication_v1 --n "$n" \
 --output "$MC_BINDING_SCRATCH/runs/vision-captures-replication-${n}-${SLURM_JOB_ID}"
