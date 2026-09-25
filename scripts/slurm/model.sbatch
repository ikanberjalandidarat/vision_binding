#!/usr/bin/env bash
#SBATCH --job-name=mc-binding
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${MC_BINDING_ENV:?Set MC_BINDING_ENV to the model environment directory}"
: "${MC_BINDING_SCRATCH:?Set MC_BINDING_SCRATCH to your scratch project directory}"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"
"$MC_BINDING_ENV/bin/python" -m mc_binding doctor --output "$MC_BINDING_SCRATCH/environment-${SLURM_JOB_ID}.json"
"$MC_BINDING_ENV/bin/python" -m mc_binding "$@"
