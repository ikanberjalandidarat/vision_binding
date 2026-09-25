#!/usr/bin/env bash
#SBATCH --job-name=mc-render-smoke
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${MC_RENDER_ENV:?Set MC_RENDER_ENV to the simulator environment directory}"
: "${MC_BINDING_SCRATCH:?Set MC_BINDING_SCRATCH}"
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$MC_RENDER_ENV/bin:$PATH"
export JAVA_HOME="$MC_RENDER_ENV"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export PYTHONUNBUFFERED=1
command -v xvfb-run >/dev/null || { echo 'xvfb-run unavailable: load a site module or use a CCV-supported container'; exit 1; }
xvfb-run -a "$MC_RENDER_ENV/bin/python" -m mc_binding smoke --backend minestudio --output "$MC_BINDING_SCRATCH/render-smoke-${SLURM_JOB_ID}"
