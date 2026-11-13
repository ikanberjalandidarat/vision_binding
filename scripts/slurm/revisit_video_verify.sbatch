#!/usr/bin/env bash
#SBATCH --job-name=revisit-verify
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/revisit-verify-%j.out
set -euo pipefail
scratch="/oscar/scratch/$USER/zef-binding"
python="${MC_RENDER_PYTHON:-$scratch/envs/render-revisit-v1/bin/python}"
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD
"$python" -m mc_binding.revisit_video "${1:?Pass revisit dataset}" "$scratch/runs/revisit-video-verify-$SLURM_JOB_ID"
