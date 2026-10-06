#!/usr/bin/env bash
set -euo pipefail
plan="${1:?Plan JSON required}"
key="${2:?Stage key required}"
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export MC_RENDER_ENV="${MC_RENDER_ENV:-$MC_BINDING_SCRATCH/envs/render}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
"$MC_BINDING_ENV/bin/python" scripts/navigation_study.py worker --plan "$plan" --key "$key"
