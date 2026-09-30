#!/usr/bin/env bash
#SBATCH --job-name=destination-diagnostic
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
"$MC_BINDING_ENV/bin/python" scripts/diagnose_destination.py \
 --dataset "$MC_BINDING_SCRATCH/runs/vision-captures-replication-24-6852730" \
 --output "$MC_BINDING_SCRATCH/runs/destination-diagnostic-${SLURM_JOB_ID}"
