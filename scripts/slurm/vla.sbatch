#!/usr/bin/env bash
#SBATCH --job-name=minecraft-vla
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
mode="${1:?Choose collect, features, train, or rollout}"
input="${2:?Supply input run directory name}"
[[ "$input" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export MC_RENDER_ENV="${MC_RENDER_ENV:-$MC_BINDING_SCRATCH/envs/render}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
runs="$MC_BINDING_SCRATCH/runs"
config="${VLA_CONFIG:-configs/vla_pilot.json}"
case "$mode" in
 collect)
 n="${3:-1}"
 [[ "${4:-}" == --reviewed-captures ]] || { echo 'Pass family count and --reviewed-captures' >&2; exit 2; }
 "$MC_BINDING_ENV/bin/python" -m mc_binding vla-collect --dataset "$runs/$input" --n "$n" \
 --render-python "$MC_RENDER_ENV/bin/python" --output "$runs/vla-demos-$SLURM_JOB_ID" --config "$config" --reviewed-captures;;
 features)
 "$MC_BINDING_ENV/bin/python" -m mc_binding vla-features --demos "$runs/$input" --output "$runs/vla-features-$SLURM_JOB_ID" --config "$config";;
 train)
 "$MC_BINDING_ENV/bin/python" -m mc_binding vla-train --cache "$runs/$input" --output "$runs/vla-policy-$SLURM_JOB_ID" --config "$config";;
 rollout)
 policy="${3:?Supply policy directory name}"
 [[ "$policy" =~ ^vla-policy-[A-Za-z0-9_-]+$ ]] || exit 2
 [[ "${4:-}" == --reviewed-captures ]] || exit 2
 "$MC_BINDING_ENV/bin/python" -m mc_binding vla-rollout --dataset "$runs/$input" --policy "$runs/$policy" \
 --render-python "$MC_RENDER_ENV/bin/python" --output "$runs/vla-rollout-$SLURM_JOB_ID" --config "$config" --reviewed-captures;;
 *) exit 2;;
esac
