#!/usr/bin/env bash
#SBATCH --job-name=destination-replay
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
# Same recorded image, same requested color, same controller; only saved choice differs.
# Both goals included to avoid showing only one direction of the effect.
for goal in yellow blue; do
  for condition in clean self patched; do
    case "$condition" in
      clean) trial="f0000:baseline:clean_recipient:${goal}" ;;
      self) trial="f0000:group30-31:self:${goal}" ;;
      patched) trial="f0000:group30-31:both_objects:${goal}" ;;
    esac
    xvfb-run -a "$MC_RENDER_ENV/bin/python" -m mc_binding approach \
      --dataset "$MC_BINDING_SCRATCH/runs/vision-captures-replication-24-6852730" \
      --decisions "$MC_BINDING_SCRATCH/runs/destination-replication-6868954" \
      --trial "$trial" \
      --output "$MC_BINDING_SCRATCH/runs/destination-replay-${SLURM_JOB_ID}/${goal}-${condition}"
  done
done
