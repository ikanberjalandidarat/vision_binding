#!/usr/bin/env bash
#SBATCH --job-name=visual-readout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
mode="${1:?Choose extract, train, evaluate, or patch}"
dataset="${2:?Supply binding-captures directory name}"
[[ "$dataset" =~ ^binding-captures-[A-Za-z0-9_-]+$ ]] || exit 2
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
config="${VISUAL_READOUT_CONFIG:-configs/visual_readout.json}"
runs="$MC_BINDING_SCRATCH/runs"
cache="$runs/visual-features-$dataset"
readout="$runs/visual-trained-$dataset"
case "$mode" in
 extract)
 [[ "${3:-}" == --reviewed-captures ]] || { echo 'Review captures first' >&2; exit 2; }
 "$MC_BINDING_ENV/bin/python" -m mc_binding visual-extract --dataset "$runs/$dataset" --output "$cache" --config "$config" --reviewed-captures;;
 train)
 "$MC_BINDING_ENV/bin/python" -m mc_binding visual-train --cache "$cache" --output "$readout" --config "$config";;
 evaluate|patch)
 [[ "${3:-}" == --reviewed-captures ]] || { echo 'Review captures first' >&2; exit 2; }
 extra=()
 if [[ "$mode" == patch ]]; then extra+=(--patch); fi
 "$MC_BINDING_ENV/bin/python" -m mc_binding visual-evaluate --cache "$cache" --dataset "$runs/$dataset" --readout "$readout" \
 --output "$runs/visual-${mode}-${SLURM_JOB_ID}" --config "$config" --reviewed-captures "${extra[@]}";;
 *) exit 2;;
esac
