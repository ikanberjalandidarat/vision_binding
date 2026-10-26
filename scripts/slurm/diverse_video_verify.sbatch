#!/usr/bin/env bash
#SBATCH --job-name=diverse-verify
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/diverse-verify-%j.out
set -euo pipefail
dataset="${1:?Pass absolute diverse-dataset-v2 root}"
scratch_root="/oscar/scratch/$USER/zef-binding"
python="${MC_RENDER_PYTHON:-$scratch_root/envs/render-spatial/bin/python}"
[[ -x "$python" ]] || { echo "Render Python missing: $python"; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD
"$python" -m mc_binding.spatial_video_dataset "$dataset" \
  "$scratch_root/runs/diverse-video-verify-${SLURM_JOB_ID}"
