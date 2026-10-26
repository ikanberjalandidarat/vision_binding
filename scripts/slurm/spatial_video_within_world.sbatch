#!/usr/bin/env bash
#SBATCH --job-name=spatial-within
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/spatial-within-%j.out
set -euo pipefail
extraction="${1:?Pass completed spatial-video-features directory}"
dataset="${2:?Pass diverse-dataset-v2 root with metadata}"
scratch_root="/oscar/scratch/$USER/zef-binding"
python="${SPATIAL_PYTHON:-$scratch_root/envs/spatial-video/bin/python}"
[[ -x "$python" ]] || { echo "Spatial-video Python missing: $python"; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD

# Match the successful extraction job's job-local bz2 workaround.
spatial_lib="$(dirname "$(dirname "$python")")/lib"
if [[ ! -e "$spatial_lib/libbz2.so.1.0" ]]; then
  system_bz2="/lib64/libbz2.so.1.0.8"
  [[ -r "$system_bz2" ]] || { echo "Missing libbz2 in environment and system"; exit 1; }
  compat_lib="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/spatial-bz2-${SLURM_JOB_ID}-XXXXXX")"
  ln -s "$system_bz2" "$compat_lib/libbz2.so.1.0"
  export LD_LIBRARY_PATH="$compat_lib"
fi
"$python" -I -c 'import bz2, numpy; assert bz2.decompress(bz2.compress(b"ready")) == b"ready"'
"$python" -m mc_binding.spatial_video_within_world "$extraction" "$dataset" \
  "$scratch_root/runs/spatial-video-withinworld-${SLURM_JOB_ID}"
