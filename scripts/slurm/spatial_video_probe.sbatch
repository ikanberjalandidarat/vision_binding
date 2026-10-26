#!/usr/bin/env bash
#SBATCH --job-name=spatial-probe
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/spatial-probe-%j.out
set -euo pipefail
extraction="${1:?Pass completed spatial-video-features directory}"
scratch_root="/oscar/scratch/$USER/zef-binding"
python="${SPATIAL_PYTHON:-$scratch_root/envs/spatial-video/bin/python}"
[[ -x "$python" ]] || { echo "Spatial-video Python missing: $python"; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD

# Match the extraction job's job-local workaround if this environment's bz2
# library link is dangling. No shared environment files are changed.
spatial_lib="$(dirname "$(dirname "$python")")/lib"
if [[ ! -e "$spatial_lib/libbz2.so.1.0" ]]; then
  system_bz2="/lib64/libbz2.so.1.0.8"
  [[ -r "$system_bz2" ]] || { echo "Missing libbz2 in environment and system"; exit 1; }
  compat_lib="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/spatial-bz2-${SLURM_JOB_ID}-XXXXXX")"
  ln -s "$system_bz2" "$compat_lib/libbz2.so.1.0"
  export LD_LIBRARY_PATH="$compat_lib"
fi
"$python" -I -c 'import bz2, numpy; assert bz2.decompress(bz2.compress(b"ready")) == b"ready"'
"$python" -m mc_binding.spatial_video_probe "$extraction"
