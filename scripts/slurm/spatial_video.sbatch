#!/usr/bin/env bash
#SBATCH --job-name=spatial-video
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/spatial-video-%j.out
set -euo pipefail
# Use a SEPARATE prepared environment. This job never installs/repairs packages.
: "${SPATIAL_PYTHON:?Set absolute Python path in a separate spatial-video environment}"
: "${MC_BINDING_SCRATCH:=/oscar/scratch/$USER/zef-binding}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
unset LD_LIBRARY_PATH LD_PRELOAD
mode="${1:?Use collect, extract, or probe}"
case "$mode" in
 collect)
  : "${MC_RENDER_PYTHON:?Set existing healthy render Python; no modifications made}"
  "$SPATIAL_PYTHON" -m mc_binding.spatial_video collect \
   --dataset "$MC_BINDING_SCRATCH/runs/binding-captures-24-6961501" \
   --render-python "$MC_RENDER_PYTHON" \
   --output "$MC_BINDING_SCRATCH/runs/spatial-video-data-${SLURM_JOB_ID}"
  ;;
 extract|smoke)
  : "${VJEPA_REPO:?Set clean official local repository path}"
  : "${VJEPA_CHECKPOINT:?Set downloaded official vitl.pt path}"
  extra=()
  if [[ "$mode" == smoke ]]; then extra+=(--smoke); fi
  "$SPATIAL_PYTHON" -m mc_binding.spatial_video extract \
   --dataset "${2:?Pass collection directory}" --repo "$VJEPA_REPO" \
   --checkpoint "$VJEPA_CHECKPOINT" \
   --output "$MC_BINDING_SCRATCH/runs/spatial-video-features-${SLURM_JOB_ID}" "${extra[@]}"
  ;;
 probe)
  "$SPATIAL_PYTHON" -m mc_binding.spatial_video_probe "${2:?Pass extraction directory}"
  ;;
 *) echo "Use collect, smoke, extract, or probe"; exit 2 ;;
esac
