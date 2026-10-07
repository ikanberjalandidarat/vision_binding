#!/usr/bin/env bash
#SBATCH --job-name=live-perception
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/live-perception-%j.out
set -euo pipefail
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
"$MC_BINDING_ENV/bin/python" -m mc_binding.live_perception evaluate \
 --audit "${1:?Reviewed audit folder required}" \
 --checkpoint "$MC_BINDING_SCRATCH/runs/perception-study-perception-v1/8x16/readout.pt" \
 --output "$MC_BINDING_SCRATCH/runs/live-perception-$SLURM_JOB_ID"
