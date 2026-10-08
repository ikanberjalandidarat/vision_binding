#!/usr/bin/env bash
#SBATCH --job-name=target-tracking
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/target-tracking-%j.out
set -euo pipefail
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONUNBUFFERED=1
output="$MC_BINDING_SCRATCH/runs/target-tracking-${SLURM_JOB_ID:?}"
checkpoint="$MC_BINDING_SCRATCH/runs/perception-study-perception-v1/8x16/readout.pt"
"$MC_BINDING_ENV/bin/python" -m mc_binding.target_tracking prepare \
 --rollout "$MC_BINDING_SCRATCH/runs/vla-demos-7045999" \
 --checkpoint "$checkpoint" --output "$output/audit" --episodes "${1:-16}" --interval 8
"$MC_BINDING_ENV/bin/python" -m mc_binding.target_tracking evaluate \
 --audit "$output/audit" --checkpoint "$checkpoint" --output "$output/evaluation"
