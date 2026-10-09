#!/usr/bin/env bash
#SBATCH --job-name=live-visual-control
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/live-visual-control-%j.out
set -euo pipefail
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD
mode="${1:-smoke}"
args=()
if [[ "$mode" == smoke ]]; then args+=(--smoke); elif [[ "$mode" != full ]]; then echo 'Use smoke or full'; exit 2; fi
"$MC_BINDING_ENV/bin/python" -I -c 'import ssl,bz2,torch,torchvision; assert torch.cuda.is_available()'
"$MC_BINDING_SCRATCH/envs/render/bin/python" -I -c 'import ssl,bz2; from minestudio.simulator import MinecraftSim'
"$MC_BINDING_ENV/bin/python" -m mc_binding.live_visual_control \
 --demos "$MC_BINDING_SCRATCH/runs/vla-demos-7045999" \
 --dataset "$MC_BINDING_SCRATCH/runs/binding-captures-24-6961501" \
 --checkpoint "$MC_BINDING_SCRATCH/runs/perception-study-perception-v1/8x16/readout.pt" \
 --policy-checkpoint "$MC_BINDING_SCRATCH/runs/rl-approach-approach-v1/seed-732-chunk-4/agent.pt" \
 --render-python "$MC_BINDING_SCRATCH/envs/render/bin/python" \
 --output "$MC_BINDING_SCRATCH/runs/live-visual-control-$SLURM_JOB_ID" "${args[@]}"
