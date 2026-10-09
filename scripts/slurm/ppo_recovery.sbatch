#!/usr/bin/env bash
#SBATCH --job-name=ppo-recovery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/ppo-recovery-%A_%a.out
set -euo pipefail
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD
stage="${1:-smoke}"
index="${SLURM_ARRAY_TASK_ID:?}"
(( index >= 0 && index <= 3 )) || exit 2
seed=$((731+index/2)); arm=terminal; extra=()
if (( index%2 )); then arm=recovery; extra=(--recover-stop); fi
case "$stage" in
 smoke) budget=16; cap=8; steps=8; evaluation=4; interval=0 ;;
 full) budget=1500; cap=1024; steps=32; evaluation=16; interval=500 ;;
 *) echo 'Use smoke or full'; exit 2 ;;
esac
"$MC_BINDING_ENV/bin/python" -I -c 'import ssl,bz2,torch,torchvision; assert torch.cuda.is_available()'
"$MC_BINDING_SCRATCH/envs/render/bin/python" -I -c 'import ssl,bz2; from minestudio.simulator import MinecraftSim'
out="$MC_BINDING_SCRATCH/runs/ppo-recovery-${SLURM_ARRAY_JOB_ID}/seed-$seed-$arm"
"$MC_BINDING_ENV/bin/python" -m mc_binding.ppo_navigation \
 --dataset "$MC_BINDING_SCRATCH/runs/binding-captures-24-6961501" \
 --checkpoint "$MC_BINDING_SCRATCH/runs/perception-study-perception-v1/8x16/readout.pt" \
 --render-python "$MC_BINDING_SCRATCH/envs/render/bin/python" --output "$out" \
 --method ppo --mode binding --seed "$seed" --easy-start \
 --transition-budget "$budget" --episodes "$cap" --max-steps "$steps" \
 --batch-episodes 4 --epochs 4 --eval-episodes "$evaluation" --probe-interval "$interval" "${extra[@]}"
