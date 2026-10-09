#!/usr/bin/env bash
#SBATCH --job-name=binding-rl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/binding-rl-%A_%a.out
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
seed=$((731+index/3)); modes=(zero binding shuffled); mode="${modes[$((index%3))]}"
if [[ "$stage" == smoke ]]; then
 episodes=2; steps=8; evaluation=2
elif [[ "$stage" == full ]]; then
 episodes=96; steps=96; evaluation=64
else
 echo 'Use smoke or full'; exit 2
fi
"$MC_BINDING_ENV/bin/python" -I -c 'import ssl,bz2,torch,torchvision; assert torch.cuda.is_available()'
"$MC_BINDING_SCRATCH/envs/render/bin/python" -I -c 'import ssl,bz2; from minestudio.simulator import MinecraftSim'
"$MC_BINDING_ENV/bin/python" -m mc_binding.rl_navigation \
 --dataset "$MC_BINDING_SCRATCH/runs/binding-captures-24-6961501" \
 --output "$MC_BINDING_SCRATCH/runs/binding-rl-${SLURM_ARRAY_JOB_ID}/seed-$seed-$mode" \
 --render-python "$MC_BINDING_SCRATCH/envs/render/bin/python" \
 --binding-checkpoint "$MC_BINDING_SCRATCH/runs/perception-study-perception-v1/8x16/readout.pt" \
 --binding-mode "$mode" --seed "$seed" --reward potential --aux-weight 0 \
 --episodes "$episodes" --max-steps "$steps" --eval-episodes "$evaluation" \
 --action-chunk 2 --start-mode near-to-far --paired-evaluation
