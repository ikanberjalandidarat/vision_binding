#!/usr/bin/env bash
#SBATCH --job-name=minecraft-rl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -euo pipefail
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export PYTHONUNBUFFERED=1
suite="${1:?Suite path required}"
mode="${2:?smoke or sweep}"
if [[ "$mode" == smoke ]]; then
 seed=731; episodes=4; steps=8; evaluation=2; name=smoke; start=near-to-far
else
 index="${SLURM_ARRAY_TASK_ID:?}"; seed=$((731+index/2)); arm=$((index%2))
 start=original; if ((arm==1)); then start=near-to-far; fi
 episodes=48; steps=96; evaluation=16; name="seed-${seed}-${start}"
fi
unset LD_LIBRARY_PATH LD_PRELOAD

"$MC_BINDING_ENV/bin/python" -I -c 'import ssl,bz2,torch,torchvision; assert torch.cuda.is_available(); print(ssl.OPENSSL_VERSION)'
"$MC_BINDING_SCRATCH/envs/render/bin/python" -I -c 'import ssl,bz2; print(ssl.OPENSSL_VERSION); from minestudio.simulator import MinecraftSim'
"$MC_BINDING_SCRATCH/envs/render/bin/java" -version
"$MC_BINDING_ENV/bin/python" -m mc_binding.rl_navigation \
 --dataset "$MC_BINDING_SCRATCH/runs/binding-captures-24-6961501" \
 --output "$suite/$name" --render-python "$MC_BINDING_SCRATCH/envs/render/bin/python" \
 --seed "$seed" --reward potential --aux-weight 0 --start-mode "$start" --paired-evaluation \
 --episodes "$episodes" --max-steps "$steps" --eval-episodes "$evaluation"
