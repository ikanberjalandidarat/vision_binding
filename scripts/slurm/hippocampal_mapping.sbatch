#!/usr/bin/env bash
#SBATCH --job-name=hippocampal-map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hippocampal-map-%j.out
set -euo pipefail
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD
case "${1:-smoke}" in
 smoke) scenes=3; steps=120; epochs=10 ;;
 pilot) scenes=6; steps=1200; epochs=100 ;;
 *) echo 'Use smoke or pilot'; exit 2 ;;
esac
"$MC_BINDING_ENV/bin/python" -I -c 'import ssl,bz2,torch,torchvision; assert torch.cuda.is_available()'
"$MC_BINDING_SCRATCH/envs/render/bin/python" -I -c 'import ssl,bz2; from minestudio.simulator import MinecraftSim'
"$MC_BINDING_ENV/bin/python" -m mc_binding.hippocampal_study \
 --dataset "$MC_BINDING_SCRATCH/runs/binding-captures-24-6961501" \
 --render-python "$MC_BINDING_SCRATCH/envs/render/bin/python" \
 --output "$MC_BINDING_SCRATCH/runs/hippocampal-mapping-${SLURM_JOB_ID:?}" \
 --scenes "$scenes" --steps "$steps" --epochs "$epochs" --stride 4 --seed "${2:-731}"
