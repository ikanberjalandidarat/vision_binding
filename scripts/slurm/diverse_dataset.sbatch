#!/usr/bin/env bash
#SBATCH --job-name=diverse-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/diverse-data-%A_%a.out
set -euo pipefail
root="${1:?Pass dataset output directory}"
mode="${2:-collect}"
scratch_root="/oscar/scratch/$USER/zef-binding"
export MC_RENDER_PYTHON="${MC_RENDER_PYTHON:-$scratch_root/envs/render-spatial/bin/python}"
export JAVA_HOME="$(dirname "$(dirname "$MC_RENDER_PYTHON")")"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/src"
export MINESTUDIO_DIR="$scratch_root/minestudio"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD
[[ -x "$MC_RENDER_PYTHON" ]] || { echo 'Render Python missing'; exit 1; }
if [[ "$mode" == gallery ]]; then
 args=(gallery --root "$root")
 if [[ -n "${3:-}" ]]; then args+=(--indices "$3"); fi
 "$MC_RENDER_PYTHON" -m mc_binding.diverse_dataset "${args[@]}"
else
 "$MC_RENDER_PYTHON" -I -c 'from minestudio.simulator import MinecraftSim'
 xvfb-run -a "$MC_RENDER_PYTHON" -m mc_binding.diverse_dataset collect \
   --root "$root" --index "${SLURM_ARRAY_TASK_ID:?Submit --array=0-7%2}"
fi
