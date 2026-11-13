#!/usr/bin/env bash
# Separate replacement renderer; leaves all existing environments/runs intact.
set -euo pipefail
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Run on Oscar Linux x86_64'; exit 2; }
scratch="/oscar/scratch/$USER/zef-binding"
target="$scratch/envs/render-revisit-v1"
export MAMBA_ROOT_PREFIX="$scratch/tools/revisit-mamba"
export PIP_CACHE_DIR="$scratch/cache/revisit-pip"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
mkdir -p "$scratch/tools/revisit-bootstrap" "$PIP_CACHE_DIR"
bootstrap="$scratch/tools/revisit-bootstrap"
if [[ ! -x "$bootstrap/bin/micromamba" ]]; then
 curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$bootstrap/micromamba.tar.bz2"
 tar -xjf "$bootstrap/micromamba.tar.bz2" -C "$bootstrap" bin/micromamba
fi
if [[ ! -d "$target/conda-meta" ]]; then
 "$bootstrap/bin/micromamba" create -y --override-channels -c conda-forge -p "$target" \
  'python=3.10.22' 'openjdk=8.0.472' pip
fi
export JAVA_HOME="$target"
export PATH="$target/bin:$PATH"
export MINESTUDIO_DIR="$scratch/minestudio"
py="$target/bin/python"
"$py" -I -c 'import encodings, ssl, bz2; print(ssl.OPENSSL_VERSION)'
"$target/bin/java" -version
# Rendering uses CPU torch; GPU reservation is for the Minecraft graphics stack.
"$py" -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cpu
"$py" -m pip install 'minestudio==1.1.6' 'numpy==1.26.4' 'albumentations==1.3.1' \
 'torch==2.6.0' 'torchvision==0.21.0' 'setuptools<81' psutil 'Pillow>=10' 'jsonschema>=4.18'
"$py" -m pip check
"$py" -I -c 'import encodings, ssl, bz2, numpy, psutil; from minestudio.simulator import MinecraftSim; print("Renderer imports OK")'
# Install the engine on the login node, before any non-interactive SLURM run.
export HF_HOME="$scratch/cache/huggingface"
"$py" -I - <<'PYENGINE'
from pathlib import Path
from zipfile import ZipFile
from minestudio.simulator.entry import check_engine
from minestudio.utils import get_mine_studio_dir
check_engine(skip_confirmation=True)
jar = Path(get_mine_studio_dir()) / 'engine/build/libs/mcprec-6.13.jar'
with ZipFile(jar) as archive:
    if archive.testzip() is not None:
        raise RuntimeError(f'Corrupt simulator JAR: {jar}')
print(f'Engine installed: {jar}')
PYENGINE
command -v xvfb-run
command -v xauth
"$py" -m pip freeze > "$target/revisit-requirements.txt"
"$bootstrap/bin/micromamba" list -p "$target" --explicit > "$target/revisit-conda-explicit.txt"
printf '\nRenderer setup passed. Actual Minecraft rendering is still unverified.\nexport MC_RENDER_PYTHON="%s/bin/python"\n' "$target"
