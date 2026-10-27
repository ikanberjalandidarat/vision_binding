#!/usr/bin/env bash
# Prepare ONLY the official predictive-coding environment case on Oscar.
# Never modifies existing Minecraft runs, render/model/spatial-video envs, or
# the author's notebooks. Run from the zef-binding repository root.
set -euo pipefail

[[ -f scripts/run_predictive_map_reference.py ]] || { echo 'Run from the zef-binding repository root'; exit 2; }
scratch_root="/oscar/scratch/$USER/zef-binding"
reference_repo="$scratch_root/vendor/predictive-coding-recovers-maps"
reference_weights="$scratch_root/reference-weights"
reference_env="$scratch_root/envs/predictive-map"
expected_commit="fed80b72925a4d3f3e9ec4b70f71ecaedec75c13"
origin="https://github.com/jgornet/predictive-coding-recovers-maps.git"
weight_origin="https://s3.us-west-1.wasabisys.com/predictive-coding-recovers-maps/weights"

mkdir -p "$scratch_root/vendor" "$reference_weights" "$scratch_root/envs"
if [[ ! -d "$reference_repo" ]]; then
  git clone --filter=blob:none --sparse "$origin" "$reference_repo"
  git -C "$reference_repo" sparse-checkout set predictive_coding
fi
[[ "$(git -C "$reference_repo" rev-parse HEAD)" == "$expected_commit" ]] || \
  git -C "$reference_repo" checkout "$expected_commit"
[[ -z "$(git -C "$reference_repo" status --porcelain)" ]] || { echo 'Official checkout has local changes'; exit 1; }
[[ -f "$reference_repo/predictive_coding/models/encoder_decoder.py" ]] || { echo 'Official model code missing'; exit 1; }

for name in predictive_coding.ckpt predictive-coder-environment-images.npy predictive-coder-environment-positions.npy; do
  target="$reference_weights/$name"
  if [[ -s "$target" ]]; then
    echo "Already present: $target"
    continue
  fi
  curl --fail --location --retry 3 --continue-at - \
    --output "$target.part" "$weight_origin/$name"
  [[ -s "$target.part" ]] || { echo "Empty download: $name"; exit 1; }
  mv "$target.part" "$target"
done

if [[ ! -f "$reference_env/.reference-dependencies-ready" ]]; then
  # A standalone venv on scratch. Do not install into any existing environment.
  if [[ ! -x "$reference_env/bin/python" ]]; then
    "${BOOTSTRAP_PYTHON:-python3}" -m venv "$reference_env"
  fi
  "$reference_env/bin/python" -m pip install --upgrade pip
  "$reference_env/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    'torch==2.6.0' 'torchvision==0.21.0' 'xformers==0.0.29.post3'
  "$reference_env/bin/python" -m pip install \
    'numpy<2' -r "$reference_repo/requirements.txt"
  "$reference_env/bin/python" -m pip check
  touch "$reference_env/.reference-dependencies-ready"
fi

"$reference_env/bin/python" -m pip check
PYTHONPATH="$PWD/src" "$reference_env/bin/python" scripts/run_predictive_map_reference.py \
  --repo "$reference_repo" --weights "$reference_weights" \
  --case environment --check-only
echo 'Reference environment case assets passed preflight. Submit the GPU job separately.'
