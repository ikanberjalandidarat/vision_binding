#!/usr/bin/env bash
set -euo pipefail
export MC_BINDING_SCRATCH="${MC_BINDING_SCRATCH:-/oscar/scratch/$USER/zef-binding}"
export MC_BINDING_ENV="${MC_BINDING_ENV:-$MC_BINDING_SCRATCH/envs/model}"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
"$MC_BINDING_ENV/bin/python" -I -c 'import ssl, bz2; from transformers import AutoProcessor; print("SSL and model imports OK", ssl.OPENSSL_VERSION)'
"$MC_BINDING_ENV/bin/python" -m mc_binding.perception_study "$@"
