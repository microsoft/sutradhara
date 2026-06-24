#!/usr/bin/env bash
# Sutradhara serving layer: vLLM v0.11.0 (precompiled, torch 2.8.0) + sutradhara.patch, editable.
# Assumes a clean tree (no pre-existing serving_layer/vllm).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # serving_layer/
SRC="$DIR/vllm"
VLLM_VERSION="0.11.0"
WHEELDIR="$DIR/.wheels"

# Clone the pinned tag and apply the patch on the clean tree, before install.
git clone --depth 1 --branch "v$VLLM_VERSION" https://github.com/vllm-project/vllm.git "$SRC"
cd "$SRC"
git apply "$DIR/sutradhara.patch"

# Resolve the official 0.11.0 wheel by VERSION (no hardcoded hash URL).
# This is the CUDA 12.8 / torch 2.8.0 build — the only x86 CUDA wheel 0.11.0 ships.
mkdir -p "$WHEELDIR"
pip download "vllm==$VLLM_VERSION" --no-deps -d "$WHEELDIR"
WHEEL="$(echo "$WHEELDIR"/vllm-"$VLLM_VERSION"-*.whl)"

# Keep the env's torch 2.8.0 out of vLLM's resolution; isolation off needs build deps present.
python use_existing_torch.py
uv pip install -r requirements/build.txt

# Editable install reusing the official precompiled kernels — seconds, no nvcc.
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
  uv pip install --no-build-isolation -e .

# Cheap insurance given the cu128-built / cu129-runtime mix: fail loud if torch or kernels are off.
python -c "import torch, vllm; assert torch.version.cuda; print(f'ok: torch {torch.__version__} cuda {torch.version.cuda}, vllm {vllm.__version__}')"
