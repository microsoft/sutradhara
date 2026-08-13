#!/usr/bin/env bash
# Usage: bash serving_layer/setup.sh   (or: pixi run install-serving-layer)
#
# Clones vLLM v0.11.0, applies sutradhara.patch, and installs it editable over
# the official precompiled wheel. Requires a clean tree: no serving_layer/vllm.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # serving_layer/
SRC="$DIR/vllm"
VLLM_VERSION="0.11.0"
WHEELDIR="$DIR/.wheels"

git clone --depth 1 --branch "v$VLLM_VERSION" https://github.com/vllm-project/vllm.git "$SRC"
cd "$SRC"
git apply "$DIR/sutradhara.patch"

# Resolve the wheel by version, not by a hardcoded hash URL.
mkdir -p "$WHEELDIR"
pip download "vllm==$VLLM_VERSION" --no-deps -d "$WHEELDIR"
WHEEL="$(echo "$WHEELDIR"/vllm-"$VLLM_VERSION"-*.whl)"

# Pin the env's torch; build deps must be present with isolation off.
python use_existing_torch.py
uv pip install -r requirements/build.txt

VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
  uv pip install --no-build-isolation -e .

# Fail loudly if torch or the kernels did not survive the install.
python -c "import torch, vllm; assert torch.version.cuda; print(f'ok: torch {torch.__version__} cuda {torch.version.cuda}, vllm {vllm.__version__}')"
