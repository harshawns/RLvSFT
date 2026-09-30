#!/usr/bin/env bash
# Create an Apple Silicon MLX env named `rl` (or $ENV_DIR) with TRL + mlx-lm.
# Run on macOS with Apple Silicon — do not use this machine's Linux box for this.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3.10}"
ENV_DIR="${ENV_DIR:-rl}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script targets macOS Apple Silicon (MLX). Detected: $(uname -s) $(uname -m)" >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "MLX requires Apple Silicon (arm64). Detected: $(uname -m)" >&2
  exit 1
fi

if [[ ! -d "$ENV_DIR" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$ENV_DIR" --python "$PYTHON"
  else
    "$PYTHON" -m venv "$ENV_DIR"
  fi
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip)
else
  PIP=(python -m pip)
  "${PIP[@]}" install -U pip
fi

echo "Installing MLX + TRL stack into ${ENV_DIR}..."
"${PIP[@]}" install -r requirements-mlx.txt

python - <<'PY'
import mlx.core as mx
import mlx_lm
import trl
import transformers
import peft
import torch

print("mlx default_device:", mx.default_device())
print("torch", torch.__version__, "mps:", torch.backends.mps.is_available())
print("trl", trl.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("mlx_lm ok")
PY

echo
echo "Activate with: source ${ENV_DIR}/bin/activate"
