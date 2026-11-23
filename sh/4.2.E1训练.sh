#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

printf 'E1 模型规模训练  CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
exec uv run --locked python -u -m E1.run \
    --output-dir E1/results/reproduce "$@"
