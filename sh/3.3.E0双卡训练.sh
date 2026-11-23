#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES=0,1

printf 'E0 双卡训练  CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
exec uv run --locked python -u -m E0.run \
    --output-dir E0/results/reproduce "$@" --world-size 2
