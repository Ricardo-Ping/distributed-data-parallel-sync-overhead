#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES=0

exec uv run --locked python -u -m E2.single_gpu.run \
    --output-dir E2/results/reproduce/single_gpu "$@"
