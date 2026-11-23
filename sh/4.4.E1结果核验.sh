#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec uv run --locked python -u -m E1.check_results \
    --results-dir E1/results/reproduce "$@"
