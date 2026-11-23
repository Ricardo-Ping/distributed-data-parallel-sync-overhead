#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if (( $# > 1 )); then
    printf '用法：bash %s [E2 结果目录]\n' "$0" >&2
    exit 2
fi
results=${1:-E2/results/reproduce}

uv run --locked python -u -m E2.check_results --results-dir "$results"
exec uv run --locked python -u -m E2.single_gpu.check_results \
    --results-dir "$results/single_gpu" --dual-results-dir "$results"
