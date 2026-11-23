#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"

if (( $# > 1 )); then
    printf '用法：bash %s [E2 结果目录]\n' "$0" >&2
    exit 2
fi
results=${1:-E2/results/reproduce}

exec uv run --locked python -m E2.single_gpu.compare \
    --results-dir "$results/single_gpu" --dual-results-dir "$results" --output-dir "$results"
