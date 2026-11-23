#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export MPLBACKEND=Agg

if (( $# > 1 )); then
    printf '用法：bash %s [已生成配对统计的 E2 结果目录]\n' "$0" >&2
    exit 2
fi
results=${1:-E2/results/reproduce}

uv run --locked python -m E2.plot_results --results-dir "$results"
exec uv run --locked python -m E2.single_gpu.plot_results --results-dir "$results"
