#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export MPLBACKEND=Agg

exec uv run --locked python -m E0.plot_results \
    --results-dir E0/results/reproduce "$@"
