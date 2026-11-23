#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if (( $# > 1 )); then
    printf '用法：bash %s [第20轮检查点路径]\n' "$0" >&2
    exit 2
fi
checkpoint=${1:-E1/results/reproduce/mnist/M/ws2_seed1234/checkpoint.pt}

printf 'E1 末轮模型的独立测试\n检查点：%s\n' "$checkpoint"
exec uv run --locked python -m E1.evaluate \
    --checkpoint "$checkpoint" --split test --device cuda:0
