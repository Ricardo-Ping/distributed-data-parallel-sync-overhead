#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES=0

if (( $# > 1 )); then
    printf '用法：bash %s [第20轮检查点路径]\n' "$0" >&2
    exit 2
fi
checkpoint=${1:-E0/results/reproduce/mnist/checkpoints/e0_baseline_k1_ws1_mM_b256_r0_epoch20.pt}

printf 'E0 单卡训练模型的独立测试\n检查点：%s\n' "$checkpoint"
exec uv run --locked python -m E0.evaluate \
    --checkpoint "$checkpoint" --split test --device cuda:0
