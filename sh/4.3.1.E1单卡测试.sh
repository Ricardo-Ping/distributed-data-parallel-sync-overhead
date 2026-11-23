#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0

if (( $# == 0 )); then
    set -- E1/results/reproduce/mnist/M/ws1_seed1234/checkpoint.pt
fi

printf '%s\n' 'E1 单卡训练模型复测'
exec bash "$script_dir/4.3.E1测试.sh" "$@"
