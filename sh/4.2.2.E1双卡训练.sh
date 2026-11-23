#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0,1

exec bash "$script_dir/4.2.E1训练.sh" "$@" --world-size 2
