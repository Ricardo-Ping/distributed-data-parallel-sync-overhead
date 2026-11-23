#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' '找不到 uv，请先运行 sh/1.环境与依赖准备.sh。' >&2
    exit 1
fi

printf '用户：%s\n主机：%s\n项目目录：%s\n' "$(whoami)" "$(hostname)" "$PWD"

printf '\n准备并核对三个数据集的官方缓存\n'
uv run --locked python -m Data.datasets --download
