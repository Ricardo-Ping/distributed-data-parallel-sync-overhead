#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if (( $# > 1 )); then
    printf '用法：bash %s [数值诊断的输出目录]\n' "$0" >&2
    exit 2
fi
results=${1:-E3/results/reproduce/numerics}

uv run --locked python -u -m E3.verify_numerics --results-dir "$results"
exec uv run --locked python - "$results/numerical_equivalence.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

full = sum(row["window"] == "full" for row in rows)
tail = sum(row["window"] == "tail" for row in rows)
fp64_passed = sum(row["fp64_gradient_check_passed"].lower() == "true" for row in rows)
devices = ", ".join(sorted({row["device"] for row in rows}))
backends = ", ".join(sorted({row["backend"] for row in rows}))
print(f"设备: {devices}  后端: {backends}")
print(f"完整窗口: {full}  尾窗口: {tail}  FP64 梯度核验通过: {fp64_passed}/{len(rows)}")
print(f"结果: {sys.argv[1]}")
PY
