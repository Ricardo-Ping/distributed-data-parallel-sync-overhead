#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$PATH:$HOME/.local/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec uv run --locked python -u - "$@" <<'PY'
import argparse

from E1.config import DATASETS, E1_ROOT, PRESETS, SEEDS, load_config
from E1.distributed import cuda_device
from E1.model import MLP
from E1.run import run

parser = argparse.ArgumentParser(description="列出 E1 的单双卡计划并在 CUDA 上核对模型规模")
parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
parser.add_argument("--seed", type=int, choices=SEEDS, default=1234)
args = parser.parse_args()

config = load_config(E1_ROOT / "configs" / f"{args.dataset}.yaml")
run([config], seed=args.seed, list_only=True)

device = cuda_device("cuda:0")
in_dim = 3072 if args.dataset == "cifar10" else 784
print(f"\n模型规模  dataset={args.dataset}  input={in_dim}  device={device}")
for preset in PRESETS:
    size = MLP(in_dim, preset).to(device).size_info()
    print(
        f"{preset}: hidden={size['hidden1']}→{size['hidden2']}, "
        f"parameters={size['parameters']:,}, gradient={size['gradient_bytes']:,} bytes"
    )
PY
