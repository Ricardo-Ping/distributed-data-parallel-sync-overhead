#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' '找不到 nvidia-smi，请先安装 NVIDIA 驱动。' >&2
    exit 1
fi

printf '用户：%s\n主机：%s\n项目目录：%s\n' "$(whoami)" "$(hostname)" "$PWD"
nvidia-smi
nvidia-smi topo -m

uv_bin_dir="$HOME/.local/bin"
export PATH="$PATH:$uv_bin_dir"
if ! command -v uv >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh |
        env UV_INSTALL_DIR="$uv_bin_dir" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

if ! uv python find --no-python-downloads 3.12 >/dev/null 2>&1; then
    uv python install 3.12
fi
uv sync --locked

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
uv run --locked --no-sync python - <<'PY'
import sys

from matplotlib import font_manager
import torch
import torchvision

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA 运行时:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("PyTorch 无法使用 CUDA，请检查 NVIDIA 驱动及 GPU 配置。")
gpu_count = torch.cuda.device_count()
print("PyTorch 可见 GPU 数量:", gpu_count)
if gpu_count < 2:
    raise SystemExit("完整实验需要至少两张可见的 GPU，请检查 CUDA_VISIBLE_DEVICES。")
if not torch.distributed.is_nccl_available():
    raise SystemExit("当前 PyTorch 缺少 NCCL 后端。")

print("NCCL:", ".".join(map(str, torch.cuda.nccl.version())))
for index in (0, 1):
    print(f"cuda:{index}:", torch.cuda.get_device_name(index))
    print(f"P2P {index} -> {1 - index}:", torch.cuda.can_device_access_peer(index, 1 - index))

for font in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError:
        continue
    print("中文绘图字体:", font)
    break
else:
    raise SystemExit("缺少中文绘图字体。Ubuntu 可用 sudo apt-get install -y fonts-noto-cjk 安装。")

print("环境与依赖检查通过。")
PY
