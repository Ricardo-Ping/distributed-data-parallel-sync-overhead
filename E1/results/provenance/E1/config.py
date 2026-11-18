"""E1 固定训练协议与模型宽度；只共享顶层 Data 的原始数据接口。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

E1_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = E1_ROOT.parent
RESULTS_ROOT = E1_ROOT / "results"
DATASETS = ("mnist", "fashionmnist", "cifar10")
SEEDS = (1234, 1235, 1236)
PRESETS = {"S": (256, 128), "M": (1024, 512), "L": (4096, 2048)}


@dataclass(frozen=True)
class Config:
    dataset: str
    learning_rate: float
    epochs: int = 20
    local_batch: int = 128
    momentum: float = .9
    seeds: tuple[int, ...] = SEEDS
    models: tuple[str, ...] = ("S", "M", "L")
    world_sizes: tuple[int, ...] = (1, 2)
    validation_fraction: float = .1
    split_seed: int = 1234
    eval_batch: int = 1024
    warmup_steps: int = 10
    bucket_cap_mb: int = 128
    host_threads: int = 4
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS or self.learning_rate <= 0:
            raise ValueError("需要三个正式数据集之一及正学习率")
        if (self.epochs, self.local_batch, self.device) != (20, 128, "cuda"):
            raise ValueError("E1 固定20轮、每卡batch=128、CUDA")
        if self.seeds != SEEDS or self.models != tuple(PRESETS) or self.world_sizes != (1, 2):
            raise ValueError("E1 使用 S/M/L、三个种子和单/双GPU")
        if not 0 < self.validation_fraction < 1 or not 0 <= self.momentum < 1:
            raise ValueError("验证比例或动量无效")
        if min(self.eval_batch, self.bucket_cap_mb, self.host_threads) < 1 or self.warmup_steps < 0:
            raise ValueError("评估批量、桶容量、线程数须为正，预热步须非负")


def load_config(path: Path) -> Config:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name in ("seeds", "models", "world_sizes"):
        if name in values:
            values[name] = tuple(values[name])
    return Config(**values)


def run_id(dataset: str, model: str, world_size: int, seed: int) -> str:
    return f"e1_{dataset}_{model}_ws{world_size}_seed{seed}"


def run_folder(output: Path, dataset: str, model: str, world_size: int, seed: int) -> Path:
    return output / dataset / model / f"ws{world_size}_seed{seed}"


def resolve_checkpoint(path: str | Path) -> Path:
    for candidate in (Path(path), PROJECT_ROOT / path, E1_ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)
