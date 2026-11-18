"""E3：两张GPU，全局batch固定256，用micro-batch和累积改变实现。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

EXPERIMENT = "E3"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
DATASETS = ("mnist", "fashionmnist", "cifar10")
SEEDS = (1234, 1235, 1236)
BATCHES = (128, 64, 32, 16)


@dataclass(frozen=True)
class Config:
    dataset: str
    learning_rate: float
    epochs: int = 20
    batch_sizes: tuple[int, ...] = BATCHES
    world_size: int = 2
    momentum: float = .9
    seeds: tuple[int, ...] = SEEDS
    validation_fraction: float = .1
    split_seed: int = 1234
    eval_batch: int = 1024
    warmup_updates: int = 10
    bucket_cap_mb: int = 25
    host_threads: int = 4
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS or self.learning_rate <= 0:
            raise ValueError("数据集或学习率无效")
        if (self.epochs, self.world_size, self.device) != (20, 2, "cuda"):
            raise ValueError("E3 固定20轮、两张GPU和CUDA")
        if self.batch_sizes != BATCHES or self.seeds != SEEDS:
            raise ValueError("E3 使用规定的四档micro-batch和三个种子")
        if not 0 < self.validation_fraction < 1 or not 0 <= self.momentum < 1:
            raise ValueError("验证比例或动量无效")
        if min(self.eval_batch, self.bucket_cap_mb, self.host_threads) < 1 or self.warmup_updates < 0:
            raise ValueError("评估批量、桶容量、线程数须为正，预热更新须非负")

    def accumulation_for(self, batch: int) -> int:
        if batch not in self.batch_sizes:
            raise ValueError("不在正式batch配置中")
        return 256 // (self.world_size * batch)

    def effective_local_batch(self, batch: int) -> int:
        return batch * self.accumulation_for(batch)

    def global_batch_for(self, batch: int) -> int:
        return self.world_size * self.effective_local_batch(batch)


def load_config(path: Path) -> Config:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("batch_sizes", "seeds"):
        if key in values:
            values[key] = tuple(values[key])
    return Config(**values)


def run_id(config: Config, batch: int, seed: int) -> str:
    return f"e3_{config.dataset}_b{batch}_k{config.accumulation_for(batch)}_g{config.global_batch_for(batch)}_seed{seed}"


def run_folder(output: Path, config: Config, batch: int, seed: int) -> Path:
    return output / config.dataset / f"b{batch}_k{config.accumulation_for(batch)}" / f"seed{seed}"


def resolve_checkpoint(path: str | Path) -> Path:
    for candidate in (Path(path), PROJECT_ROOT / path, EXPERIMENT_ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)
