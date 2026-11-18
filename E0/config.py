"""E0 的配置与路径，不依赖命令执行时的工作目录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

E0_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = E0_ROOT.parent
RESULTS_ROOT = E0_ROOT / "results"
EXPERIMENTS = {
    "mnist": "e0_baseline",
    "fashionmnist": "e0_baseline_fashion",
    "cifar10": "e0_baseline_cifar",
}


@dataclass(frozen=True)
class Config:
    dataset: str
    learning_rate: float
    epochs: int = 20
    global_batch: int = 256
    momentum: float = 0.9
    seeds: tuple[int, ...] = (1234, 1235, 1236)
    world_sizes: tuple[int, ...] = (1, 2)
    validation_fraction: float = 0.1
    split_seed: int = 1234
    eval_batch: int = 2048
    warmup_steps: int = 10
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.dataset not in EXPERIMENTS:
            raise ValueError("E0 只使用 MNIST、FashionMNIST 和 CIFAR-10")
        if self.epochs != 20 or self.global_batch != 256 or self.device != "cuda":
            raise ValueError("E0 固定 epochs=20、global_batch=256、device=cuda")
        if tuple(self.seeds) != (1234, 1235, 1236) or tuple(self.world_sizes) != (1, 2):
            raise ValueError("E0 使用三个配对种子和单/双 GPU 对照")
        if not 0 < self.validation_fraction < 1 or self.eval_batch < 1:
            raise ValueError("validation_fraction 必须在 (0,1) 内，eval_batch 须为正数")
        if (
            self.learning_rate <= 0
            or not 0 <= self.momentum < 1
            or self.warmup_steps < 0
        ):
            raise ValueError(
                "学习率须为正，momentum 须在 [0,1) 内，warmup_steps 须非负"
            )


def load_config(path: Path) -> Config:
    with Path(path).open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError("配置必须是 YAML 字典")
    for key in ("seeds", "world_sizes"):
        if key in values:
            values[key] = tuple(values[key])
    return Config(**values)


def run_id(config: Config, seed: int, world_size: int) -> str:
    repeat = tuple(config.seeds).index(seed)
    batch = config.global_batch // world_size
    return f"{EXPERIMENTS[config.dataset]}_k1_ws{world_size}_mM_b{batch}_r{repeat}"


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    for candidate in (path, PROJECT_ROOT / path, E0_ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)
