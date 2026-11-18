"""保持全局 batch 和累积次数，单 GPU micro-batch 为双卡每卡值的两倍。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from E3.config import (
    BATCHES,
    DATASETS,
    EXPERIMENT,
    EXPERIMENT_ROOT,
    RESULTS_ROOT,
    SEEDS,
    load_config,
    run_folder,
)

BASELINE_ROOT = RESULTS_ROOT / "single_gpu"
POINTS = BATCHES
ROLES = ("global_batch_matched",)


@dataclass(frozen=True)
class Case:
    dataset: str
    reference_level: int
    seed: int
    micro_batch: int
    accumulation: int
    global_batch: int
    warmup_updates: int
    warmup_micro_steps: int = 0
    role: str = "global_batch_matched"
    replicas: int = 1

    def __post_init__(self):
        if (
            self.dataset not in DATASETS
            or self.seed not in SEEDS
            or self.reference_level not in POINTS
        ):
            raise ValueError("不在正式基线网格中")
        if (
            self.global_batch != self.micro_batch * self.accumulation
            or self.role not in ROLES
        ):
            raise ValueError("单 GPU 全局 batch 或基线角色错误")
        if min(self.micro_batch, self.accumulation) < 1 or self.replicas != 1:
            raise ValueError("单模型基线配置错误")

    @property
    def identifier(self):
        return f"{EXPERIMENT.lower()}_{self.dataset}_single_ref{self.reference_level}_b{self.micro_batch}_k{self.accumulation}_seed{self.seed}"

    def folder(self, root: Path = BASELINE_ROOT):
        return (
            Path(root)
            / self.dataset
            / f"ref{self.reference_level}"
            / f"seed{self.seed}"
        )


def reference_config(dataset):
    return load_config(EXPERIMENT_ROOT / "configs" / f"{dataset}.yaml")


def make_case(config, point, seed):
    if point not in POINTS:
        raise ValueError("未知双 GPU 每卡 batch")
    return Case(
        config.dataset,
        point,
        seed,
        2 * point,
        config.accumulation_for(point),
        config.global_batch_for(point),
        config.warmup_updates,
    )


def cases_for(dataset="all", role="all"):
    if role not in ("all", *ROLES):
        raise ValueError("未知基线角色")
    jobs = []
    for name in DATASETS if dataset == "all" else (dataset,):
        config = reference_config(name)
        for index, seed in enumerate(SEEDS):
            points = POINTS[index:] + POINTS[:index]
            jobs.extend(make_case(config, point, seed) for point in points)
    return jobs


def reference_folders(config, case):
    return (run_folder(RESULTS_ROOT, config, case.reference_level, case.seed),)


def saved_config(case):
    # reference_config 是双卡来源；case 是本次实际单卡配置，两者均显式保留。
    return {
        "case": asdict(case),
        "reference_config": asdict(reference_config(case.dataset)),
    }
