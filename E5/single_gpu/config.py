"""Local SGD 同算法串行双副本基线，以及普通单模型 SGD 参考。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from E5.config import (
    DATASETS,
    EXPERIMENT_ROOT,
    LEVELS,
    RESULTS_ROOT,
    SEEDS,
    load_config,
    run_folder,
)

BASELINE_ROOT = RESULTS_ROOT / "single_gpu"
POINTS = LEVELS
ROLES = ("serial_local_sgd", "ordinary_sgd")


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
    role: str = "serial_local_sgd"
    replicas: int = 2

    def __post_init__(self):
        if (
            self.dataset not in DATASETS
            or self.seed not in SEEDS
            or self.reference_level not in POINTS
        ):
            raise ValueError("不在正式 E5 基线网格中")
        if self.role not in ROLES or self.accumulation != 1 or self.global_batch != 256:
            raise ValueError("E5 每个本地 batch 都立即更新，全局每步256样本")
        expected = (128, 2) if self.role == "serial_local_sgd" else (256, 1)
        if (self.micro_batch, self.replicas) != expected:
            raise ValueError("逻辑副本数与 micro-batch 不符")
        if self.role == "ordinary_sgd" and self.reference_level != 1:
            raise ValueError("普通单模型 SGD 只训练一次，不按 H 伪造重复运行")

    @property
    def identifier(self):
        kind = (
            f"serial_h{self.reference_level}"
            if self.role == "serial_local_sgd"
            else "ordinary_sgd"
        )
        return f"e5_single_{self.dataset}_{kind}_seed{self.seed}"

    def folder(self, root: Path = BASELINE_ROOT):
        kind = (
            f"serial_h{self.reference_level}"
            if self.role == "serial_local_sgd"
            else "ordinary_sgd"
        )
        return Path(root) / self.dataset / kind / f"seed{self.seed}"

    @property
    def units_filename(self):
        return "rounds.csv" if self.role == "serial_local_sgd" else "steps.csv"


def reference_config(dataset):
    return load_config(EXPERIMENT_ROOT / "configs" / f"{dataset}.yaml")


def make_case(config, point, seed, role="serial_local_sgd"):
    serial = role == "serial_local_sgd"
    return Case(
        config.dataset,
        point,
        seed,
        config.micro_batch if serial else 2 * config.micro_batch,
        1,
        2 * config.micro_batch,
        config.warmup_updates,
        role=role,
        replicas=2 if serial else 1,
    )


def cases_for(dataset="all", role="all"):
    if role not in ("all", *ROLES):
        raise ValueError("未知 E5 单卡基线角色")
    jobs = []
    for name in DATASETS if dataset == "all" else (dataset,):
        config = reference_config(name)
        for index, seed in enumerate(SEEDS):
            points = POINTS[index:] + POINTS[:index]
            if role in ("all", "serial_local_sgd"):
                jobs.extend(make_case(config, point, seed) for point in points)
            if role in ("all", "ordinary_sgd"):
                jobs.append(make_case(config, 1, seed, "ordinary_sgd"))
    return jobs


def reference_folders(config, case):
    return (
        run_folder(RESULTS_ROOT, config, case.reference_level, "local_sgd", case.seed),
    )


def saved_config(case):
    return {
        "case": asdict(case),
        "reference_config": asdict(reference_config(case.dataset)),
    }
