"""E0 的逐步、逐轮和最终指标；字段兼容已有结果 CSV。"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

RUNTIME_FIELDS = ["device", "backend", "gpu_name", "host_threads", "timing_mode"]
SEEDS = (1234, 1235, 1236)
SUMMARY_METRICS = (
    "train_throughput",
    "avg_update_ms",
    "avg_sync_ms",
    "sync_share",
    "train_loss",
    "last_val_loss",
    "last_val_acc",
    "test_loss",
    "test_acc",
)
# 三个独立重复、双侧95%区间：t(0.975, df=2)。不使用大样本正态近似1.96。
T95_DF2 = 4.302652729911275


def three_seed_statistics(values: list[float]) -> dict:
    """三种子样本统计和均值95% t区间；区间假设重复近似独立且正态。"""
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("区间需要恰好三个有限的独立种子观测")
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    standard_error = std / math.sqrt(3)
    half_width = T95_DF2 * standard_error
    return {
        "n": 3,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "se": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_seeds(
    records: list[dict], metrics: tuple[str, ...] = SUMMARY_METRICS
) -> list[dict]:
    """对三个完整运行的指标等权汇总，区分观测范围和均值置信区间。"""
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], int(record["ws"]))].append(record)
    summary = []
    for (dataset, world_size), group in sorted(groups.items()):
        if sorted(row["seed"] for row in group) != list(SEEDS):
            raise ValueError(f"{dataset}/GPU{world_size} 必须恰有种子1234、1235、1236")
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary.append(
                {
                    "dataset": dataset,
                    "ws": world_size,
                    "metric": metric,
                    **three_seed_statistics(values),
                }
            )
    return summary


RUN_FIELDS = [
    "run_id",
    "exp",
    "model",
    "ws",
    "k",
    "local_batch",
    "compress",
    "lr",
    "scale_lr",
    "data",
    "epoch",
    "step",
    "in_warmup",
    "t_data_ms",
    "t_fwd_bwd_ms",
    "t_sync_ms",
    "t_opt_ms",
    "t_step_ms",
    "samples",
    "ar_count",
    "ar_bytes",
    "loss",
    *RUNTIME_FIELDS,
]

EPOCH_FIELDS = [
    "run_id",
    "exp",
    "model",
    "ws",
    "k",
    "local_batch",
    "compress",
    "lr",
    "scale_lr",
    "data",
    "epoch",
    "train_loss",
    "test_acc",
    "wall_s",
    *RUNTIME_FIELDS,
]

VALIDATION_EPOCH_FIELDS = [*EPOCH_FIELDS, "val_loss", "val_acc", "is_best"]

FINAL_FIELDS = [
    "run_id",
    "exp",
    "model",
    "ws",
    "k",
    "local_batch",
    "compress",
    "lr",
    "scale_lr",
    "data",
    "seed",
    "split_seed",
    "train_samples",
    "val_samples",
    "test_samples",
    "epochs_ran",
    "best_epoch",
    "best_val_loss",
    "best_val_acc",
    "train_loss",
    "test_loss",
    "test_acc",
    "stopped_early",
    "total_train_samples",
    "total_updates",
    "train_throughput",
    "avg_update_ms",
    "avg_sync_ms",
    "train_time_s",
    "wall_time_s",
    "training_mode",
    "test_epoch",
    "last_val_loss",
    "last_val_acc",
    "checkpoint_path",
    "sync_share",
    *RUNTIME_FIELDS,
]


class CSVWriter:
    """逐行追加写 CSV；文件为空或不存在时先写表头，每行落盘即 flush。"""

    def __init__(self, path: str | Path, fields: list[str]):
        self.path = Path(path)
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        if not new_file:
            with self.path.open(newline="", encoding="utf-8") as existing:
                if next(csv.reader(existing)) != fields:
                    raise ValueError(
                        f"{self.path} 的字段与当前版本不同，请归档后使用新的输出目录"
                    )
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
