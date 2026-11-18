"""保留两个 rank 的原始时间，按每步最慢 rank 计算可加和的分段指标。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PHASES = ("t_data_ms", "t_compute_ms", "t_sync_ms", "t_opt_ms")
STEP_FIELDS = [
    "epoch",
    "step",
    "iteration",
    "rank",
    "measured",
    "samples",
    *PHASES,
    "t_step_ms",
    "ar_count",
    "ar_bytes",
    "loss",
]
SUMMARY_METRICS = (
    "avg_compute_ms",
    "avg_sync_ms",
    "avg_update_ms",
    "avg_data_ms",
    "avg_opt_ms",
    "sync_share",
    "compute_share",
    "sync_compute_ratio",
    "sync_compute_share",
    "train_throughput",
    "p50_update_ms",
    "p95_update_ms",
    "peak_allocated_mb",
    "test_acc",
)


def write_rows(
    path: Path, rows: list[dict], fields: list[str], *, append: bool = False
) -> None:
    with path.open("a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not append or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def summarize_steps(
    frame: pd.DataFrame, world_size: int, local_batch: int, gradient_bytes: int
) -> dict:
    if frame.empty or frame.duplicated(["iteration", "rank"]).any():
        raise ValueError("逐步数据为空或有重复rank")
    if set(frame["rank"]) != set(range(world_size)):
        raise ValueError("缺少rank")
    if not frame.groupby("iteration")["rank"].nunique().eq(world_size).all():
        raise ValueError("某次更新的rank不完整")
    if not frame.samples.eq(local_batch).all():
        raise ValueError("每卡batch必须固定，不能混入尾批")
    if not np.isfinite(frame[[*PHASES, "t_step_ms", "loss"]].to_numpy()).all():
        raise ValueError("指标包含非有限值")
    if (frame[list(PHASES)] < 0).any().any() or not np.allclose(
        frame[list(PHASES)].sum(axis=1), frame.t_step_ms
    ):
        raise ValueError("时间分段无法加和")
    if (
        not frame.ar_count.eq(int(world_size > 1)).all()
        or not frame.ar_bytes.eq(gradient_bytes if world_size > 1 else 0).all()
    ):
        raise ValueError("必须每步一次完整梯度AllReduce；单卡无梯度同步")
    if world_size == 1 and not frame.t_sync_ms.eq(0).all():
        raise ValueError("单卡同步时间须为0")
    if not frame.groupby("iteration").measured.nunique().eq(1).all():
        raise ValueError("rank之间预热标记不一致")
    chosen = frame.loc[frame.groupby("iteration").t_step_ms.idxmax()].sort_values(
        "iteration"
    )
    steady = chosen[chosen.measured.astype(bool)]
    if steady.empty:
        raise ValueError("没有预热后的计时样本")
    totals = steady[list(PHASES)].sum()
    total_ms, updates = float(steady.t_step_ms.sum()), len(steady)
    compute, sync = float(totals.t_compute_ms), float(totals.t_sync_ms)
    return {
        "total_updates": len(chosen),
        "measured_updates": updates,
        "total_train_samples": len(chosen) * local_batch * world_size,
        "measured_samples": updates * local_batch * world_size,
        "measured_time_s": total_ms / 1000,
        "avg_data_ms": float(totals.t_data_ms) / updates,
        "avg_compute_ms": compute / updates,
        "avg_sync_ms": sync / updates,
        "avg_opt_ms": float(totals.t_opt_ms) / updates,
        "avg_update_ms": total_ms / updates,
        "sync_share": sync / total_ms,
        "compute_share": compute / total_ms,
        "sync_compute_ratio": sync / compute,
        "sync_compute_share": sync / (compute + sync),
        "train_throughput": updates * local_batch * world_size * 1000 / total_ms,
        "p50_update_ms": float(steady.t_step_ms.quantile(0.5)),
        "p95_update_ms": float(steady.t_step_ms.quantile(0.95)),
    }


def seed_summary(finals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, preset, ws), group in finals.groupby(["dataset", "model", "ws"]):
        if sorted(group.seed) != [1234, 1235, 1236]:
            raise ValueError("每组必须有三个不同的训练种子")
        for metric in SUMMARY_METRICS:
            values = group[metric]
            rows.append(
                {
                    "dataset": dataset,
                    "model": preset,
                    "ws": ws,
                    "metric": metric,
                    "n": 3,
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def paired_scaling(finals: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "seed"]
    pairs = finals[finals.ws.eq(1)].merge(
        finals[finals.ws.eq(2)],
        on=keys,
        suffixes=("_single", "_dual"),
        validate="one_to_one",
    )
    if len(pairs) * 2 != len(finals):
        raise ValueError("单卡与双卡配对不完整")
    pairs["throughput_scaling"] = (
        pairs.train_throughput_dual / pairs.train_throughput_single
    )
    pairs["weak_scaling_efficiency"] = pairs.throughput_scaling / 2
    pairs["update_time_ratio"] = pairs.avg_update_ms_dual / pairs.avg_update_ms_single
    return pairs[
        [*keys, "throughput_scaling", "weak_scaling_efficiency", "update_time_ratio"]
    ]
