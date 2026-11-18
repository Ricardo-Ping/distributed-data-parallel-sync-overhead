"""整次更新计时，明确区分训练尾批、预热、micro-batch与参数更新。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PHASES = ("t_data_ms", "t_compute_ms", "t_sync_ms", "t_opt_ms", "t_other_ms")
STEP_FIELDS = ["epoch", "step", "iteration", "rank", "full_batch", "measured", "local_samples",
               "micro_count", "sync_micro_count", *PHASES, "t_update_ms", "ar_count", "ar_bytes", "loss"]
SUMMARY_METRICS = ("train_throughput", "avg_update_ms", "avg_compute_ms", "avg_sync_ms", "sync_share",
                   "compute_share", "local_compute_throughput", "sync_ms_per_sample", "peak_allocated_mb",
                   "avg_data_ms", "avg_opt_ms", "avg_other_ms", "p50_update_ms", "p95_update_ms", "test_acc", "test_loss")


def write_rows(path: Path, rows: list[dict], *, append: bool = False) -> None:
    with path.open("a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STEP_FIELDS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def summarize_steps(frame: pd.DataFrame, world_size: int, batch: int, accumulation: int,
                    gradient_bytes: int, warmup_updates: int) -> dict:
    effective = batch * accumulation
    if frame.empty or frame.duplicated(["iteration", "rank"]).any():
        raise ValueError("原始记录为空或重复")
    if set(frame['rank']) != set(range(world_size)) or not frame.groupby('iteration')['rank'].nunique().eq(world_size).all():
        raise ValueError("rank不完整")
    if not np.isfinite(frame[[*PHASES, 't_update_ms', 'loss']].to_numpy()).all() or (frame[list(PHASES)] < 0).any().any():
        raise ValueError("非有限指标或负耗时")
    if not np.allclose(frame[list(PHASES)].sum(axis=1), frame.t_update_ms, rtol=1e-10, atol=1e-9):
        raise ValueError("分段时间无法加和")
    if not frame.ar_count.eq(1).all() or not frame.ar_bytes.eq(gradient_bytes).all() or not frame.sync_micro_count.eq(1).all():
        raise ValueError("每次参数更新必须恰好同步一次完整梯度")
    if not frame.local_samples.between(1, effective).all() or not np.array_equal(frame.micro_count, np.ceil(frame.local_samples / batch)):
        raise ValueError("窗口样本数或micro-batch数错误")
    if not np.array_equal(frame.full_batch, frame.local_samples.eq(effective)):
        raise ValueError("完整批次标记错误")
    expected = frame.full_batch & frame.iteration.ge(warmup_updates)
    if not np.array_equal(frame.measured, expected):
        raise ValueError("预热/尾批筛选错误")
    for field in ('local_samples', 'measured', 'full_batch', 'micro_count'):
        if not frame.groupby('iteration')[field].nunique().eq(1).all():
            raise ValueError("rank之间的窗口不一致")
    selected = frame.loc[frame.groupby('iteration').t_update_ms.idxmax()].sort_values('iteration')
    steady = selected[selected.measured]
    if steady.empty:
        raise ValueError("没有预热后的完整更新")
    updates, total_ms = len(steady), float(steady.t_update_ms.sum())
    totals = steady[list(PHASES)].sum()
    compute, sync = float(totals.t_compute_ms), float(totals.t_sync_ms)
    samples = int(steady.local_samples.sum()) * world_size
    return {"total_updates": len(selected), "total_train_samples": int(selected.local_samples.sum()) * world_size,
            "measured_updates": updates, "measured_samples": samples, "measured_time_s": total_ms/1000,
            "train_throughput": samples*1000/total_ms, "avg_update_ms": total_ms/updates,
            "avg_data_ms": float(totals.t_data_ms)/updates, "avg_compute_ms": compute/updates,
            "avg_sync_ms": sync/updates, "avg_opt_ms": float(totals.t_opt_ms)/updates,
            "avg_other_ms": float(totals.t_other_ms)/updates,
            "sync_share": sync/total_ms, "compute_share": compute/total_ms,
            "local_compute_throughput": samples/world_size*1000/compute,
            "sync_ms_per_sample": sync/samples,
            "gradient_bytes_per_sample": gradient_bytes/(world_size*effective),
            "p50_update_ms": float(steady.t_update_ms.quantile(.5)),
            "p95_update_ms": float(steady.t_update_ms.quantile(.95))}


def seed_summary(finals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, batch), group in finals.groupby(['dataset', 'micro_batch']):
        if sorted(group.seed) != [1234, 1235, 1236]:
            raise ValueError("每个配置必须恰好有三个种子")
        for metric in SUMMARY_METRICS:
            values = group[metric]
            rows.append(dict(dataset=dataset, micro_batch=batch, accumulation=int(group.accumulation.iloc[0]),
                             global_batch=int(group.global_batch.iloc[0]), metric=metric, n=3,
                             mean=values.mean(), std=values.std(ddof=1), min=values.min(), max=values.max()))
    return pd.DataFrame(rows)
