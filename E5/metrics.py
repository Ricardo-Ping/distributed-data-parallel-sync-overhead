"""E5按完整H步同步轮次统计，选同一关键rank整行分段。"""

import csv
from pathlib import Path
import numpy as np
import pandas as pd

PHASES = ("t_data_ms", "t_compute_ms", "t_sync_ms", "t_opt_ms", "t_other_ms")
ROUND_FIELDS = [
    "epoch",
    "round",
    "iteration",
    "first_step",
    "global_first_step",
    "rank",
    "steps_count",
    "local_samples",
    "full_round",
    "measured",
    *PHASES,
    "t_round_ms",
    "diagnostic_ms",
    "t_round_with_diagnostic_ms",
    "t_pack_ms",
    "t_collective_ms",
    "t_copy_ms",
    "sync_arrival_s",
    "ar_count",
    "ar_bytes",
    "drift_measured",
    "drift_relative_l2",
    "loss",
]
LOCAL_STEP_FIELDS = [
    "epoch",
    "step",
    "iteration",
    "round_iteration",
    "rank",
    "measured",
    "local_samples",
    "t_data_ms",
    "t_compute_ms",
    "t_sync_ms",
    "t_opt_ms",
    "t_other_ms",
    "t_update_ms",
    "local_throughput",
    "sync_happened",
    "sync_type",
    "ar_count",
    "ar_bytes",
    "loss",
]
SUMMARY_METRICS = (
    "train_throughput",
    "avg_update_ms",
    "avg_sync_ms",
    "avg_sync_per_update_ms",
    "sync_share",
    "sync_us_per_sample",
    "avg_compute_per_update_ms",
    "arrival_gap_ms",
    "drift_mean_relative_l2",
    "drift_max_relative_l2",
    "total_model_allreduces",
    "total_logical_payload_bytes",
    "test_acc",
    "test_loss",
    "training_stage_wall_s",
)


def write_rows(path, rows, append=False):
    with Path(path).open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROUND_FIELDS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def write_local_steps(path, rows, append=False):
    with Path(path).open("a" if append else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOCAL_STEP_FIELDS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def summarize_rounds(frame, config, level, payload_bytes):
    if frame.empty or frame.duplicated(["iteration", "rank"]).any():
        raise ValueError("轮次记录为空或重复")
    if (
        set(frame["rank"]) != {0, 1}
        or not frame.groupby("iteration")["rank"].nunique().eq(2).all()
    ):
        raise ValueError("rank不完整")
    if (
        not np.isfinite(
            frame[[*PHASES, "t_round_ms", "diagnostic_ms", "sync_arrival_s", "loss"]]
        )
        .all()
        .all()
    ):
        raise ValueError("记录非有限")
    if (frame[list(PHASES)] < -1e-7).any().any() or not np.allclose(
        frame[list(PHASES)].sum(axis=1), frame.t_round_ms, rtol=1e-10, atol=1e-9
    ):
        raise ValueError("分段计时无效")
    if not np.allclose(
        frame.t_round_ms + frame.diagnostic_ms,
        frame.t_round_with_diagnostic_ms,
        rtol=1e-10,
        atol=1e-9,
    ):
        raise ValueError("诊断时间扣除错误")
    full = frame.steps_count.eq(level) & frame.local_samples.eq(
        level * config.micro_batch
    )
    if not np.array_equal(frame.full_round, full) or not np.array_equal(
        frame.measured, full & frame.global_first_step.ge(config.warmup_updates)
    ):
        raise ValueError("完整H步轮次或预热标记错误")
    if not frame.ar_count.eq(1).all() or not frame.ar_bytes.eq(payload_bytes).all():
        raise ValueError("每轮须一次完整参数归约")
    if not frame.steps_count.between(1, level).all():
        raise ValueError("同步间隔越界")
    for key in (
        "local_samples",
        "steps_count",
        "measured",
        "global_first_step",
        "drift_measured",
    ):
        if not frame.groupby("iteration")[key].nunique().eq(1).all():
            raise ValueError("rank同步窗口不一致")
    selected = frame.loc[frame.groupby("iteration").t_round_ms.idxmax()].sort_values(
        "iteration"
    )
    steady = selected[selected.measured]
    if steady.empty:
        raise ValueError("无稳态完整H步轮次")
    u = int(steady.steps_count.sum())
    rounds = len(steady)
    n = int(steady.local_samples.sum()) * 2
    t = steady.t_round_ms.sum()
    s = steady.t_sync_ms.sum()
    c = steady.t_compute_ms.sum()
    arrived = (
        frame[frame.measured].groupby("iteration").sync_arrival_s.agg(["min", "max"])
    )
    drift = frame[frame.drift_measured].groupby(["epoch"]).drift_relative_l2.mean()
    return dict(
        total_rounds=len(selected),
        total_updates=int(selected.steps_count.sum()),
        total_train_samples=int(selected.local_samples.sum()) * 2,
        total_model_allreduces=int(selected.ar_count.sum()),
        total_logical_payload_bytes=int(selected.ar_bytes.sum()),
        measured_rounds=rounds,
        measured_updates=u,
        measured_samples=n,
        measured_time_s=t / 1000,
        train_throughput=n * 1000 / t,
        avg_update_ms=t / u,
        avg_sync_ms=s / rounds,
        avg_sync_per_update_ms=s / u,
        sync_share=s / t,
        sync_us_per_sample=s * 1000 / n,
        avg_compute_per_update_ms=c / u,
        avg_data_per_update_ms=steady.t_data_ms.sum() / u,
        avg_opt_per_update_ms=steady.t_opt_ms.sum() / u,
        avg_other_per_update_ms=steady.t_other_ms.sum() / u,
        arrival_gap_ms=float((arrived["max"] - arrived["min"]).mean() * 1000),
        drift_mean_relative_l2=float(drift.mean()) if len(drift) else None,
        drift_max_relative_l2=float(drift.max()) if len(drift) else None,
    )


def seed_summary(frame):
    rows = []
    for (dataset, level), group in frame.groupby(["dataset", "level"]):
        if sorted(group.seed) != [1234, 1235, 1236]:
            raise ValueError("每个H须有三个种子")
        for metric in SUMMARY_METRICS:
            v = group[metric]
            rows.append(
                dict(
                    dataset=dataset,
                    level=level,
                    metric=metric,
                    n=int(v.notna().sum()),
                    mean=v.mean(),
                    std=v.std(ddof=1),
                    min=v.min(),
                    max=v.max(),
                )
            )
    return pd.DataFrame(rows)
