"""单 GPU 按完整工作单元计时，跨卡通信必须为零。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PHASES = ("t_data_ms", "t_compute_ms", "t_local_average_ms", "t_sync_ms", "t_opt_ms", "t_other_ms")
FIELDS = ["epoch", "unit", "iteration", "first_update", "global_first_update", "steps_count",
          "global_samples", "micro_count", "micro_start", "full_batch", "measured", *PHASES,
          "t_unit_ms", "diagnostic_ms", "t_unit_with_diagnostic_ms", "post_unit_ms", "optimizer_calls",
          "local_average_count", "ar_count", "ar_bytes", "drift_measured",
          "drift_relative_l2", "drift_replica0_relative_l2", "drift_replica1_relative_l2", "loss"]
LOCAL_FIELDS=["epoch","step","iteration","replica","round_iteration","local_samples",
              "t_data_ms","t_compute_ms","t_opt_ms","t_local_average_ms","t_sync_ms","t_other_ms",
              "t_update_ms","local_throughput","local_average_count","ar_count","ar_bytes","loss"]


def write_rows(path, rows, append=False):
    with Path(path).open("a" if append else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def write_local_steps(path, rows, append=False):
    with Path(path).open("a" if append else "w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=LOCAL_FIELDS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def summarize(frame, case):
    if frame.empty or frame.iteration.duplicated().any():
        raise ValueError("基线记录为空或重复")
    if not np.isfinite(frame[[*PHASES, "t_unit_ms", "diagnostic_ms", "loss"]]).all().all():
        raise ValueError("基线计时或损失非有限")
    if (frame[list(PHASES)] < -1e-7).any().any():
        raise ValueError("基线分段耗时为负")
    if not np.allclose(frame[list(PHASES)].sum(axis=1), frame.t_unit_ms, rtol=1e-10, atol=1e-8):
        raise ValueError("单元时间分段无法加和")
    if not np.allclose(frame.t_unit_ms + frame.diagnostic_ms, frame.t_unit_with_diagnostic_ms,
                       rtol=1e-10, atol=1e-8):
        raise ValueError("诊断扣时错误")
    if not frame.ar_count.eq(0).all() or not frame.ar_bytes.eq(0).all() or not frame.t_sync_ms.eq(0).all():
        raise ValueError("单 GPU 不应出现跨卡通信")
    if not np.isfinite(frame.post_unit_ms).all() or (frame.post_unit_ms < 0).any():
        raise ValueError("单元间日志与控制耗时无效")
    serial = case.role == "serial_local_sgd"
    if serial:
        if not frame.steps_count.between(1, case.reference_level).all():
            raise ValueError("本地更新步数越界")
        if not np.array_equal(frame.optimizer_calls, frame.steps_count * 2):
            raise ValueError("每步必须串行执行两个副本的SGD")
        if not np.array_equal(frame.micro_count, frame.steps_count * 2) or not frame.local_average_count.eq(1).all():
            raise ValueError("每轮应执行两份本地工作和一次设备内模型平均")
        full = frame.steps_count.eq(case.reference_level) & frame.global_samples.eq(case.global_batch * case.reference_level)
        measured = full & frame.global_first_update.ge(case.warmup_updates)
        for epoch, rows in frame.groupby("epoch"):
            full_steps = int(rows.global_samples.sum()) // case.global_batch
            diagnostic_end = (full_steps // case.reference_level) * case.reference_level
            expected = rows.first_update + rows.steps_count == diagnostic_end
            if not np.array_equal(rows.drift_measured, expected) or int(rows.drift_measured.sum()) != 1:
                raise ValueError("每epoch必须在最后一个完整H步轮次诊断一次漂移")
        observed = frame[frame.drift_measured]
        if not np.isfinite(observed[["drift_relative_l2", "drift_replica0_relative_l2", "drift_replica1_relative_l2"]]).all().all():
            raise ValueError("模型漂移诊断缺失")
        if not np.allclose(observed.drift_relative_l2,
                           (observed.drift_replica0_relative_l2 + observed.drift_replica1_relative_l2) / 2,
                           rtol=1e-9, atol=1e-12):
            raise ValueError("漂移没有按两个逻辑副本平均")
    else:
        if not frame.steps_count.eq(1).all() or not frame.optimizer_calls.eq(1).all():
            raise ValueError("每个累积窗口必须只更新一次")
        micros = np.ceil(frame.global_samples / case.micro_batch).astype(int)
        if not np.array_equal(frame.micro_count, micros):
            raise ValueError("实际 micro 数错误")
        full = frame.global_samples.eq(case.global_batch)
        measured = full & frame.global_first_update.ge(case.warmup_updates) & frame.micro_start.ge(case.warmup_micro_steps)
        if not frame.local_average_count.eq(0).all() or frame.drift_measured.any():
            raise ValueError("普通单模型SGD不应执行副本平均或漂移诊断")
    starts = frame.micro_count.cumsum().shift(fill_value=0)
    if not np.array_equal(frame.micro_start, starts):
        raise ValueError("micro 计数不连续")
    if not np.array_equal(frame.full_batch, full) or not np.array_equal(frame.measured, measured):
        raise ValueError("完整窗口或预热标记错误")
    steady = frame[frame.measured]
    if steady.empty:
        raise ValueError("没有预热后的完整计时单元")
    updates, units = int(steady.steps_count.sum()), len(steady)
    samples, total_ms = int(steady.global_samples.sum()), float(steady.t_unit_ms.sum())
    return {"total_units": len(frame), "total_updates": int(frame.steps_count.sum()),
            "total_optimizer_calls": int(frame.optimizer_calls.sum()),
            "total_backward_calls": int(frame.micro_count.sum()),
            "total_train_samples": int(frame.global_samples.sum()),
            "total_local_averages": int(frame.local_average_count.sum()),
            "total_gradient_allreduces": 0, "total_model_allreduces": 0, "total_logical_payload_bytes": 0,
            "measured_units": units, "measured_updates": updates, "measured_samples": samples,
            "measured_time_s": total_ms / 1000, "train_throughput": samples * 1000 / total_ms,
            "avg_unit_ms": total_ms / units, "avg_update_ms": total_ms / updates,
            "avg_compute_ms": float(steady.t_compute_ms.sum()) / updates,
            "avg_data_ms": float(steady.t_data_ms.sum()) / updates,
            "avg_opt_ms": float(steady.t_opt_ms.sum()) / updates,
            "avg_other_ms": float(steady.t_other_ms.sum()) / updates,
            "avg_sync_ms": 0.0, "sync_share": 0.0,
            "avg_local_average_ms": float(steady.t_local_average_ms.sum()) / units,
            "avg_local_average_per_update_ms": float(steady.t_local_average_ms.sum()) / updates,
            "local_average_share": float(steady.t_local_average_ms.sum()) / total_ms,
            "drift_mean_relative_l2": float(frame.loc[frame.drift_measured, "drift_relative_l2"].mean()) if serial else 0.0,
            "drift_max_relative_l2": float(frame.loc[frame.drift_measured, "drift_relative_l2"].max()) if serial else 0.0}
