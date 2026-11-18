"""复算匹配工作量的吞吐加速比、计算段加速比及双GPU并行效率。"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from E3.config import EXPERIMENT, RESULTS_ROOT
from E3.single_gpu.config import BASELINE_ROOT, cases_for, reference_config, reference_folders

SUMMARY_METRICS = ("single_throughput", "dual_throughput", "speedup", "compute_speedup", "parallel_efficiency",
                   "time_saved_fraction", "single_update_ms", "dual_update_ms", "single_compute_ms", "dual_compute_ms",
                   "dual_sync_ms", "single_local_average_ms", "single_test_acc", "dual_test_acc", "accuracy_difference_pp",
                   "single_peak_memory_mb", "dual_peak_memory_mb", "dual_total_memory_mb", "gpu_cost_ratio")


def dual_record(folder, case):
    folder = Path(folder)
    final = json.loads((folder / "final.json").read_text())
    point = final.get("level", final.get("micro_batch"))
    identity = (final["experiment"], final["dataset"], final["seed"], final["ws"], final["epochs_ran"],
                final["test_epoch"], final["selection"], final["global_batch"], point)
    expected = (EXPERIMENT, case.dataset, case.seed, 2, 20, 20, "last", case.global_batch, case.reference_level)
    if identity != expected:
        raise ValueError(f"双 GPU 配置不匹配：{folder}")
    is_round = (folder / "rounds_rank0.csv").exists()
    frames = []
    memories = []
    for rank in (0, 1):
        runtime = json.loads((folder / f"runtime_rank{rank}.json").read_text())
        if (runtime["device"], runtime["backend"], runtime["world_size"], runtime["dtype"], runtime["allow_tf32"]) != (f"cuda:{rank}", "nccl", 2, "float32", False):
            raise ValueError("双卡来源必须为CUDA/NCCL/FP32")
        frames.append(pd.read_csv(folder / f'{"rounds" if is_round else "steps"}_rank{rank}.csv'))
        memories.append(json.loads((folder / f"memory_rank{rank}.json").read_text())["allocated_bytes"] / 1e6)
    raw = pd.concat(frames, ignore_index=True)
    parent_metrics = importlib.import_module(f"{EXPERIMENT}.metrics")
    config = reference_config(case.dataset)
    if is_round:
        recomputed = parent_metrics.summarize_rounds(raw, config, case.reference_level, final["gradient_bytes"])
    elif "micro_start" in raw:
        recomputed = parent_metrics.summarize_steps(raw, config, case.reference_level, final["variant"], final["gradient_bytes"])
    else:
        recomputed = parent_metrics.summarize_steps(raw, 2, final["micro_batch"], final["accumulation"],
                                                   final["gradient_bytes"], config.warmup_updates)
    for key, value in recomputed.items():
        if value is None and final[key] is None:
            continue
        if not math.isclose(final[key], value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"双卡来源复算不符：{folder}/{key}")
    time_column = "t_round_ms" if is_round else "t_update_ms"
    selected = raw.loc[raw.groupby("iteration")[time_column].idxmax()].sort_values("iteration")
    normalized = pd.DataFrame({"epoch": selected.epoch, "unit": selected["round" if is_round else "step"],
                               "iteration": selected.iteration, "global_samples": selected.local_samples * 2,
                               "steps_count": selected.steps_count if is_round else 1,
                               "measured": selected.measured, "t_unit_ms": selected[time_column],
                               "t_compute_ms": selected.t_compute_ms, "t_sync_ms": selected.t_sync_ms}).reset_index(drop=True)
    final = dict(final, peak_memory_mb=max(memories), total_memory_mb=sum(memories))
    return normalized, final


def assert_matching_units(single, dual):
    columns = ("epoch", "unit", "iteration", "global_samples", "steps_count", "measured")
    if len(single) != len(dual):
        raise ValueError("单双 GPU 的训练单元数不同")
    for key in columns:
        if not np.array_equal(single[key], dual[key]):
            raise ValueError(f"单双 GPU 的{key}不匹配")


def paired_row(case, single, dual_units, dual):
    if single["parameters"] != dual["parameters"] or single["gradient_bytes"] != dual["gradient_bytes"]:
        raise ValueError("单双 GPU 模型规模不一致")
    measured = dual_units[dual_units.measured]
    samples = int(measured.global_samples.sum())
    if samples != single["measured_samples"] or dual["total_train_samples"] != single["total_train_samples"]:
        raise ValueError("单双 GPU 的训练或计时样本数不同")
    dual_ms, compute_ms = measured.t_unit_ms.sum(), measured.t_compute_ms.sum()
    updates = int(measured.steps_count.sum())
    q1, q2 = single["train_throughput"], 1000 * samples / dual_ms
    speedup = q2 / q1
    if not math.isclose(speedup, single["measured_time_s"] / (dual_ms / 1000), rel_tol=1e-10):
        raise ValueError("吞吐比与相同工作量的耗时比不一致")
    target1, target2 = single.get("time_to_target_s"), dual.get("time_to_target_s")
    both_reached = target1 is not None and target2 is not None
    return {"dataset": case.dataset, "level": case.reference_level, "variant": dual.get("variant", "ddp"),
            "seed": case.seed, "single_run_id": single["run_id"], "dual_run_id": dual["run_id"],
            "single_role": case.role, "algorithm_matched": True, "single_micro_batch": case.micro_batch,
            "dual_micro_batch": dual["micro_batch"], "accumulation": case.accumulation, "global_batch": case.global_batch,
            "measured_samples": samples, "measured_updates": updates,
            "single_throughput": q1, "dual_throughput": q2, "speedup": speedup,
            "compute_speedup": single["avg_compute_ms"] / (compute_ms / updates),
            "parallel_efficiency": speedup / 2, "time_saved_fraction": 1 - 1 / speedup,
            "single_update_ms": single["avg_update_ms"], "dual_update_ms": dual_ms / updates,
            "single_compute_ms": single["avg_compute_ms"], "dual_compute_ms": compute_ms / updates,
            "dual_sync_ms": measured.t_sync_ms.sum() / updates,
            "single_local_average_ms": single["avg_local_average_per_update_ms"],
            "single_test_acc": single["test_acc"], "dual_test_acc": dual["test_acc"],
            "accuracy_difference_pp": (dual["test_acc"] - single["test_acc"]) * 100,
            "single_peak_memory_mb": single["peak_allocated_mb"], "dual_peak_memory_mb": dual["peak_memory_mb"],
            "dual_total_memory_mb": dual["total_memory_mb"], "gpu_cost_ratio": 2 / speedup,
            "single_stage_wall_s": single.get("training_stage_wall_s"), "dual_stage_wall_s": dual.get("training_stage_wall_s"),
            "single_target_reached": single.get("target_reached", False), "dual_target_reached": dual.get("target_reached", False),
            "both_targets_reached": both_reached, "single_time_to_target_s": target1, "dual_time_to_target_s": target2,
            "time_to_target_speedup": target1 / target2 if both_reached else None}


def summarize_pairs(frame):
    rows = []
    for (dataset, level, variant), group in frame.groupby(["dataset", "level", "variant"]):
        if sorted(group.seed) != [1234, 1235, 1236]:
            raise ValueError("每个单双卡配对必须具有三个种子")
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = metric != "time_to_target_speedup" or group.both_targets_reached.all()
            rows.append({"dataset": dataset, "level": level, "variant": variant, "metric": metric,
                         "n": int(values.notna().sum()), "mean": values.mean() if valid else None,
                         "std": values.std(ddof=1) if valid else None, "min": values.min(), "max": values.max()})
    return pd.DataFrame(rows)


def compare(root=BASELINE_ROOT, output=RESULTS_ROOT):
    root, output = Path(root), Path(output)
    rows, verified = [], []
    for case in cases_for():
        if case.role == "ordinary_sgd":
            continue
        folder = case.folder(root)
        single = json.loads((folder / "final.json").read_text())
        units = pd.read_csv(folder / "steps.csv")
        config = reference_config(case.dataset)
        for reference in reference_folders(config, case):
            dual_units, dual = dual_record(reference, case)
            assert_matching_units(units, dual_units)
            rows.append(paired_row(case, single, dual_units, dual))
            verified.append({"single_run_id": single["run_id"], "dual_run_id": dual["run_id"],
                             "same_training_and_measured_units": True, "dual_metrics_recomputed": True})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "scaling.csv", index=False)
    summarize_pairs(frame).to_csv(output / "scaling_summary.csv", index=False)
    (output / "scaling_validation.json").write_text(json.dumps({"experiment": EXPERIMENT, "pairs": len(rows),
        "source_single_runs": frame.single_run_id.nunique(), "all_measured_workloads_match": True,
        "speedup_definition": "Q2 / Q1 = measured_T1 / measured_T2 for identical measured samples",
        "compute_speedup_definition": "single compute wall / critical-rank dual compute wall",
        "parallel_efficiency_definition": "speedup / 2", "checked": verified}, indent=2) + "\n")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=BASELINE_ROOT)
    args = parser.parse_args()
    result = compare(args.results_dir)
    print(f"{EXPERIMENT}: verified {len(result)} single/dual pairs")
