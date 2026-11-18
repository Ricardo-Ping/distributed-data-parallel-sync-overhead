"""从E5独立Profiler原始时间线复算活动指标和训练通信kernel次数。"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from E5.profile_activity import analyze_trace

ROOT = Path(__file__).resolve().parents[1] / "E5/results"


def run():
    frame = pd.read_csv(ROOT / "profile_activity.csv")
    if len(frame) != 90 or frame.duplicated(["dataset", "level", "seed", "rank"]).any():
        raise ValueError("独立Profiler网格不完整")
    hashes, checked = {}, []
    for row in frame.to_dict("records"):
        path = ROOT / row["trace"]
        with gzip.open(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        metrics = analyze_trace(events, row["rank"])
        for key, value in metrics.items():
            if not math.isclose(row[key], value, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Profiler指标无法从时间线复算：{path}/{key}")
        if metrics["nccl_kernel_count"] != 64 // row["level"]:
            raise ValueError("64步内实际NCCL kernel次数与H不符")
        if row["warmup_steps"] != 16 or row["profile_steps"] != 64 or row["formal_training_metrics"]:
            raise ValueError("独立Profiler测量范围有误")
        if row["profile_start_alignment"] != "barrier_after_profiler_initialization":
            raise ValueError("Profiler初始化后未明确对齐两rank")
        hashes[str(path.relative_to(ROOT.parent.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        checked.append({"dataset": row["dataset"], "level": row["level"], "seed": row["seed"],
                        "rank": row["rank"], "metrics_recomputed": True,
                        "nccl_kernels": metrics["nccl_kernel_count"]})
    paired = pd.read_csv(ROOT / "profile_seed_summary.csv")
    expected = frame.groupby(["dataset", "level", "seed"])[["region_wall_ms", "gpu_activity_ms", "gpu_gap_ms", "gpu_gap_fraction", "nccl_kernel_ms"]].mean().reset_index()
    pd.testing.assert_frame_equal(paired, expected, check_exact=False, rtol=1e-10, atol=1e-10)
    (ROOT / "profile_traces_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    result = {"traces": len(checked), "all_metrics_recomputed": True, "all_nccl_counts_match_64_div_H": True,
              "two_ranks_averaged_before_seed_summary": True, "checks": checked}
    (ROOT / "profile_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("E5: all 90 independent CUDA traces and 45 two-rank summaries passed", flush=True)


if __name__ == "__main__":
    run()
