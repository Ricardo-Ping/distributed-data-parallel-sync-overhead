"""独立GPU时间线诊断：16步预热+64步采集，不混入正式训练性能。"""

from __future__ import annotations
import argparse
import gzip
import json
import shutil
import socket
import tempfile
from pathlib import Path
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, ProfilerActivity, record_function
from Data.datasets import DATA_ROOT
from E5.config import DATASETS, SEEDS, EXPERIMENT_ROOT, RESULTS_ROOT, load_config
from E5.data import load_split
from E5.distributed import initialize, cleanup
from E5.train import Trainer


def union_duration(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if end is None or left > end:
            total += right - left
            end = right
        elif right > end:
            total += right - end
            end = right
    return total


def analyze_trace(events, rank):
    regions = [
        e
        for e in events
        if e.get("name") == "E5_measured_region"
        and e.get("cat") == "user_annotation"
        and e.get("ph") == "X"
    ]
    if len(regions) != 1:
        raise ValueError("Profiler训练区域缺失或重复")
    start = regions[0]["ts"]
    end = start + regions[0]["dur"]
    active = []
    nccl = []
    count = 0
    for e in events:
        if (
            e.get("ph") != "X"
            or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset")
            or e.get("args", {}).get("device") != rank
        ):
            continue
        left = max(start, e["ts"])
        right = min(end, e["ts"] + e["dur"])
        if right <= left:
            continue
        active.append((left, right))
        count += 1
        if "nccl" in e.get("name", "").lower():
            nccl.append((left, right))
    if not count:
        raise RuntimeError("未捕获CUDA活动，不能报告GPU空闲指标")
    busy = union_duration(active)
    span = end - start
    return dict(
        region_wall_ms=span / 1000,
        gpu_activity_ms=busy / 1000,
        gpu_gap_ms=max(0.0, span - busy) / 1000,
        gpu_gap_fraction=max(0.0, 1 - busy / span),
        nccl_kernel_ms=union_duration(nccl) / 1000,
        cuda_activities=count,
        nccl_kernel_count=len(nccl),
    )


def steps(trainer, iterator, start, count):
    for index in range(start, start + count):
        features, labels = next(iterator)
        trainer.local_update(features, labels)
        if (index + 1) % trainer.level == 0:
            trainer.averager.average()


def worker(rank, port, results, data_root, scratch):
    initialize(rank, 2, port, 4)
    rows = []
    try:
        for dataset in DATASETS:
            config = load_config(EXPERIMENT_ROOT / "configs" / f"{dataset}.yaml")
            split = load_split(config, data_root)
            for level in config.levels:
                for seed in SEEDS:
                    trainer = Trainer(
                        config,
                        level,
                        "local_sgd",
                        seed,
                        rank,
                        scratch / dataset / f"h{level}" / f"seed{seed}",
                        split=split,
                    )
                    iterator = iter(trainer.loader)
                    steps(trainer, iterator, 0, 16)
                    dist.barrier()
                    torch.cuda.synchronize(trainer.device)
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                    ) as trace:
                        # CUPTI/Profiler初始化可在两rank耗时不同；对齐发生在测量区域外。
                        dist.barrier()
                        torch.cuda.synchronize(trainer.device)
                        with record_function("E5_measured_region"):
                            steps(trainer, iterator, 16, 64)
                            torch.cuda.synchronize(trainer.device)
                    folder = (
                        results / "profiles" / dataset / f"h{level}" / f"seed{seed}"
                    )
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"rank{rank}.trace.json"
                    trace.export_chrome_trace(str(path))
                    events = json.loads(path.read_text())["traceEvents"]
                    metrics = analyze_trace(events, rank)
                    compressed = path.with_suffix(".json.gz")
                    with (
                        path.open("rb") as source,
                        gzip.open(compressed, "wb") as target,
                    ):
                        shutil.copyfileobj(source, target)
                    path.unlink()
                    rows.append(
                        dict(
                            dataset=dataset,
                            level=level,
                            seed=seed,
                            rank=rank,
                            device=f"cuda:{rank}",
                            warmup_steps=16,
                            profile_steps=64,
                            micro_batch=config.micro_batch,
                            initialization="fresh_seeded_model",
                            formal_training_metrics=False,
                            profile_start_alignment="barrier_after_profiler_initialization",
                            trace=str(compressed.relative_to(results)),
                            **metrics,
                        )
                    )
                    dist.barrier()
                    del trainer, iterator, trace, events
            del split
        pd.DataFrame(rows).to_csv(results / f"profile_rank{rank}.csv", index=False)
    finally:
        cleanup()


def run(results=RESULTS_ROOT, data_root=DATA_ROOT):
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or not dist.is_nccl_available()
    ):
        raise RuntimeError("Profiler诊断需要双CUDA GPU/NCCL")
    results.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="e5_profile_") as folder:
        mp.spawn(
            worker,
            args=(port, results.resolve(), data_root.resolve(), Path(folder)),
            nprocs=2,
            join=True,
        )
    frame = pd.concat(
        [pd.read_csv(results / f"profile_rank{r}.csv") for r in (0, 1)],
        ignore_index=True,
    )
    if len(frame) != 90:
        raise ValueError("Profiler配置不完整")
    frame.to_csv(results / "profile_activity.csv", index=False)
    metrics = [
        "region_wall_ms",
        "gpu_activity_ms",
        "gpu_gap_ms",
        "gpu_gap_fraction",
        "nccl_kernel_ms",
    ]
    paired = frame.groupby(["dataset", "level", "seed"])[metrics].mean().reset_index()
    paired.to_csv(results / "profile_seed_summary.csv", index=False)
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = p.parse_args()
    run(args.results_dir, args.data_root)
