"""E0 运行入口；按种子交替启动单 GPU 与双 GPU 配对实验。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.multiprocessing as mp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Data.datasets import DATA_ROOT, DATASETS
from E0.config import E0_ROOT, RESULTS_ROOT, Config, load_config, run_id
from E0.train import train_worker


def schedule(config: Config) -> list[tuple[int, int]]:
    return [
        (seed, ws)
        for repeat, seed in enumerate(config.seeds)
        for ws in (
            config.world_sizes
            if repeat % 2 == 0
            else tuple(reversed(config.world_sizes))
        )
    ]


def check_output(config: Config, jobs: list[tuple[int, int]], output: Path) -> None:
    selected = {run_id(config, seed, ws) for seed, ws in jobs}
    folder = output / config.dataset
    saved_config = folder / "config.json"
    if saved_config.exists() and json.loads(saved_config.read_text()) != json.loads(
        json.dumps(asdict(config))
    ):
        raise ValueError(f"{folder} 已有不同的配置，请使用新的输出目录")
    for name in ("runs.csv", "epochs.csv", "final.csv"):
        path = folder / name
        if path.exists():
            with path.open(newline="") as file:
                previous = {row["run_id"] for row in csv.DictReader(file)}
            if previous & selected:
                raise FileExistsError(
                    f"{folder} 已有所选运行的记录；请先归档或指定新的 --output-dir"
                )
    for identifier in selected:
        if (folder / "checkpoints" / f"{identifier}_epoch20.pt").exists():
            raise FileExistsError(f"已存在模型 {identifier}，请使用新的输出目录")


def run(
    configs: list[Config],
    output: Path = RESULTS_ROOT,
    data_root: Path = DATA_ROOT,
    seed: int | None = None,
    world_size: int | None = None,
    *,
    list_only: bool = False,
) -> None:
    planned = [
        (
            config,
            [
                (s, ws)
                for s, ws in schedule(config)
                if (seed is None or seed == s)
                and (world_size is None or world_size == ws)
            ],
        )
        for config in configs
    ]
    for config, jobs in planned:
        for s, ws in jobs:
            print(
                f"{config.dataset}: seed={s}, GPUs={ws}, local_batch={config.global_batch // ws}, epochs=20"
            )
    if list_only:
        return
    if not any(jobs for _, jobs in planned):
        raise ValueError("没有匹配的运行组合")
    required = max(ws for _, jobs in planned for _, ws in jobs)
    if not torch.cuda.is_available() or torch.cuda.device_count() < required:
        raise RuntimeError(f"需要 {required} 张可见 CUDA GPU")
    if required > 1 and not torch.distributed.is_nccl_available():
        raise RuntimeError("双 GPU 训练需要 NCCL")
    for config, jobs in planned:
        check_output(config, jobs, output)
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()),
        "host_logical_cores": os.cpu_count(),
        "gpus": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "data_root": str(data_root.resolve()),
        "configs": [asdict(c) for c in configs],
    }
    (output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n"
    )
    for config, jobs in planned:
        folder = output / config.dataset
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n"
        )
        for s, ws in jobs:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(
                train_worker,
                args=(ws, port, config, s, output.resolve(), data_root.resolve()),
                nprocs=ws,
                join=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--seed", type=int, choices=(1234, 1235, 1236))
    parser.add_argument("--world-size", type=int, choices=(1, 2))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    names = DATASETS if args.dataset == "all" else (args.dataset,)
    configs = [load_config(E0_ROOT / "configs" / f"{name}.yaml") for name in names]
    run(
        configs,
        args.output_dir,
        args.data_root,
        args.seed,
        args.world_size,
        list_only=args.list,
    )


if __name__ == "__main__":
    main()
