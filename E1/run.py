"""E1 独立运行入口：三个数据集、S/M/L、三个种子、单/双GPU，共54次。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.multiprocessing as mp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Data.datasets import DATA_ROOT
from E1.config import (
    DATASETS,
    E1_ROOT,
    PRESETS,
    PROJECT_ROOT,
    RESULTS_ROOT,
    SEEDS,
    Config,
    load_config,
    run_folder,
)
from E1.train import train_worker


def schedule(config: Config) -> list[tuple[str, int, int]]:
    jobs = []
    for repeat, seed in enumerate(config.seeds):
        models = config.models[repeat:] + config.models[:repeat]
        for index, preset in enumerate(models):
            sizes = (
                config.world_sizes
                if (repeat + index) % 2 == 0
                else tuple(reversed(config.world_sizes))
            )
            jobs.extend((preset, seed, ws) for ws in sizes)
    return jobs


def check_output(
    config: Config, jobs: list[tuple[str, int, int]], output: Path
) -> None:
    for preset, seed, ws in jobs:
        folder = run_folder(output, config.dataset, preset, ws, seed)
        if folder.exists():
            raise FileExistsError(
                f"{folder} 已存在；请指定独立的 --output-dir，禁止追加运行记录"
            )


def record_environment(output: Path, configs: list[Config], data_root: Path) -> None:
    sources = sorted(
        [
            *E1_ROOT.glob("*.py"),
            *E1_ROOT.glob("configs/*.yaml"),
            PROJECT_ROOT / "Data/datasets.py",
        ]
    )
    hashes = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sources
    }
    archive = output / "provenance"
    if archive.exists():
        if json.loads((archive / "source_sha256.json").read_text()) != hashes:
            raise ValueError("输出目录的代码快照不同，不能继续混用结果")
        return
    archive.mkdir()
    for path in sources:
        target = archive / path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (archive / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True
    ).stdout
    gpu_details = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    environment = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()),
        "gpu_details": gpu_details,
        "topology": topology,
        "p2p_0_to_1": torch.cuda.can_device_access_peer(0, 1)
        if torch.cuda.device_count() > 1
        else None,
        "data_root": str(data_root.resolve()),
        "configs": [asdict(c) for c in configs],
        "protocol": "E1_local_batch128_fp32_blocking_one_bucket_20epochs_three_seeds",
        "source_sha256": hashes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")


def run(
    configs: list[Config],
    output: Path = RESULTS_ROOT,
    data_root: Path = DATA_ROOT,
    seed: int | None = None,
    model: str | None = None,
    world_size: int | None = None,
    *,
    list_only: bool = False,
) -> None:
    planned = [
        (
            config,
            [
                (p, s, ws)
                for p, s, ws in schedule(config)
                if (seed is None or seed == s)
                and (model is None or model == p)
                and (world_size is None or world_size == ws)
            ],
        )
        for config in configs
    ]
    for config, jobs in planned:
        for preset, s, ws in jobs:
            print(
                f"{config.dataset}: model={preset}{PRESETS[preset]}, seed={s}, GPUs={ws}, "
                f"local_batch=128, global_batch={128 * ws}, epochs=20",
                flush=True,
            )
    if list_only:
        return
    required = max((ws for _, jobs in planned for _, _, ws in jobs), default=0)
    if required == 0:
        raise ValueError("没有匹配的实验")
    if not torch.cuda.is_available() or torch.cuda.device_count() < required:
        raise RuntimeError(f"需要{required}张可见CUDA GPU")
    if required > 1 and not torch.distributed.is_nccl_available():
        raise RuntimeError("需要NCCL")
    for config, jobs in planned:
        check_output(config, jobs, output)
    output.mkdir(parents=True, exist_ok=True)
    record_environment(output, configs, data_root)
    for config, jobs in planned:
        for preset, s, ws in jobs:
            folder = run_folder(output, config.dataset, preset, ws, s)
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(
                json.dumps(asdict(config), indent=2) + "\n"
            )
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(
                train_worker,
                args=(
                    ws,
                    port,
                    config,
                    preset,
                    s,
                    folder.resolve(),
                    data_root.resolve(),
                ),
                nprocs=ws,
                join=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--model", choices=tuple(PRESETS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--world-size", type=int, choices=(1, 2))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    configs = [
        load_config(E1_ROOT / "configs" / f"{dataset}.yaml") for dataset in datasets
    ]
    run(
        configs,
        args.output_dir.resolve(),
        args.data_root,
        args.seed,
        args.model,
        args.world_size,
        list_only=args.list,
    )


if __name__ == "__main__":
    main()
