"""E2：固定双GPU，五档每卡batch，三个数据集、三个种子，共45次运行。"""

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
from E2.config import (
    DATASETS,
    EXPERIMENT,
    EXPERIMENT_ROOT,
    PROJECT_ROOT,
    RESULTS_ROOT,
    SEEDS,
    Config,
    load_config,
    run_folder,
)
from E2.train import train_worker


def schedule(config: Config) -> list[tuple[int, int]]:
    return [
        (batch, seed)
        for index, seed in enumerate(config.seeds)
        for batch in config.batch_sizes[index:] + config.batch_sizes[:index]
    ]


def check_output(config: Config, jobs: list[tuple[int, int]], output: Path) -> None:
    for batch, seed in jobs:
        folder = run_folder(output, config, batch, seed)
        if folder.exists():
            raise FileExistsError(f"{folder} 已存在，请先归档或选择新的输出目录")


def record_environment(
    output: Path, configs: list[Config], data_root: Path, *, resume: bool = False
) -> None:
    paths = sorted(
        [
            *EXPERIMENT_ROOT.glob("*.py"),
            *EXPERIMENT_ROOT.glob("configs/*.yaml"),
            PROJECT_ROOT / "Data/datasets.py",
            PROJECT_ROOT / "training_common/epoch_metrics.py",
        ]
    )
    hashes = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    provenance = output / "provenance"
    if provenance.exists():
        previous = json.loads((provenance / "source_sha256.json").read_text())
        if previous != hashes:
            changed = {
                key
                for key in previous.keys() | hashes.keys()
                if previous.get(key) != hashes.get(key)
            }
            if not resume or changed != {f"{EXPERIMENT}/run.py"}:
                raise ValueError("训练或分析代码快照已变化，请使用新的输出目录")
            version = (
                provenance / "launcher_versions" / hashes[f"{EXPERIMENT}/run.py"][:12]
            )
            version.mkdir(parents=True, exist_ok=True)
            shutil.copy2(EXPERIMENT_ROOT / "run.py", version / "run.py")
            (version / "change.json").write_text(
                json.dumps(
                    dict(
                        changed_files=sorted(changed),
                        source_sha256=hashes,
                        reason="启动器更新；训练、计时、同步与配置来源保持一致",
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        return
    provenance.mkdir()
    for path in paths:
        target = provenance / path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (provenance / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")

    def capture(args):
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout

    environment = dict(
        recorded_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        experiment=EXPERIMENT,
        platform=platform.platform(),
        python=sys.version,
        pytorch=str(torch.__version__),
        cuda=torch.version.cuda,
        nccl=list(torch.cuda.nccl.version()),
        gpu_details=capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit",
                "--format=csv",
            ]
        ),
        topology=capture(["nvidia-smi", "topo", "-m"]),
        host_context=capture(["lscpu"]),
        p2p_0_to_1=torch.cuda.can_device_access_peer(0, 1),
        source_sha256=hashes,
        data_root=str(data_root.resolve()),
        configs=[asdict(config) for config in configs],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    )
    (output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n"
    )


def completed(folder: Path, config: Config, batch: int, seed: int) -> bool:
    if not folder.exists():
        return False
    required = (
        "final.json",
        "checkpoint.pt",
        "epochs.csv",
        "steps_rank0.csv",
        "steps_rank1.csv",
        "runtime_rank0.json",
        "runtime_rank1.json",
        "sampling_rank0.json",
        "sampling_rank1.json",
        "memory_rank0.json",
        "memory_rank1.json",
        "dataset.json",
        "split_indices.npz",
        "config.json",
    )
    if not all((folder / name).is_file() for name in required):
        raise FileExistsError(f"{folder} 不完整，请先归档该次运行后再续跑")
    final = json.loads((folder / "final.json").read_text())
    keys = (
        "experiment",
        "dataset",
        "micro_batch",
        "seed",
        "epochs_ran",
        "test_epoch",
        "selection",
        "ws",
        "global_batch",
        "accumulation",
        "model",
    )
    identity = tuple(final.get(key) for key in keys)
    expected = (
        EXPERIMENT,
        config.dataset,
        batch,
        seed,
        20,
        20,
        "last",
        2,
        config.global_batch_for(batch),
        config.accumulation_for(batch),
        "M",
    )
    if identity != expected or load_config(folder / "config.json") != config:
        raise ValueError(f"{folder} 已有结果与当前计划不符")
    return True


def launch(
    config: Config, batch: int, seed: int, folder: Path, data_root: Path
) -> None:
    for attempt in range(3):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        try:
            mp.spawn(
                train_worker,
                args=(port, config, batch, seed, folder.resolve(), data_root.resolve()),
                nprocs=2,
                join=True,
            )
            return
        except mp.ProcessRaisedException as error:
            untouched = {path.name for path in folder.iterdir()} <= {"config.json"}
            if attempt == 2 or not untouched or "EADDRINUSE" not in str(error):
                raise
            print(
                f"[retry] NCCL启动端口{port}已占用，重新选择端口（{attempt + 1}/2）",
                flush=True,
            )


def run(
    configs: list[Config],
    output: Path = RESULTS_ROOT,
    data_root: Path = DATA_ROOT,
    seed: int | None = None,
    batch: int | None = None,
    *,
    list_only: bool = False,
    resume: bool = False,
) -> None:
    planned = [
        (
            config,
            [
                (b, s)
                for b, s in schedule(config)
                if (batch is None or b == batch) and (seed is None or s == seed)
            ],
        )
        for config in configs
    ]
    for config, jobs in planned:
        for b, s in jobs:
            print(
                f"{EXPERIMENT}/{config.dataset}: seed={s}, GPUs=2, micro_batch={b}, k={config.accumulation_for(b)}, global_batch={config.global_batch_for(b)}, epochs=20",
                flush=True,
            )
    if list_only:
        return
    if resume:
        planned = [
            (
                config,
                [
                    (b, s)
                    for b, s in jobs
                    if not completed(run_folder(output, config, b, s), config, b, s)
                ],
            )
            for config, jobs in planned
        ]
        if not any(jobs for _, jobs in planned):
            print("[resume] 所选运行已全部完成", flush=True)
            return
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or not torch.distributed.is_nccl_available()
    ):
        raise RuntimeError("需要两张可见CUDA GPU与NCCL")
    if not any(jobs for _, jobs in planned):
        raise ValueError("没有匹配的配置")
    for config, jobs in planned:
        check_output(config, jobs, output)
    output.mkdir(parents=True, exist_ok=True)
    record_environment(output, configs, data_root, resume=resume)
    for config, jobs in planned:
        for b, s in jobs:
            folder = run_folder(output, config, b, s)
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(
                json.dumps(asdict(config), indent=2) + "\n"
            )
            launch(config, b, s, folder, data_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="跳过已完整保存的运行，拒绝覆盖未完成目录"
    )
    args = parser.parse_args()
    configs = [
        load_config(EXPERIMENT_ROOT / "configs" / f"{name}.yaml")
        for name in (DATASETS if args.dataset == "all" else (args.dataset,))
    ]
    run(
        configs,
        args.output_dir.resolve(),
        args.data_root,
        args.seed,
        args.batch,
        list_only=args.list,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
