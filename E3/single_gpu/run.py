"""单 GPU 基线启动器；每次训练使用独立子进程，正式计时不并行运行。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from Data.datasets import DATA_ROOT
from E3.config import DATASETS, EXPERIMENT, EXPERIMENT_ROOT, PROJECT_ROOT, SEEDS
from E3.single_gpu.config import BASELINE_ROOT, ROLES, Case, cases_for, saved_config


RAW_FILES = (
    "config.json",
    "dataset.json",
    "split_indices.npz",
    "runtime.json",
    "sampling.json",
    "steps.csv",
    "epochs.csv",
    "memory.json",
    "checkpoint.pt",
    "final.json",
)


def complete(case, root):
    folder = case.folder(root)
    if not folder.exists():
        return False
    if not all((folder / name).is_file() for name in RAW_FILES):
        raise FileExistsError(f"{folder} 不完整，请先归档该次运行")
    final = json.loads((folder / "final.json").read_text())
    expected = (case.identifier, case.role, 1, 20, 20, "last", case.global_batch)
    actual = tuple(
        final.get(key)
        for key in (
            "run_id",
            "baseline_role",
            "ws",
            "epochs_ran",
            "test_epoch",
            "selection",
            "global_batch",
        )
    )
    normalized = json.loads(json.dumps(saved_config(case)))
    if (
        actual != expected
        or json.loads((folder / "config.json").read_text()) != normalized
    ):
        raise ValueError("现有单 GPU 结果与计划不一致")
    return True


def record_environment(root, data_root):
    files = [
        *Path(__file__).parent.glob("*.py"),
        *EXPERIMENT_ROOT.glob("configs/*.yaml"),
        *Path(__file__).parent.glob("configs/*.yaml"),
        PROJECT_ROOT / "Data/datasets.py",
    ]
    files += [
        EXPERIMENT_ROOT / name
        for name in (
            "config.py",
            "data.py",
            "model.py",
            "evaluate.py",
            "distributed.py",
        )
    ]
    hashes = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(set(files))
    }
    snapshot = root / "provenance"
    if (snapshot / "source_sha256.json").exists():
        previous = json.loads((snapshot / "source_sha256.json").read_text())
        core = lambda source: {
            key: value
            for key, value in source.items()
            if not key.endswith(("check_results.py", "plot_results.py", "compare.py"))
        }
        if core(previous) != core(hashes):
            raise ValueError("单 GPU 训练来源已变化，请指定新输出目录")
        return
    snapshot.mkdir(parents=True)
    for path in files:
        target = snapshot / path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (snapshot / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")

    def capture(args):
        return subprocess.run(args, text=True, capture_output=True, check=True).stdout

    environment = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "experiment": EXPERIMENT,
        "role": "matched_single_gpu_baselines",
        "physical_gpus_per_run": 1,
        "training_processes_per_run": 1,
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "data_root": str(data_root.resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "gpu_details": capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit",
                "--format=csv",
            ]
        ),
        "host_context": capture(["lscpu"]),
        "source_sha256": hashes,
        "gpu_policy": "cuda:0 only; no process group and no cross-GPU NCCL",
    }
    (root / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n"
    )
    (root / "plan.json").write_text(
        json.dumps([saved_config(case) for case in cases_for()], indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--level", type=int, help="对应双 GPU 配置的每卡 batch 或 K/H")
    parser.add_argument("--role", choices=("all", *ROLES), default="all")
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_case:
        from E3.single_gpu.train import worker

        config = json.loads(args.worker_case.read_text())
        worker(Case(**config["case"]), args.worker_case.parent, args.data_root)
        return
    cases = [
        case
        for case in cases_for(args.dataset, args.role)
        if (args.seed is None or case.seed == args.seed)
        and (args.level is None or case.reference_level == args.level)
    ]
    for case in cases:
        print(
            f"{case.identifier}: role={case.role}, GPUs=1, replicas={case.replicas}, micro={case.micro_batch}, K={case.accumulation}, global={case.global_batch}, epochs=20",
            flush=True,
        )
    if args.list:
        return
    root = args.output_dir.resolve()
    if args.resume:
        cases = [case for case in cases if not complete(case, root)]
        if not cases:
            print("[resume] 所选单 GPU 运行已全部完成", flush=True)
            return
    if not cases:
        raise ValueError("没有匹配的运行")
    if not torch.cuda.is_available():
        raise RuntimeError("单 GPU 基线必须使用 CUDA，不回退 CPU")
    for case in cases:
        if case.folder(root).exists():
            raise FileExistsError(f"{case.folder(root)} 已存在，拒绝覆盖")
    root.mkdir(parents=True, exist_ok=True)
    record_environment(root, args.data_root)
    for case in cases:
        folder = case.folder(root)
        folder.mkdir(parents=True)
        config_path = folder / "config.json"
        config_path.write_text(json.dumps(saved_config(case), indent=2) + "\n")
        subprocess.run(
            [
                sys.executable,
                "-m",
                f"{EXPERIMENT}.single_gpu.run",
                "--worker-case",
                str(config_path),
                "--data-root",
                str(args.data_root.resolve()),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
