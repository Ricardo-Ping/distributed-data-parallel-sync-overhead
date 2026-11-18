"""交错执行E5双GPU与匹配单GPU基线；普通SGD每个种子只运行一次。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from Data.datasets import DATA_ROOT
from E5.config import DATASETS, RESULTS_ROOT, SEEDS
from E5.single_gpu.config import BASELINE_ROOT, cases_for, reference_config


def plan(dataset="all", seed=None):
    jobs = []
    for name in DATASETS if dataset == "all" else (dataset,):
        serial = cases_for(name, "serial_local_sgd")
        for index, run_seed in enumerate(SEEDS):
            if seed is not None and seed != run_seed:
                continue
            points = [case.reference_level for case in serial if case.seed == run_seed]
            ordinary_position = (0, 2, len(points))[index]
            for offset in range(len(points) + 1):
                if offset == ordinary_position:
                    jobs.append((name, "ordinary_sgd", 1, run_seed))
                if offset == len(points):
                    continue
                modes = (
                    ("dual", "serial_local_sgd")
                    if (index + offset) % 2 == 0
                    else ("serial_local_sgd", "dual")
                )
                jobs.extend((name, mode, points[offset], run_seed) for mode in modes)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--dual-output-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--single-output-dir", type=Path, default=BASELINE_ROOT)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--list", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prime", choices=("dual", "single"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.prime:
        if not torch.cuda.is_available() or torch.cuda.device_count() < (
            2 if args.prime == "dual" else 1
        ):
            raise RuntimeError("E5配对实验需要对应的CUDA GPU，不回退CPU")
        if args.prime == "dual":
            from E5.run import record_environment

            args.dual_output_dir.mkdir(parents=True, exist_ok=True)
            record_environment(
                args.dual_output_dir,
                [reference_config(name) for name in DATASETS],
                args.data_root,
            )
        else:
            from E5.single_gpu.run import record_environment

            args.single_output_dir.mkdir(parents=True, exist_ok=True)
            record_environment(args.single_output_dir, args.data_root)
        return
    jobs = plan(args.dataset, args.seed)
    for name, mode, h, run_seed in jobs:
        print(f"[plan] {name} {mode} H={h} seed={run_seed}", flush=True)
    if args.list:
        return
    common = [
        "--dual-output-dir",
        str(args.dual_output_dir.resolve()),
        "--single-output-dir",
        str(args.single_output_dir.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
    ]
    environments = {
        mode: dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0,1" if mode == "dual" else "0",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            NCCL_DEBUG="WARN",
            PYTHONUNBUFFERED="1",
        )
        for mode in ("dual", "single")
    }
    for mode in ("dual", "single"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "E5.single_gpu.paired_run",
                "--prime",
                mode,
                *common,
            ],
            env=environments[mode],
            check=True,
        )
    for index, (name, mode, h, run_seed) in enumerate(jobs, 1):
        dual = mode == "dual"
        output = args.dual_output_dir if dual else args.single_output_dir
        command = [
            sys.executable,
            "-m",
            "E5.run" if dual else "E5.single_gpu.run",
            "--dataset",
            name,
            "--seed",
            str(run_seed),
            "--level",
            str(h),
            "--output-dir",
            str(output.resolve()),
            "--data-root",
            str(args.data_root.resolve()),
        ]
        if not dual:
            command += ["--role", mode]
        if args.resume:
            command.append("--resume")
        with (output / f"paired_training_{name}.log").open("a") as log:
            subprocess.run(
                command,
                env=environments["dual" if dual else "single"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(
            f"[done] {index}/{len(jobs)} {name} {mode} H={h} seed={run_seed}",
            flush=True,
        )


if __name__ == "__main__":
    main()
