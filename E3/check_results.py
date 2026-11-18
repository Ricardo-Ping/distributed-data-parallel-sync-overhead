"""E3：原始更新记录、尾批、样本顺序与末轮GPU模型复测。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Data.datasets import DATA_ROOT
from E3.config import (
    DATASETS,
    EXPERIMENT,
    EXPERIMENT_ROOT,
    RESULTS_ROOT,
    SEEDS,
    load_config,
    run_folder,
    run_id,
)
from E3.evaluate import evaluate_checkpoint
from E3.metrics import summarize_steps
from E3.model import load_checkpoint


def load_final_runs(results: Path) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        config = load_config(EXPERIMENT_ROOT / "configs" / f"{dataset}.yaml")
        for batch in config.batch_sizes:
            for seed in SEEDS:
                folder = run_folder(results, config, batch, seed)
                row = json.loads((folder / "final.json").read_text())
                expected = (
                    EXPERIMENT,
                    run_id(config, batch, seed),
                    dataset,
                    batch,
                    config.accumulation_for(batch),
                    seed,
                    2,
                    config.global_batch_for(batch),
                    20,
                    20,
                    "last",
                    "M",
                )
                keys = (
                    "experiment",
                    "run_id",
                    "dataset",
                    "micro_batch",
                    "accumulation",
                    "seed",
                    "ws",
                    "global_batch",
                    "epochs_ran",
                    "test_epoch",
                    "selection",
                    "model",
                )
                if tuple(row[key] for key in keys) != expected:
                    raise ValueError(f"{folder}: 运行身份或训练协议不符")
                rows.append(row)
    return pd.DataFrame(rows)


def check(
    results: Path = RESULTS_ROOT,
    data_root: Path = DATA_ROOT,
    *,
    records_only: bool = False,
) -> dict:
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    finals = load_final_runs(results)
    checked = []
    split_hashes, sampling = {}, {}
    for final in finals.to_dict("records"):
        dataset, batch, seed = final["dataset"], final["micro_batch"], final["seed"]
        config = load_config(EXPERIMENT_ROOT / "configs" / f"{dataset}.yaml")
        folder = run_folder(results, config, batch, seed)
        saved_config = load_config(folder / "config.json")
        if saved_config != config:
            raise ValueError("实际配置与正式配置不同")
        n_train, n_val, original = (
            (45000, 5000, 50000) if dataset == "cifar10" else (54000, 6000, 60000)
        )
        split = np.load(folder / "split_indices.npz")
        a, b = split["train"], split["validation"]
        if (len(a), len(b)) != (n_train, n_val) or not np.array_equal(
            np.sort(np.r_[a, b]), np.arange(original)
        ):
            raise ValueError("数据划分不是互斥且完整的训练验证划分")
        metadata = json.loads((folder / "dataset.json").read_text())
        hashes = tuple(hashlib.sha256(ids.tobytes()).hexdigest() for ids in (a, b))
        if hashes != (metadata["train_indices_sha256"], metadata["val_indices_sha256"]):
            raise ValueError("数据索引哈希不符")
        if dataset in split_hashes and split_hashes[dataset] != hashes:
            raise ValueError("配置之间数据划分不同")
        split_hashes[dataset] = hashes
        effective = config.effective_local_batch(batch)
        updates = math.ceil(n_train / config.global_batch_for(batch))
        frames = []
        for rank in (0, 1):
            runtime = json.loads((folder / f"runtime_rank{rank}.json").read_text())
            if (
                runtime["device"],
                runtime["backend"],
                runtime["rank"],
                runtime["world_size"],
                runtime["host_threads"],
                runtime["timing_mode"],
            ) != (f"cuda:{rank}", "nccl", rank, 2, 4, "cuda_synchronized_update_wall"):
                raise ValueError("GPU运行元数据不符")
            frame = pd.read_csv(folder / f"steps_rank{rank}.csv")
            iterations = np.arange(updates * 20)
            expected_samples = np.tile(
                [
                    min(effective, n_train // 2 - step * effective)
                    for step in range(updates)
                ],
                20,
            )
            if (
                not np.array_equal(frame.iteration, iterations)
                or not frame["rank"].eq(rank).all()
            ):
                raise ValueError("逐rank更新记录缺失或重复")
            if (
                not np.array_equal(frame.local_samples, expected_samples)
                or not np.array_equal(frame.epoch, iterations // updates + 1)
                or not np.array_equal(frame.step, iterations % updates)
            ):
                raise ValueError("尾批、轮次或更新数不符")
            order = json.loads((folder / f"sampling_rank{rank}.json").read_text())
            key = (dataset, seed, rank)
            if set(order) != set(map(str, range(1, 21))) or (
                key in sampling and sampling[key] != order
            ):
                raise ValueError("不同实现的样本顺序不一致")
            sampling[key] = order
            frames.append(frame)
        model, saved = load_checkpoint(folder / "checkpoint.pt")
        size = model.size_info()
        for key, value in size.items():
            if final[key] != value:
                raise ValueError("模型参数量或梯度载荷不符")
        if (
            saved["run_id"] != final["run_id"]
            or saved["seed"] != seed
            or saved["global_batch"] != config.global_batch_for(batch)
        ):
            raise ValueError("模型检查点身份不符")
        for key in ("test_loss", "test_acc", "val_loss", "val_acc"):
            if saved["metrics"][key] != final[key]:
                raise ValueError("检查点指标与最终记录不符")
        del model, saved
        gc.collect()
        recomputed = summarize_steps(
            pd.concat(frames, ignore_index=True),
            2,
            batch,
            config.accumulation_for(batch),
            size["gradient_bytes"],
            config.warmup_updates,
        )
        for key, value in recomputed.items():
            if not math.isclose(final[key], value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{final['run_id']}: {key}复算失败")
        if (
            final["total_train_samples"] != n_train * 20
            or final["total_updates"] != updates * 20
        ):
            raise ValueError("没有完整使用每轮全部训练样本")
        history = pd.read_csv(folder / "epochs.csv")
        if list(history.epoch) != list(range(1, 21)) or any(
            "test" in key for key in history
        ):
            raise ValueError("逐轮只能记录训练和验证")
        for key in ("train_loss", "val_loss", "val_acc"):
            if not math.isclose(
                final[key], float(history.iloc[-1][key]), rel_tol=1e-9, abs_tol=1e-10
            ):
                raise ValueError("最终指标不属于第20轮")
        peaks = [
            json.loads((folder / f"memory_rank{rank}.json").read_text())
            for rank in (0, 1)
        ]
        if (
            final["peak_allocated_mb"]
            != max(value["allocated_bytes"] for value in peaks) / 1e6
        ):
            raise ValueError("显存记录不符")
        entry = dict(run_id=final["run_id"], records_ok=True)
        if not records_only:
            tested = evaluate_checkpoint(
                folder / "checkpoint.pt",
                device=f"cuda:{len(checked) % 2}",
                data_root=data_root,
            )
            if tested["accuracy"] != final["test_acc"] or not math.isclose(
                tested["loss"], final["test_loss"], rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ValueError("独立GPU复测不匹配")
            entry["independent_test"] = tested
        checked.append(entry)
        print(f"[checked] {final['run_id']}", flush=True)
    record = dict(
        experiment=EXPERIMENT,
        runs=len(checked),
        records_only=records_only,
        checked=checked,
        sample_order_consistent=True,
        tail_samples_preserved=True,
        one_gradient_allreduce_per_update=True,
    )
    (
        results / ("records_validation.json" if records_only else "validation.json")
    ).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--records-only", action="store_true")
    args = parser.parse_args()
    check(args.results_dir, args.data_root, records_only=args.records_only)
