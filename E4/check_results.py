"""E4检查完整实验网格、原始计时与末轮GPU复测。"""

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
from E4.config import (
    DATASETS,
    SEEDS,
    EXPERIMENT,
    EXPERIMENT_ROOT,
    RESULTS_ROOT,
    load_config,
    run_folder,
    run_id,
)
from E4.model import load_checkpoint
from E4.evaluate import evaluate_checkpoint
from E4.metrics import summarize_steps


def load_final_runs(results=RESULTS_ROOT):
    rows = []
    for dataset in DATASETS:
        c = load_config(EXPERIMENT_ROOT / "configs" / f"{dataset}.yaml")
        for level in c.levels:
            for variant in c.variants:
                for seed in SEEDS:
                    row = json.loads(
                        (
                            run_folder(results, c, level, variant, seed) / "final.json"
                        ).read_text()
                    )
                    keys = (
                        "experiment",
                        "run_id",
                        "dataset",
                        "level",
                        "variant",
                        "seed",
                        "epochs_ran",
                        "test_epoch",
                        "selection",
                        "ws",
                        "global_batch",
                    )
                    expected = (
                        EXPERIMENT,
                        run_id(c, level, variant, seed),
                        dataset,
                        level,
                        variant,
                        seed,
                        20,
                        20,
                        "last",
                        2,
                        c.global_batch_for(level),
                    )
                    if tuple(row[k] for k in keys) != expected:
                        raise ValueError("运行身份不符")
                    rows.append(row)
    return pd.DataFrame(rows)


def check(results=RESULTS_ROOT, data_root=DATA_ROOT, records_only=False):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    finals = load_final_runs(results)
    checked = []
    splits = {}
    orders = {}
    for final in finals.to_dict("records"):
        ds, level, variant, seed = (
            final["dataset"],
            final["level"],
            final["variant"],
            final["seed"],
        )
        c = load_config(EXPERIMENT_ROOT / "configs" / f"{ds}.yaml")
        folder = run_folder(results, c, level, variant, seed)
        if load_config(folder / "config.json") != c:
            raise ValueError("实际配置不同")
        n, nv, original = (
            (45000, 5000, 50000) if ds == "cifar10" else (54000, 6000, 60000)
        )
        ids = np.load(folder / "split_indices.npz")
        a, b = ids["train"], ids["validation"]
        if (len(a), len(b)) != (n, nv) or not np.array_equal(
            np.sort(np.r_[a, b]), np.arange(original)
        ):
            raise ValueError("数据划分错误")
        hashes = tuple(hashlib.sha256(v.tobytes()).hexdigest() for v in (a, b))
        meta = json.loads((folder / "dataset.json").read_text())
        if hashes != (meta["train_indices_sha256"], meta["val_indices_sha256"]):
            raise ValueError("划分哈希不符")
        if ds in splits and splits[ds] != hashes:
            raise ValueError("配置间划分不同")
        splits[ds] = hashes
        frames = []
        count = math.ceil(n / c.global_batch_for(level))
        effective = c.effective_local_batch(level)
        expected_sizes = np.tile(
            [min(effective, n // 2 - j * effective) for j in range(count)], 20
        )
        for rank in (0, 1):
            runtime = json.loads((folder / f"runtime_rank{rank}.json").read_text())
            if (
                runtime["device"],
                runtime["backend"],
                runtime["rank"],
                runtime["world_size"],
                runtime["dtype"],
                runtime["allow_tf32"],
            ) != (f"cuda:{rank}", "nccl", rank, 2, "float32", False):
                raise ValueError("设备协议不符")
            f = pd.read_csv(folder / f"steps_rank{rank}.csv")
            index = np.arange(count * 20)
            if (
                not np.array_equal(f.iteration, index)
                or not np.array_equal(f.epoch, index // count + 1)
                or not np.array_equal(f.step, index % count)
                or not np.array_equal(f.local_samples, expected_sizes)
            ):
                raise ValueError("样本、尾批或更新数错误")
            order = json.loads((folder / f"sampling_rank{rank}.json").read_text())
            key = (ds, seed, rank)
            if (
                set(order) != set(map(str, range(1, 21)))
                or key in orders
                and orders[key] != order
            ):
                raise ValueError("采样顺序变化")
            orders[key] = order
            frames.append(f)
        model, saved = load_checkpoint(folder / "checkpoint.pt")
        size = model.size_info()
        if (saved["run_id"], saved["seed"], saved["level"], saved["variant"]) != (
            final["run_id"],
            seed,
            level,
            variant,
        ):
            raise ValueError("模型身份错误")
        if any(final[k] != v for k, v in size.items()):
            raise ValueError("模型规模变化")
        for key in ("test_acc", "test_loss", "val_acc", "val_loss"):
            if final[key] != saved["metrics"][key]:
                raise ValueError("检查点指标不符")
        del model, saved
        gc.collect()
        recomputed = summarize_steps(
            pd.concat(frames, ignore_index=True),
            c,
            level,
            variant,
            size["gradient_bytes"],
        )
        for k, v in recomputed.items():
            if not math.isclose(final[k], v, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{final['run_id']}: {k}复算不符")
        if (
            final["total_train_samples"] != n * 20
            or final["total_updates"] != count * 20
        ):
            raise ValueError("训练预算不符")
        expected_ar = (
            20 * math.ceil(n / (2 * c.micro_batch))
            if variant == "every_micro"
            else count * 20
        )
        if final["total_gradient_allreduces"] != expected_ar:
            raise ValueError("梯度归约次数错误")
        history = pd.read_csv(folder / "epochs.csv")
        if list(history.epoch) != list(range(1, 21)) or any(
            "test" in k for k in history
        ):
            raise ValueError("验证测试未隔离")
        for k in ("train_loss", "val_acc", "val_loss"):
            if not math.isclose(
                final[k], history.iloc[-1][k], rel_tol=1e-9, abs_tol=1e-10
            ):
                raise ValueError("最终指标非末轮")
        record = dict(run_id=final["run_id"], records_ok=True)
        if not records_only:
            measured = evaluate_checkpoint(
                folder / "checkpoint.pt",
                device=f"cuda:{len(checked) % 2}",
                data_root=data_root,
            )
            if measured["accuracy"] != final["test_acc"] or not math.isclose(
                measured["loss"], final["test_loss"], rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ValueError("GPU复测不符")
            record["independent_test"] = measured
        checked.append(record)
        print(f"[checked] {final['run_id']}", flush=True)
    record = dict(
        experiment=EXPERIMENT,
        runs=len(checked),
        records_only=records_only,
        checked=checked,
        sample_order_consistent=True,
        tail_samples_preserved=True,
    )
    (
        results / ("records_validation.json" if records_only else "validation.json")
    ).write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--records-only", action="store_true")
    args = p.parse_args()
    check(args.results_dir, args.data_root, args.records_only)
