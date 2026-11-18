"""从逐rank记录复算指标，并在CUDA上独立复测54个末轮模型。"""

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
from E1.config import (
    DATASETS,
    PRESETS,
    RESULTS_ROOT,
    SEEDS,
    load_config,
    run_folder,
    run_id,
)
from E1.evaluate import evaluate_checkpoint
from E1.metrics import summarize_steps
from E1.model import load_checkpoint


def load_final_runs(results: Path) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for preset in PRESETS:
            for ws in (1, 2):
                for seed in SEEDS:
                    folder = run_folder(results, dataset, preset, ws, seed)
                    row = json.loads((folder / "final.json").read_text())
                    expected = (
                        dataset,
                        preset,
                        ws,
                        seed,
                        run_id(dataset, preset, ws, seed),
                    )
                    if (
                        tuple(
                            row[key]
                            for key in ("dataset", "model", "ws", "seed", "run_id")
                        )
                        != expected
                    ):
                        raise ValueError(f"{folder}: 运行身份不一致")
                    if (
                        row["epochs_ran"],
                        row["test_epoch"],
                        row["selection"],
                        row["local_batch"],
                        row["global_batch"],
                    ) != (20, 20, "last", 128, 128 * ws):
                        raise ValueError(f"{folder}: 不是固定每卡128、20轮的末轮模型")
                    rows.append(row)
    return pd.DataFrame(rows)


def check(
    results: Path = RESULTS_ROOT,
    data_root: Path = DATA_ROOT,
    *,
    records_only: bool = False,
) -> dict:
    finals = load_final_runs(results)
    checked = []
    split_hashes = {}
    for final in finals.to_dict("records"):
        dataset, preset, ws, seed = (
            final[key] for key in ("dataset", "model", "ws", "seed")
        )
        folder = run_folder(results, dataset, preset, ws, seed)
        config = load_config(folder / "config.json")
        dataset_record = json.loads((folder / "dataset.json").read_text())
        split = np.load(folder / "split_indices.npz")
        expected_sizes = {
            "mnist": (54000, 6000, 60000),
            "fashionmnist": (54000, 6000, 60000),
            "cifar10": (45000, 5000, 50000),
        }
        expected_train, expected_val, original = expected_sizes[dataset]
        train_ids, val_ids = split["train"], split["validation"]
        if (len(train_ids), len(val_ids)) != (
            expected_train,
            expected_val,
        ) or not np.array_equal(
            np.sort(np.r_[train_ids, val_ids]), np.arange(original)
        ):
            raise ValueError("训练验证索引不是完整、不相交的固定划分")
        current_hashes = tuple(
            hashlib.sha256(ids.tobytes()).hexdigest() for ids in (train_ids, val_ids)
        )
        if current_hashes != (
            dataset_record["train_indices_sha256"],
            dataset_record["val_indices_sha256"],
        ):
            raise ValueError("划分哈希不匹配")
        if dataset in split_hashes and split_hashes[dataset] != current_hashes:
            raise ValueError("不同模型或种子使用了不同数据划分")
        split_hashes[dataset] = current_hashes
        steps_per_epoch = expected_train // (128 * ws)
        frames = []
        for rank in range(ws):
            runtime = json.loads((folder / f"runtime_rank{rank}.json").read_text())
            if (
                runtime["rank"],
                runtime["world_size"],
                runtime["device"],
                runtime["backend"],
                runtime["timing_mode"],
                runtime["host_threads"],
            ) != (
                rank,
                ws,
                f"cuda:{rank}",
                "nccl" if ws > 1 else "none",
                "cuda_synchronized_wall",
                4,
            ):
                raise ValueError("运行设备、后端、线程或计时方式错误")
            frame = pd.read_csv(folder / f"steps_rank{rank}.csv")
            iterations = np.arange(steps_per_epoch * 20)
            if (
                not np.array_equal(frame.iteration, iterations)
                or not frame["rank"].eq(rank).all()
            ):
                raise ValueError("逐rank记录缺失或顺序错误")
            if not np.array_equal(
                frame.epoch, iterations // steps_per_epoch + 1
            ) or not np.array_equal(frame.step, iterations % steps_per_epoch):
                raise ValueError("轮次或步数错误")
            if not np.array_equal(frame.measured, iterations >= config.warmup_steps):
                raise ValueError("预热排除范围错误")
            frames.append(frame)
        model, saved = load_checkpoint(folder / "checkpoint.pt")
        size = model.size_info()
        for key, value in size.items():
            if final[key] != value:
                raise ValueError(f"模型大小信息 {key} 不匹配")
        if (
            saved["run_id"] != final["run_id"]
            or saved["seed"] != seed
            or saved["training_config"]["local_batch"] != 128
        ):
            raise ValueError("检查点身份或配置不符")
        del model, saved
        gc.collect()
        computed = summarize_steps(
            pd.concat(frames, ignore_index=True), ws, 128, size["gradient_bytes"]
        )
        for key, value in computed.items():
            if not math.isclose(final[key], value, rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(f"{final['run_id']}: {key} 无法从原始记录复算")
        history = pd.read_csv(folder / "epochs.csv")
        if list(history.epoch) != list(range(1, 21)) or any(
            "test" in key for key in history
        ):
            raise ValueError("逐轮记录必须只有训练和验证")
        if not np.isfinite(
            history[["train_loss", "val_loss", "val_acc"]].to_numpy()
        ).all():
            raise ValueError("逐轮指标不是有限值")
        for key in ("train_loss", "val_loss", "val_acc"):
            if not math.isclose(
                final[key], float(history.iloc[-1][key]), rel_tol=1e-9, abs_tol=1e-10
            ):
                raise ValueError("最终指标不对应第20轮")
        peaks = [
            json.loads((folder / f"memory_rank{rank}.json").read_text())
            for rank in range(ws)
        ]
        if final["peak_allocated_mb"] != max(p["allocated_bytes"] for p in peaks) / 1e6:
            raise ValueError("显存峰值不匹配")
        entry = {"run_id": final["run_id"], "records_ok": True}
        if not records_only:
            measured = evaluate_checkpoint(
                folder / "checkpoint.pt",
                device=f"cuda:{len(checked) % torch.cuda.device_count()}",
                data_root=data_root,
            )
            if measured["accuracy"] != final["test_acc"] or not math.isclose(
                measured["loss"], final["test_loss"], rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ValueError("独立GPU复测不匹配")
            entry["independent_test"] = measured
        checked.append(entry)
        print(f"[checked] {final['run_id']}", flush=True)
    output = {
        "experiment": "E1",
        "runs": len(checked),
        "records_only": records_only,
        "timing": "每步取耗时最大的rank整行分段；预热排除；未相加不同rank的分段最大值",
        "checked": checked,
    }
    name = "records_validation.json" if records_only else "validation.json"
    (results / name).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--records-only", action="store_true")
    args = parser.parse_args()
    check(args.results_dir, args.data_root, records_only=args.records_only)
