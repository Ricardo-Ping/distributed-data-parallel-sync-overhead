"""校验单GPU网格、原始计时、样本匹配和第20轮模型的独立CUDA复测。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Data.datasets import DATA_ROOT
from E3.config import EXPERIMENT
from E3.evaluate import evaluate_checkpoint
from E3.model import load_checkpoint
from E3.single_gpu.compare import assert_matching_units, dual_record
from E3.single_gpu.config import BASELINE_ROOT, cases_for, reference_config, reference_folders, saved_config
from E3.single_gpu.metrics import summarize
from E3.single_gpu.run import RAW_FILES


def check(root=BASELINE_ROOT, data_root=DATA_ROOT):
    root = Path(root)
    if not torch.cuda.is_available():
        raise RuntimeError("独立模型复测必须使用CUDA")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    checked = []
    for case in cases_for():
        folder = case.folder(root)
        if not all((folder / name).is_file() for name in RAW_FILES):
            raise ValueError(f"单GPU记录不完整：{folder}")
        actual_config = json.loads((folder / "config.json").read_text())
        if actual_config != json.loads(json.dumps(saved_config(case))):
            raise ValueError("单 GPU 实际配置不符")
        final = json.loads((folder / "final.json").read_text())
        identity = tuple(final[key] for key in ("run_id", "baseline_role", "ws", "epochs_ran", "test_epoch", "selection", "global_batch", "logical_replicas"))
        if identity != (case.identifier, case.role, 1, 20, 20, "last", case.global_batch, case.replicas):
            raise ValueError("单 GPU 运行身份错误")
        runtime = json.loads((folder / "runtime.json").read_text())
        if (runtime["device"], runtime["world_size"], runtime["backend"], runtime["dtype"], runtime["allow_tf32"], runtime["logical_replicas"]) != ("cuda:0", 1, None, "float32", False, case.replicas):
            raise ValueError("基线并非单进程单GPU FP32")
        frame = pd.read_csv(folder / "steps.csv")
        if not np.array_equal(frame.iteration, np.arange(len(frame))):
            raise ValueError("计时单元不连续")
        performance = summarize(frame, case)
        for key, value in performance.items():
            if not math.isclose(final[key], value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{case.identifier}/{key}复算不符")
        history = pd.read_csv(folder / "epochs.csv")
        if list(history.epoch) != list(range(1, 21)) or any("test" in key for key in history):
            raise ValueError("训练轮数或验证测试隔离错误")
        for key in ("train_loss", "val_loss", "val_acc"):
            if not math.isclose(final[key], history.iloc[-1][key], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError("最终指标不属于第20轮")
        target = reference_config(case.dataset).__dict__.get("target_accuracy")
        if target is not None:
            reached = history[history.val_acc.ge(target)]
            if final["target_reached"] != bool(len(reached)):
                raise ValueError("验证达标标记错误")
            if len(reached):
                first = reached.iloc[0]
                if final["target_epoch"] != int(first.epoch) or not math.isclose(final["time_to_target_s"], first.elapsed_wall_s, rel_tol=1e-10):
                    raise ValueError("Time-to-Target不是首次验证达标")
            elif final["target_epoch"] is not None or final["time_to_target_s"] is not None:
                raise ValueError("未达标运行不应有达标时间")
        sampling = json.loads((folder / "sampling.json").read_text())
        if set(sampling) != set(map(str, range(1, 21))):
            raise ValueError("样本顺序记录缺少epoch")
        split = np.load(folder / "split_indices.npz")
        meta = json.loads((folder / "dataset.json").read_text())
        config = reference_config(case.dataset)
        for reference in reference_folders(config, case):
            reference_meta = json.loads((reference / "dataset.json").read_text())
            if meta != reference_meta:
                raise ValueError("单双GPU数据划分或数据量不一致")
            for name, key in (("train", "train_indices_sha256"), ("validation", "val_indices_sha256")):
                if hashlib.sha256(split[name].tobytes()).hexdigest() != meta[key]:
                    raise ValueError("数据划分索引哈希不符")
            for rank in (0, 1):
                reference_order = json.loads((reference / f"sampling_rank{rank}.json").read_text())
                if any(sampling[str(epoch)][f"reference_rank{rank}_sha256"] != reference_order[str(epoch)] for epoch in range(1, 21)):
                    raise ValueError("单卡全局样本顺序不能还原双卡rank顺序")
            dual_units, dual = dual_record(reference, case)
            assert_matching_units(frame, dual_units)
        if final["total_train_samples"] != 20 * meta["train_samples"]:
            raise ValueError("未完整使用20轮全部训练样本")
        peak = json.loads((folder / "memory.json").read_text())["allocated_bytes"] / 1e6
        if peak != final["peak_allocated_mb"]:
            raise ValueError("显存记录不符")
        device = f"cuda:{len(checked) % torch.cuda.device_count()}"
        model, saved = load_checkpoint(folder / "checkpoint.pt", device)
        if any(final[key] != value for key, value in model.size_info().items()):
            raise ValueError("模型规模与最终记录不符")
        if saved["baseline_case"] != asdict(case) or saved["run_id"] != case.identifier or saved["ws"] != 1:
            raise ValueError("检查点与基线不符")
        for key in ("val_loss", "val_acc", "test_loss", "test_acc"):
            if saved["metrics"][key] != final[key]:
                raise ValueError("检查点指标与最终记录不符")
        del model, saved
        gc.collect()
        tested = evaluate_checkpoint(folder / "checkpoint.pt", device=device, data_root=data_root)
        if tested["accuracy"] != final["test_acc"] or not math.isclose(tested["loss"], final["test_loss"], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("末轮模型GPU独立复测不符")
        checked.append({"run_id": case.identifier, "records_ok": True, "sample_order_matches_dual": True,
                        "measured_workload_matches_dual": True, "independent_test": tested})
        print(f"[checked] {case.identifier}", flush=True)
    record = {"experiment": EXPERIMENT, "runs": len(checked), "all_models_retested_on_cuda": True,
              "all_records_recomputed": True, "all_sample_orders_match": True, "checked": checked}
    (root / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    check(args.results_dir, args.data_root)
