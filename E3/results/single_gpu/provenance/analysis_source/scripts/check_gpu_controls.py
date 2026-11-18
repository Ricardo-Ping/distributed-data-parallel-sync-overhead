"""在CUDA上比较相同配置控制模型及E5新旧计时版本，不参与训练。"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mnist", "fashionmnist", "cifar10")
SEEDS = (1234, 1235, 1236)


def compare(left, right, **identity):
    a = torch.load(left, map_location="cuda:0", weights_only=True)["state_dict"]
    b = torch.load(right, map_location="cuda:0", weights_only=True)["state_dict"]
    if list(a) != list(b) or any(a[key].shape != b[key].shape for key in a):
        raise ValueError("控制模型结构不同")
    first = torch.cat([value.flatten() for value in a.values()])
    second = torch.cat([value.flatten() for value in b.values()])
    error = first.double() - second.double()
    if not torch.isfinite(error).all():
        raise ValueError("控制模型包含非有限值")
    row = {**identity, "left": str(left.relative_to(ROOT)), "right": str(right.relative_to(ROOT)),
           "device": "cuda:0", "bitwise_equal": torch.equal(first, second),
           "max_absolute_difference": error.abs().max().item(),
           "relative_l2_difference": (error.norm() / second.double().norm().clamp_min(1e-30)).item()}
    del a, b, first, second, error
    gc.collect()
    return row


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("控制模型比较必须在CUDA上进行")
    torch.set_num_threads(4)
    old_info = json.loads((ROOT / "E5/results/previous_round_only_results.json").read_text())
    archived = old_info.get("archive_root", old_info.get("archive"))
    if archived is None:
        raise ValueError("旧E5归档元数据缺少路径")
    archived = ROOT / archived
    if not (archived / "E5/results").is_dir():
        raise ValueError("旧E5结果目录不存在")
    rows = {"E3": [], "E4": [], "E5": []}
    for dataset in DATASETS:
        for seed in SEEDS:
            suffix = f"{dataset}/ref128/seed{seed}/checkpoint.pt"
            rows["E3"].append(compare(ROOT / "E2/results/single_gpu" / suffix,
                                      ROOT / "E3/results/single_gpu" / suffix,
                                      kind="independent_E2_E3_single_batch256", dataset=dataset, seed=seed))
            rows["E4"].append(compare(ROOT / "E3/results/single_gpu" / suffix,
                                      ROOT / f"E4/results/single_gpu/{dataset}/ref1/seed{seed}/checkpoint.pt",
                                      kind="independent_E3_E4_single_batch256", dataset=dataset, seed=seed))
            for h in (1, 2, 4, 8, 16):
                current = ROOT / f"E5/results/{dataset}/h{h}/seed{seed}/checkpoint.pt"
                old = archived / f"E5/results/{dataset}/h{h}/seed{seed}/checkpoint.pt"
                serial = ROOT / f"E5/results/single_gpu/{dataset}/serial_h{h}/seed{seed}/checkpoint.pt"
                rows["E5"].append(compare(old, current, kind="E5_round_v1_vs_step_v2", dataset=dataset, seed=seed, level=h))
                rows["E5"].append(compare(serial, current, kind="E5_serial_vs_dual_same_algorithm", dataset=dataset, seed=seed, level=h))
    for experiment, checks in rows.items():
        result = {"experiment": experiment, "pairs": len(checks), "device": "cuda:0",
                  "bitwise_equal_pairs": sum(row["bitwise_equal"] for row in checks),
                  "max_absolute_difference": max(row["max_absolute_difference"] for row in checks),
                  "checks": checks}
        (ROOT / experiment / "results/gpu_control_comparisons.json").write_text(json.dumps(result, indent=2) + "\n")
        print(experiment, result["pairs"], "pairs; bitwise equal", result["bitwise_equal_pairs"], flush=True)


if __name__ == "__main__":
    run()
