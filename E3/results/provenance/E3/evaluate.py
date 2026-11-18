"""验证与测试共用的度量函数，以及保存模型的独立 GPU 评估入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Data.datasets import DATA_ROOT, load_dataset
from E3.data import stratified_indices
from E3.model import load_checkpoint


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, loader: DataLoader) -> dict[str, float | int | str]:
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("验证与测试必须在 GPU 上执行")
    was_training = model.training
    model.eval()
    total_loss, correct, samples = 0.0, 0, 0
    try:
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            samples += len(labels)
    finally:
        model.train(was_training)
    if samples == 0:
        raise ValueError("评估数据集不能为空")
    return {"samples": samples, "loss": total_loss / samples, "accuracy": correct / samples,
            "device": str(device)}


def evaluate_checkpoint(path: str | Path, *, split: str = "test", device: str = "cuda:0",
                         data_root: Path = DATA_ROOT, batch_size: int = 1024) -> dict:
    if split not in ("test", "validation") or batch_size < 1:
        raise ValueError("split 只支持 test/validation，batch_size 须为正数")
    model, saved = load_checkpoint(path, device)
    raw = load_dataset(saved["dataset"], data_root)
    if split == "validation":
        config = saved["training_config"]
        _, indices = stratified_indices(raw.y_train, config["validation_fraction"], config["split_seed"])
        features, labels = raw.x_train[indices], raw.y_train[indices]
    else:
        features, labels = raw.x_test, raw.y_test
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    metrics = evaluate_model(model, loader)
    return {**metrics, "dataset": raw.name, "split": split, "checkpoint_epoch": saved["epoch"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(evaluate_checkpoint(args.checkpoint, split=args.split, device=args.device,
                                       data_root=args.data_root, batch_size=args.batch_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
