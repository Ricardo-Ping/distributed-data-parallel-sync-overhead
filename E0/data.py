"""E0 的分层验证划分和分布式采样；原始数据由 Data 提供。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from Data.datasets import DATA_ROOT, ImageData, load_dataset
from E0.config import Config


def stratified_indices(
    labels: torch.Tensor, fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回互斥、覆盖训练全集的训练/验证索引，不使用测试标签。"""
    if not 0 < fraction < 1:
        raise ValueError("验证比例须在 (0,1) 内")
    counts = torch.bincount(labels, minlength=10)
    n_val = round(len(labels) * fraction)
    if not 0 < n_val <= len(labels) - int((counts > 0).sum()):
        raise ValueError("划分后每个已有类别须至少保留一个训练样本")
    desired = counts.double() * fraction
    quotas = desired.floor().long().minimum((counts - 1).clamp_min(0))
    order = torch.argsort(desired - quotas, descending=True, stable=True)
    while int(quotas.sum()) < n_val:
        for label in order.tolist():
            if quotas[label] < counts[label] - 1:
                quotas[label] += 1
                if int(quotas.sum()) == n_val:
                    break
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label, count in enumerate(quotas.tolist()):
        ids = torch.where(labels == label)[0]
        selected.append(ids[torch.randperm(len(ids), generator=generator)[:count]])
    validation = torch.cat(selected).sort().values
    mask = torch.ones(len(labels), dtype=torch.bool)
    mask[validation] = False
    return torch.where(mask)[0], validation


@dataclass
class Split:
    raw: ImageData
    train_indices: torch.Tensor
    validation_indices: torch.Tensor

    def subset(self, name: str) -> TensorDataset:
        if name == "test":
            return TensorDataset(self.raw.x_test, self.raw.y_test)
        if name not in ("train", "validation"):
            raise ValueError(f"未知划分：{name}")
        ids = self.train_indices if name == "train" else self.validation_indices
        return TensorDataset(self.raw.x_train[ids], self.raw.y_train[ids])

    def save(self, folder: Path, config: Config) -> None:
        train, validation = self.train_indices.numpy(), self.validation_indices.numpy()
        metadata = {
            "data": self.raw.name,
            "in_dim": self.raw.in_dim,
            "n_classes": 10,
            "original_train_samples": len(self.raw.y_train),
            "train_samples": len(train),
            "val_samples": len(validation),
            "test_samples": len(self.raw.y_test),
            "validation_fraction": config.validation_fraction,
            "split_seed": config.split_seed,
            "train_class_counts": torch.bincount(
                self.raw.y_train[self.train_indices], minlength=10
            ).tolist(),
            "val_class_counts": torch.bincount(
                self.raw.y_train[self.validation_indices], minlength=10
            ).tolist(),
            "test_class_counts": torch.bincount(self.raw.y_test, minlength=10).tolist(),
            "train_indices_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
            "val_indices_sha256": hashlib.sha256(validation.tobytes()).hexdigest(),
        }
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "dataset.json"
        if path.exists() and json.loads(path.read_text()) != metadata:
            raise ValueError("结果目录中已存在其他数据划分")
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        np.savez_compressed(
            folder / "split_indices.npz", train=train, validation=validation
        )


def load_split(config: Config, root: Path = DATA_ROOT) -> Split:
    raw = load_dataset(config.dataset, root)
    return Split(
        raw,
        *stratified_indices(raw.y_train, config.validation_fraction, config.split_seed),
    )


def train_loader(
    split: Split, config: Config, rank: int, world_size: int, seed: int
) -> DataLoader:
    data = split.subset("train")
    if len(data) % world_size:
        raise ValueError("训练集大小须能被进程数整除，避免补齐重复样本")
    sampler = DistributedSampler(
        data, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )
    return DataLoader(
        data,
        batch_size=config.global_batch // world_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )
