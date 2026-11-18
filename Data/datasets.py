"""读取官方图像数据；划分和采样策略由各实验自行决定。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST

DATA_ROOT = Path(__file__).resolve().parent
DATASETS = ("mnist", "fashionmnist", "cifar10")


@dataclass
class ImageData:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def in_dim(self) -> int:
        return self.x_train.shape[1]


def load_dataset(
    name: str, root: Path = DATA_ROOT, *, download: bool = False
) -> ImageData:
    """像素转 float32 / 255 并展平；彩色图像保持 HWC 顺序。"""
    classes = {"mnist": MNIST, "fashionmnist": FashionMNIST, "cifar10": CIFAR10}
    if name not in classes:
        raise ValueError(f"未知数据集 {name!r}，可选 {DATASETS}")
    cls = classes[name]
    train = cls(str(root), train=True, download=download)
    test = cls(str(root), train=False, download=download)
    dimension = 3072 if name == "cifar10" else 784

    def features(data) -> torch.Tensor:
        return torch.as_tensor(data).float().div(255).reshape(-1, dimension)

    return ImageData(
        name,
        features(train.data),
        torch.as_tensor(train.targets, dtype=torch.long),
        features(test.data),
        torch.as_tensor(test.targets, dtype=torch.long),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    for name in DATASETS if args.dataset == "all" else (args.dataset,):
        data = load_dataset(name, download=args.download)
        print(
            f"{name}: train={len(data.y_train)}, test={len(data.y_test)}, in_dim={data.in_dim}"
        )
