"""E0 只使用 MLP-M；检查点兼容已有的第20轮模型。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from E0.config import Config, resolve_checkpoint
from E0.distributed import cuda_device


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def save_checkpoint(
    path: Path,
    model: MLP,
    config: Config,
    seed: int,
    identifier: str,
    metrics: dict[str, float],
) -> None:
    training = {**asdict(config), "data": config.dataset, "seed": seed, "model": "M"}
    payload = {
        "format_version": 1,
        "state_dict": {
            key: value.detach().clone() for key, value in model.state_dict().items()
        },
        "dataset": config.dataset,
        "model_config": {
            "preset": "M",
            "hidden": [1024, 512],
            "in_dim": model.net[0].in_features,
            "num_classes": 10,
        },
        "training_config": training,
        "run_id": identifier,
        "epoch": 20,
        "selection": "last",
        "metrics": metrics,
        "host_threads": torch.get_num_threads(),
    }
    shapes = {"mnist": [28, 28], "fashionmnist": [28, 28], "cifar10": [32, 32, 3]}
    classes = {
        "mnist": [str(i) for i in range(10)],
        "fashionmnist": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
        "cifar10": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    }
    payload["preprocessing"] = {
        "input_shape": shapes[config.dataset],
        "normalization": "divide_255",
        "channel_order": "HWC" if config.dataset == "cifar10" else "none",
    }
    payload["class_names"] = classes[config.dataset]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cuda:0") -> tuple[MLP, dict]:
    target = cuda_device(device)
    saved = torch.load(resolve_checkpoint(path), map_location=target, weights_only=True)
    if (
        saved.get("format_version") != 1
        or saved.get("epoch") != 20
        or saved.get("selection") != "last"
    ):
        raise ValueError("E0 需要第20轮的末轮模型检查点")
    shape = saved["model_config"]
    if shape.get("hidden") != [1024, 512] or shape.get("num_classes") != 10:
        raise ValueError("检查点结构必须为 E0 的 MLP-M")
    model = MLP(shape["in_dim"]).to(target)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved
