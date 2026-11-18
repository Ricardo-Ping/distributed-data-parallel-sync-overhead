"""相同两层隐藏层架构，S/M/L 只按4倍递增隐藏宽度。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from E1.config import Config, PRESETS, resolve_checkpoint
from E1.distributed import cuda_device


class MLP(nn.Module):
    def __init__(self, in_dim: int, preset: str):
        super().__init__()
        first, second = PRESETS[preset]
        self.preset, self.in_dim = preset, in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def size_info(self) -> dict:
        first, second = PRESETS[self.preset]
        parameters = sum(parameter.numel() for parameter in self.parameters())
        weights = self.in_dim * first + first * second + second * 10
        return {
            "parameters": parameters,
            "gradient_bytes": parameters * 4,
            "forward_flops_per_sample": 2 * weights,
            "hidden1": first,
            "hidden2": second,
            "in_dim": self.in_dim,
        }


def save_checkpoint(
    path: Path, model: MLP, config: Config, seed: int, identifier: str, metrics: dict
) -> None:
    torch.save(
        {
            "format_version": 1,
            "experiment": "E1",
            "run_id": identifier,
            "epoch": 20,
            "selection": "last",
            "dataset": config.dataset,
            "preset": model.preset,
            "in_dim": model.in_dim,
            "training_config": asdict(config),
            "seed": seed,
            "metrics": metrics,
            "state_dict": {
                key: value.detach().clone() for key, value in model.state_dict().items()
            },
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cuda:0") -> tuple[MLP, dict]:
    target = cuda_device(device)
    saved = torch.load(resolve_checkpoint(path), map_location=target, weights_only=True)
    if (saved.get("experiment"), saved.get("epoch"), saved.get("selection")) != (
        "E1",
        20,
        "last",
    ):
        raise ValueError("需要 E1 第20轮末轮检查点")
    model = MLP(saved["in_dim"], saved["preset"]).to(target)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved
