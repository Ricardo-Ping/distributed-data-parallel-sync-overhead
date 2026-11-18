"""在同一 CUDA 设备平均两个模型；不调用 NCCL，不修改本地 momentum。"""

from __future__ import annotations

import time

import torch


class LocalAverager:
    def __init__(self, models):
        if len(models) != 2:
            raise ValueError("需要两个逻辑副本")
        self.parameters = [list(model.parameters()) for model in models]
        self.device = self.parameters[0][0].device
        if self.device.type != "cuda" or any(
            p.device != self.device for group in self.parameters for p in group
        ):
            raise ValueError("两个逻辑副本必须在同一个 CUDA GPU")
        count = sum(p.numel() for p in self.parameters[0])
        self.buffers = [
            torch.empty(count, device=self.device, dtype=torch.float32) for _ in (0, 1)
        ]

    @torch.no_grad()
    def average(self, diagnose=False):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for parameters, buffer in zip(self.parameters, self.buffers):
            offset = 0
            for parameter in parameters:
                buffer[offset : offset + parameter.numel()].copy_(parameter.reshape(-1))
                offset += parameter.numel()
        mean = self.buffers[0]
        mean.add_(self.buffers[1]).div_(2)
        torch.cuda.synchronize(self.device)
        diagnostic_ms, drift = 0.0, (None, None)
        if diagnose:
            before = time.perf_counter()
            observed = []
            for parameters in self.parameters:
                offset, squared, norm = 0, 0.0, 0.0
                for parameter in parameters:
                    target = (
                        mean[offset : offset + parameter.numel()]
                        .view_as(parameter)
                        .double()
                    )
                    squared += (parameter.double() - target).square().sum().item()
                    norm += target.square().sum().item()
                    offset += parameter.numel()
                observed.append(2 * (squared / norm) ** 0.5)
            torch.cuda.synchronize(self.device)
            diagnostic_ms = (time.perf_counter() - before) * 1000
            drift = tuple(observed)
        for parameters in self.parameters:
            offset = 0
            for parameter in parameters:
                parameter.copy_(
                    mean[offset : offset + parameter.numel()].view_as(parameter)
                )
                offset += parameter.numel()
        torch.cuda.synchronize(self.device)
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "local_average_ms": total_ms - diagnostic_ms,
            "diagnostic_ms": diagnostic_ms,
            "drift": drift,
            "cross_gpu_allreduces": 0,
        }
