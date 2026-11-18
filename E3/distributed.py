"""E3 的 CUDA/NCCL 初始化和阻塞式梯度同步计时。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


def cuda_device(device: str = "cuda:0") -> torch.device:
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("模型训练和评估必须使用 CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA GPU")
    target = torch.device("cuda", 0 if target.index is None else target.index)
    if not 0 <= target.index < torch.cuda.device_count():
        raise ValueError(f"不可用的 GPU 序号：{target.index}")
    return target


def initialize(
    rank: int, world_size: int, port: int, host_threads: int = 4
) -> torch.device:
    if world_size not in (1, 2) or not 0 <= rank < world_size:
        raise ValueError("E3 只支持单 GPU 或双 GPU")
    device = cuda_device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(host_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world_size > 1:
        if not dist.is_nccl_available():
            raise RuntimeError("双 GPU 训练需要 NCCL")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120),
            device_id=device,
        )
    return device


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class SyncStatistics:
    count: int = 0
    nbytes: int = 0
    milliseconds: float = 0.0

    def reset(self) -> None:
        self.count, self.nbytes, self.milliseconds = 0, 0, 0.0


def register_hook(model, statistics: SyncStatistics, world_size: int) -> None:
    def hook(state, bucket):
        gradient = bucket.buffer()
        torch.cuda.synchronize(gradient.device)
        start = time.perf_counter()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(world_size)
        torch.cuda.synchronize(gradient.device)
        statistics.count += 1
        statistics.nbytes += gradient.numel() * gradient.element_size()
        statistics.milliseconds += (time.perf_counter() - start) * 1000
        future: torch.futures.Future = torch.futures.Future()
        future.set_result(gradient)
        return future

    model.register_comm_hook(None, hook)
