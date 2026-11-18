"""E5 的 CUDA/NCCL 初始化和阻塞式梯度同步计时。"""

from __future__ import annotations

import time
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
        raise ValueError("E5 只支持单 GPU 或双 GPU")
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


class ParameterAverager:
    """只平均参数；不触碰各rank的SGD momentum。"""

    def __init__(self, model):
        self.parameters = list(model.parameters())
        self.device = self.parameters[0].device
        self.buffer = torch.empty(
            sum(p.numel() for p in self.parameters),
            device=self.device,
            dtype=torch.float32,
        )
        self.payload_bytes = self.buffer.numel() * self.buffer.element_size()

    @torch.no_grad()
    def pack(self):
        offset = 0
        for p in self.parameters:
            self.buffer[offset : offset + p.numel()].copy_(p.reshape(-1))
            offset += p.numel()

    @torch.no_grad()
    def unpack(self):
        offset = 0
        for p in self.parameters:
            p.copy_(self.buffer[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    @torch.no_grad()
    def initialize_parameters(self):
        self.pack()
        dist.broadcast(self.buffer, src=0)
        self.unpack()
        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def average(self, *, diagnose=False):
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.pack()
        torch.cuda.synchronize(self.device)
        arrival = time.perf_counter()
        dist.all_reduce(self.buffer)
        self.buffer.div_(2)
        torch.cuda.synchronize(self.device)
        reduced = time.perf_counter()
        drift = None
        diagnostic_ms = 0.0
        if diagnose:
            before = time.perf_counter()
            squared = 0.0
            norm = 0.0
            offset = 0
            for p in self.parameters:
                mean = self.buffer[offset : offset + p.numel()].view_as(p).double()
                squared += (p.double() - mean).square().sum().item()
                norm += mean.square().sum().item()
                offset += p.numel()
            drift = 2 * (squared / norm) ** 0.5
            torch.cuda.synchronize(self.device)
            diagnostic_ms = (time.perf_counter() - before) * 1000
        copy_start = time.perf_counter()
        self.unpack()
        torch.cuda.synchronize(self.device)
        end = time.perf_counter()
        return dict(
            t_sync_ms=(end - start) * 1000 - diagnostic_ms,
            t_pack_ms=(arrival - start) * 1000,
            t_collective_ms=(reduced - arrival) * 1000,
            t_copy_ms=(end - copy_start) * 1000,
            sync_arrival_s=arrival,
            diagnostic_ms=diagnostic_ms,
            drift_relative_l2=drift,
            ar_count=1,
            ar_bytes=self.payload_bytes,
        )
