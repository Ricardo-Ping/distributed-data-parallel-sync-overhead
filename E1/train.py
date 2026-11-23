"""E1 固定每卡batch，记录每个rank；20轮验证后只测试末轮模型。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from Data.datasets import DATA_ROOT
from E1.config import Config, run_id
from E1.data import Split, load_split, train_loader
from E1.distributed import SyncStatistics, cleanup, initialize, register_hook
from E1.evaluate import evaluate_model
from E1.metrics import STEP_FIELDS, summarize_steps, write_rows
from E1.model import MLP, save_checkpoint
from training_common.epoch_metrics import EpochMetrics, format_epoch_metrics


class Trainer:
    def __init__(
        self,
        config: Config,
        preset: str,
        seed: int,
        rank: int,
        world_size: int,
        folder: Path,
        data_root: Path = DATA_ROOT,
        *,
        split: Split | None = None,
    ):
        self.config, self.preset, self.seed = config, preset, seed
        self.rank, self.ws, self.folder = rank, world_size, folder
        self.device = torch.device("cuda", rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(seed)
        self.split = load_split(config, data_root) if split is None else split
        self.loader = train_loader(self.split, config, rank, world_size, seed)
        if (
            len(self.loader) == 0
            or len(self.loader) * config.epochs <= config.warmup_steps
        ):
            raise ValueError("完整训练批次不足以完成预热与计时")
        self.model = MLP(self.split.raw.in_dim, preset).to(self.device)
        self.network = (
            DistributedDataParallel(
                self.model,
                device_ids=[rank],
                output_device=rank,
                bucket_cap_mb=config.bucket_cap_mb,
                broadcast_buffers=False,
            )
            if world_size > 1
            else self.model
        )
        self.synchronization = SyncStatistics()
        if world_size > 1:
            register_hook(self.network, self.synchronization, world_size)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.identifier = run_id(config.dataset, preset, world_size, seed)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.runtime = {
            "run_id": self.identifier,
            "rank": rank,
            "world_size": world_size,
            "device": str(self.device),
            "backend": "nccl" if world_size > 1 else "none",
            "gpu_name": torch.cuda.get_device_name(rank),
            "host_threads": torch.get_num_threads(),
            "timing_mode": "cuda_synchronized_wall",
            "bucket_cap_mb": config.bucket_cap_mb,
            "seed": seed,
            "dtype": "float32",
            "allow_tf32": False,
        }
        (folder / f"runtime_rank{rank}.json").write_text(
            json.dumps(self.runtime, indent=2) + "\n"
        )
        if rank == 0:
            self.split.save(folder, config)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_allocated, self.peak_reserved = 0, 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.loader.sampler.set_epoch(epoch - 1)
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        rows, loss_sum, samples = [], 0.0, 0
        torch.cuda.synchronize(self.device)
        previous = time.perf_counter()
        for step, (features, labels) in enumerate(self.loader):
            features, labels = features.to(self.device), labels.to(self.device)
            torch.cuda.synchronize(self.device)
            data_ms = (time.perf_counter() - previous) * 1000
            self.synchronization.reset()
            start = time.perf_counter()
            loss = self.criterion(self.network(features), labels)
            loss.backward()
            torch.cuda.synchronize(self.device)
            forward_backward_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.device)
            opt_ms = (time.perf_counter() - start) * 1000
            sync = self.synchronization
            compute_ms = forward_backward_ms - sync.milliseconds
            if compute_ms < 0 or sync.count != int(self.ws > 1):
                raise RuntimeError(
                    "分段计时无效或梯度不在一个桶内，不能作为模型规模对照"
                )
            if self.ws > 1 and sync.nbytes != self.model.size_info()["gradient_bytes"]:
                raise RuntimeError("未同步完整梯度")
            value = loss.item()
            iteration = (epoch - 1) * len(self.loader) + step
            rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "iteration": iteration,
                    "rank": self.rank,
                    "measured": iteration >= self.config.warmup_steps,
                    "samples": len(labels),
                    "t_data_ms": data_ms,
                    "t_compute_ms": compute_ms,
                    "t_sync_ms": sync.milliseconds,
                    "t_opt_ms": opt_ms,
                    "t_step_ms": data_ms + forward_backward_ms + opt_ms,
                    "ar_count": sync.count,
                    "ar_bytes": sync.nbytes,
                    "loss": value,
                }
            )
            loss_sum += value * len(labels)
            samples += len(labels)
            previous = time.perf_counter()
        self.peak_allocated = max(
            self.peak_allocated, torch.cuda.max_memory_allocated(self.device)
        )
        self.peak_reserved = max(
            self.peak_reserved, torch.cuda.max_memory_reserved(self.device)
        )
        # 统计汇总、文件写入和验证都放在训练计时段外。
        write_rows(
            self.folder / f"steps_rank{self.rank}.csv",
            rows,
            STEP_FIELDS,
            append=epoch > 1,
        )
        totals = torch.tensor(
            [loss_sum, samples], dtype=torch.float64, device=self.device
        )
        if self.ws > 1:
            dist.all_reduce(totals)
        return (totals[0] / totals[1]).item()

    def train(self) -> dict | None:
        validation = DataLoader(
            self.split.subset("validation"),
            batch_size=self.config.eval_batch,
            shuffle=False,
            num_workers=0,
        )
        history, best_loss, best_epoch = [], float("inf"), 0
        epoch_metrics = EpochMetrics(
            (self.folder / f"steps_rank{rank}.csv" for rank in range(self.ws)),
            time_column="t_step_ms",
            samples_column="samples",
        )
        started = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            if self.rank == 0:
                measured = evaluate_model(self.model, validation)
                if measured["loss"] < best_loss:
                    best_loss, best_epoch = measured["loss"], epoch
                performance = epoch_metrics.read(epoch)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": measured["loss"],
                        "val_acc": measured["accuracy"],
                        **performance,
                    }
                )
                pd.DataFrame(history).to_csv(self.folder / "epochs.csv", index=False)
                print(
                    f"[epoch] {self.identifier} {epoch}/20 train={train_loss:.5f} "
                    f"val={measured['loss']:.5f} acc={measured['accuracy']:.4%}\n"
                    f"        {format_epoch_metrics(performance, scale_experiment=True)}",
                    flush=True,
                )
            if self.ws > 1:
                dist.barrier()
        peak = {
            "allocated_bytes": self.peak_allocated,
            "reserved_bytes": self.peak_reserved,
        }
        (self.folder / f"memory_rank{self.rank}.json").write_text(
            json.dumps(peak, indent=2) + "\n"
        )
        if self.ws > 1:
            dist.barrier()
        result = None
        if self.rank == 0:
            test_loader = DataLoader(
                self.split.subset("test"),
                batch_size=self.config.eval_batch,
                shuffle=False,
                num_workers=0,
            )
            tested = evaluate_model(self.model, test_loader)
            if not all(math.isfinite(row["train_loss"]) for row in history):
                raise RuntimeError("训练损失不是有限值")
            size = self.model.size_info()
            frame = pd.concat(
                [
                    pd.read_csv(self.folder / f"steps_rank{rank}.csv")
                    for rank in range(self.ws)
                ],
                ignore_index=True,
            )
            performance = summarize_steps(
                frame, self.ws, self.config.local_batch, size["gradient_bytes"]
            )
            peaks = [
                json.loads((self.folder / f"memory_rank{rank}.json").read_text())
                for rank in range(self.ws)
            ]
            result = {
                "run_id": self.identifier,
                "dataset": self.config.dataset,
                "model": self.preset,
                "seed": self.seed,
                "ws": self.ws,
                "local_batch": self.config.local_batch,
                "global_batch": self.config.local_batch * self.ws,
                "epochs_ran": self.config.epochs,
                "test_epoch": self.config.epochs,
                "best_val_epoch": best_epoch,
                "selection": "last",
                "train_samples": len(self.split.train_indices),
                "val_samples": len(self.split.validation_indices),
                "test_samples": tested["samples"],
                "steps_per_epoch": len(self.loader),
                "dropped_per_epoch": len(self.split.train_indices)
                - len(self.loader) * self.config.local_batch * self.ws,
                "learning_rate": self.config.learning_rate,
                "warmup_steps": self.config.warmup_steps,
                "bucket_cap_mb": self.config.bucket_cap_mb,
                "host_threads": self.config.host_threads,
                "train_loss": train_loss,
                "val_loss": measured["loss"],
                "val_acc": measured["accuracy"],
                "test_loss": tested["loss"],
                "test_acc": tested["accuracy"],
                "peak_allocated_mb": max(p["allocated_bytes"] for p in peaks) / 1e6,
                "peak_reserved_mb": max(p["reserved_bytes"] for p in peaks) / 1e6,
                "wall_time_s": time.perf_counter() - started,
                **size,
                **performance,
            }
            save_checkpoint(
                self.folder / "checkpoint.pt",
                self.model,
                self.config,
                self.seed,
                self.identifier,
                {
                    key: result[key]
                    for key in ("val_loss", "val_acc", "test_loss", "test_acc")
                },
            )
            result["checkpoint_path"] = str((self.folder / "checkpoint.pt").resolve())
            (self.folder / "final.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"[final] {self.identifier} compute={result['avg_compute_ms']:.3f}ms "
                f"sync={result['avg_sync_ms']:.3f}ms sync_share={result['sync_share']:.2%} "
                f"throughput={result['train_throughput']:.1f} test={result['test_acc']:.4%}",
                flush=True,
            )
        if self.ws > 1:
            dist.barrier()
        return result


def train_worker(
    rank: int,
    world_size: int,
    port: int,
    config: Config,
    preset: str,
    seed: int,
    folder: Path,
    data_root: Path,
) -> None:
    initialize(rank, world_size, port, config.host_threads)
    try:
        Trainer(config, preset, seed, rank, world_size, folder, data_root).train()
    finally:
        cleanup()
