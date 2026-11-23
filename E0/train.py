"""E0 单次训练：20轮训练与验证，然后测试并保存末轮模型。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from Data.datasets import DATA_ROOT
from E0.config import EXPERIMENTS, PROJECT_ROOT, Config, run_id
from E0.data import Split, load_split, train_loader
from E0.distributed import SyncStatistics, cleanup, initialize, register_hook
from E0.evaluate import evaluate_model
from E0.metrics import (
    CSVWriter,
    FINAL_FIELDS,
    RUN_FIELDS,
    VALIDATION_EPOCH_FIELDS,
    summarize_timing,
)
from E0.model import MLP, save_checkpoint


class Trainer:
    """只实现 E0 协议，不包含累积、压缩、早停等其他实验逻辑。"""

    def __init__(
        self,
        config: Config,
        seed: int,
        rank: int,
        world_size: int,
        output: Path,
        data_root: Path = DATA_ROOT,
        *,
        split: Split | None = None,
    ):
        self.config, self.seed = config, seed
        self.rank, self.world_size = rank, world_size
        self.device = torch.device("cuda", rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(seed)
        self.split = load_split(config, data_root) if split is None else split
        self.loader = train_loader(self.split, config, rank, world_size, seed)
        self.model = MLP(self.split.raw.in_dim).to(self.device)
        self.network = (
            DistributedDataParallel(self.model, device_ids=[rank], output_device=rank)
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
        self.identifier = run_id(config, seed, world_size)
        self.folder = output / config.dataset
        self.common = {
            "run_id": self.identifier,
            "exp": EXPERIMENTS[config.dataset],
            "model": "M",
            "ws": world_size,
            "k": 1,
            "local_batch": config.global_batch // world_size,
            "compress": "none",
            "lr": config.learning_rate,
            "scale_lr": False,
            "data": config.dataset,
            "device": str(self.device),
            "backend": dist.get_backend() if world_size > 1 else "none",
            "gpu_name": torch.cuda.get_device_name(self.device),
            "host_threads": torch.get_num_threads(),
            "timing_mode": "cuda_synchronized_wall",
        }
        self.folder.mkdir(parents=True, exist_ok=True)
        runtime = self.folder / "runtime"
        runtime.mkdir(exist_ok=True)
        (runtime / f"{self.identifier}_rank{rank}.json").write_text(
            json.dumps(
                {
                    **self.common,
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": world_size,
                    "seed": seed,
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                },
                indent=2,
            )
            + "\n"
        )
        self.totals = {"samples": 0, "updates": 0, "train_ms": 0.0, "sync_ms": 0.0}
        if rank == 0:
            self.split.save(self.folder, config)

    def train_epoch(self, epoch: int, writer: CSVWriter | None) -> float:
        self.model.train()
        self.loader.sampler.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, sample_count = 0.0, 0
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
            optimizer_ms = (time.perf_counter() - start) * 1000
            synchronization = self.synchronization
            compute_ms = max(0.0, forward_backward_ms - synchronization.milliseconds)
            samples, batch_loss = len(labels), loss.item()
            if self.rank == 0:
                row = {
                    **self.common,
                    "epoch": epoch,
                    "step": step,
                    "in_warmup": step < self.config.warmup_steps,
                    "t_data_ms": round(data_ms, 3),
                    "t_fwd_bwd_ms": round(compute_ms, 3),
                    "t_sync_ms": round(synchronization.milliseconds, 3),
                    "t_opt_ms": round(optimizer_ms, 3),
                    "t_step_ms": round(data_ms + forward_backward_ms + optimizer_ms, 3),
                    "samples": samples,
                    "ar_count": synchronization.count,
                    "ar_bytes": synchronization.nbytes,
                    "loss": round(batch_loss, 5),
                }
                writer.write(row)
                self.totals["samples"] += samples * self.world_size
                self.totals["updates"] += 1
                self.totals["train_ms"] += row["t_step_ms"]
                self.totals["sync_ms"] += row["t_sync_ms"]
            loss_sum += batch_loss * samples
            sample_count += samples
            previous = time.perf_counter()
        totals = torch.tensor(
            [loss_sum, sample_count], dtype=torch.float64, device=self.device
        )
        if self.world_size > 1:
            dist.all_reduce(totals)
        return (totals[0] / totals[1]).item()

    def train(self) -> dict | None:
        step_writer = epoch_writer = None
        if self.rank == 0:
            step_writer = CSVWriter(self.folder / "runs.csv", RUN_FIELDS)
            epoch_writer = CSVWriter(
                self.folder / "epochs.csv", VALIDATION_EPOCH_FIELDS
            )
        validation = DataLoader(
            self.split.subset("validation"),
            batch_size=self.config.eval_batch,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        started = time.perf_counter()
        best_loss, best_accuracy, best_epoch = float("inf"), 0.0, 0
        final = None
        try:
            for epoch in range(self.config.epochs):
                previous_totals = self.totals.copy()
                start = time.perf_counter()
                train_loss = self.train_epoch(epoch, step_writer)
                if self.rank == 0:
                    measured = evaluate_model(self.model, validation)
                    if not math.isfinite(measured["loss"]):
                        raise RuntimeError("验证损失不是有限值")
                    is_best = measured["loss"] < best_loss
                    if is_best:
                        best_loss, best_accuracy, best_epoch = (
                            measured["loss"],
                            measured["accuracy"],
                            epoch + 1,
                        )
                    epoch_writer.write(
                        {
                            **self.common,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "test_acc": "",
                            "wall_s": time.perf_counter() - start,
                            "val_loss": measured["loss"],
                            "val_acc": measured["accuracy"],
                            "is_best": is_best,
                        }
                    )
                    timing = summarize_timing(
                        **{
                            field: total - previous_totals[field]
                            for field, total in self.totals.items()
                        }
                    )
                    print(
                        f"[epoch] {self.identifier} {epoch + 1}/20 loss={train_loss:.5f} "
                        f"val_loss={measured['loss']:.5f} val_acc={measured['accuracy']:.4%}\n"
                        f"        throughput={timing['train_throughput']:.1f} samples/s  "
                        f"avg_update={timing['avg_update_ms']:.3f} ms  "
                        f"avg_sync={timing['avg_sync_ms']:.3f} ms  "
                        f"sync_share={timing['sync_share']:.2%}",
                        flush=True,
                    )
                if self.world_size > 1:
                    # 等待验证结束，避免下一轮的同步计时混入验证耗时。
                    dist.barrier()
            if self.rank == 0:
                # 官方测试集仅在20轮训练和验证全部结束后进入评估。
                test_loader = DataLoader(
                    self.split.subset("test"),
                    batch_size=self.config.eval_batch,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                )
                tested = evaluate_model(self.model, test_loader)
                checkpoint = (
                    self.folder / "checkpoints" / f"{self.identifier}_epoch20.pt"
                )
                save_checkpoint(
                    checkpoint,
                    self.model,
                    self.config,
                    self.seed,
                    self.identifier,
                    {
                        "val_loss": measured["loss"],
                        "val_acc": measured["accuracy"],
                        "test_loss": tested["loss"],
                        "test_acc": tested["accuracy"],
                    },
                )
                total = self.totals
                try:
                    checkpoint_path = str(
                        checkpoint.resolve().relative_to(PROJECT_ROOT)
                    )
                except ValueError:
                    checkpoint_path = str(checkpoint.resolve())
                final = {
                    **self.common,
                    "seed": self.seed,
                    "split_seed": self.config.split_seed,
                    "train_samples": len(self.split.train_indices),
                    "val_samples": len(self.split.validation_indices),
                    "test_samples": len(self.split.raw.y_test),
                    "epochs_ran": 20,
                    "test_epoch": 20,
                    "training_mode": "fixed_epochs",
                    "stopped_early": False,
                    "train_loss": train_loss,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                    "best_val_acc": best_accuracy,
                    "last_val_loss": measured["loss"],
                    "last_val_acc": measured["accuracy"],
                    "test_loss": tested["loss"],
                    "test_acc": tested["accuracy"],
                    "checkpoint_path": checkpoint_path,
                    "total_train_samples": total["samples"],
                    "total_updates": total["updates"],
                    **summarize_timing(**total),
                    "train_time_s": total["train_ms"] / 1000,
                    "wall_time_s": time.perf_counter() - started,
                }
                final_writer = CSVWriter(self.folder / "final.csv", FINAL_FIELDS)
                try:
                    final_writer.write(final)
                finally:
                    final_writer.close()
                print(
                    f"[final] {self.identifier} test_acc={tested['accuracy']:.4%}",
                    flush=True,
                )
            if self.world_size > 1:
                dist.barrier()
        finally:
            if step_writer is not None:
                step_writer.close()
                epoch_writer.close()
        return final


def train_worker(
    rank: int,
    world_size: int,
    port: int,
    config: Config,
    seed: int,
    output: Path,
    data_root: Path,
) -> None:
    initialize(rank, world_size, port)
    try:
        Trainer(config, seed, rank, world_size, output, data_root).train()
    finally:
        cleanup()
