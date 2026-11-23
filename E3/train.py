"""E3固定全局更新窗口，micro-batch按实际样本数加权，最后一次反向才同步。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from Data.datasets import DATA_ROOT
from E3.config import Config, EXPERIMENT, run_id
from E3.data import Split, load_split, train_loader
from E3.distributed import SyncStatistics, cleanup, initialize, register_hook
from E3.evaluate import evaluate_model
from E3.metrics import summarize_steps, write_rows
from E3.model import MLP, save_checkpoint
from training_common.epoch_metrics import EpochMetrics, format_epoch_metrics


class Trainer:
    def __init__(
        self,
        config: Config,
        batch: int,
        seed: int,
        rank: int,
        folder: Path,
        data_root: Path = DATA_ROOT,
        *,
        split: Split | None = None,
    ):
        self.config, self.batch, self.seed, self.rank, self.folder = (
            config,
            batch,
            seed,
            rank,
            folder,
        )
        self.ws, self.accumulation = config.world_size, config.accumulation_for(batch)
        self.effective_local = config.effective_local_batch(batch)
        self.device = torch.device("cuda", rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(seed)
        self.split = load_split(config, data_root) if split is None else split
        self.loader = train_loader(self.split, config, rank, self.ws, seed, batch)
        self.model = MLP(self.split.raw.in_dim).to(self.device)
        self.network = DistributedDataParallel(
            self.model,
            device_ids=[rank],
            output_device=rank,
            bucket_cap_mb=config.bucket_cap_mb,
        )
        self.synchronization = SyncStatistics()
        register_hook(self.network, self.synchronization, self.ws)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.identifier = run_id(config, batch, seed)
        folder.mkdir(parents=True, exist_ok=True)
        runtime = dict(
            run_id=self.identifier,
            experiment=EXPERIMENT,
            rank=rank,
            world_size=self.ws,
            device=str(self.device),
            backend=dist.get_backend(),
            gpu_name=torch.cuda.get_device_name(rank),
            host_threads=torch.get_num_threads(),
            dtype="float32",
            timing_mode="cuda_synchronized_update_wall",
            bucket_cap_mb=config.bucket_cap_mb,
            micro_batch=batch,
            accumulation=self.accumulation,
            global_batch=config.global_batch_for(batch),
            seed=seed,
            allow_tf32=False,
        )
        (folder / f"runtime_rank{rank}.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )
        if rank == 0:
            self.split.save(folder, config)
        self.peak_allocated, self.peak_reserved = 0, 0
        self.sampling_hashes = {}

    def backward_window(self, features, labels):
        data_ms, forward_backward_ms, sync_micros = 0.0, 0.0, 0
        size, losses = len(labels), []
        chunks = range(0, size, self.batch)
        for offset in chunks:
            end = min(offset + self.batch, size)
            start = time.perf_counter()
            micro_x, micro_y = (
                features[offset:end].to(self.device),
                labels[offset:end].to(self.device),
            )
            torch.cuda.synchronize(self.device)
            data_ms += (time.perf_counter() - start) * 1000
            last = end == size
            previous_count = self.synchronization.count
            start = time.perf_counter()
            # no_sync必须同时包住forward与backward；尾批也按真实窗口样本数归一化。
            with nullcontext() if last else self.network.no_sync():
                loss = self.criterion(self.network(micro_x), micro_y)
                (loss * ((end - offset) / size)).backward()
            torch.cuda.synchronize(self.device)
            forward_backward_ms += (time.perf_counter() - start) * 1000
            synchronized = self.synchronization.count > previous_count
            if synchronized and not last:
                raise RuntimeError("累积窗口尚未结束就同步了梯度")
            sync_micros += int(synchronized)
            losses.append((loss.detach(), end - offset))
        return data_ms, forward_backward_ms, len(chunks), sync_micros, losses

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.loader.sampler.set_epoch(epoch - 1)
        order = np.asarray(list(self.loader.sampler), dtype=np.int64)
        self.sampling_hashes[str(epoch)] = hashlib.sha256(order.tobytes()).hexdigest()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        rows, loss_sum, samples = [], 0.0, 0
        torch.cuda.synchronize(self.device)
        previous = time.perf_counter()
        for step, (features, labels) in enumerate(self.loader):
            window_start = previous
            fetch_ms = (time.perf_counter() - window_start) * 1000
            self.synchronization.reset()
            data_ms, forward_backward_ms, micro_count, sync_micros, losses = (
                self.backward_window(features, labels)
            )
            start = time.perf_counter()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.device)
            end = time.perf_counter()
            opt_ms, update_ms = (end - start) * 1000, (end - window_start) * 1000
            sync = self.synchronization
            compute_ms = forward_backward_ms - sync.milliseconds
            data_ms += fetch_ms
            other_ms = update_ms - data_ms - forward_backward_ms - opt_ms
            if min(compute_ms, other_ms) < -1e-7 or sync.count != 1 or sync_micros != 1:
                raise RuntimeError("分段计时无效或一次参数更新发生了额外梯度同步")
            if sync.nbytes != self.model.size_info()["gradient_bytes"]:
                raise RuntimeError("同步载荷不是完整梯度")
            size = len(labels)
            value = sum(loss.item() * n for loss, n in losses) / size
            iteration = (epoch - 1) * len(self.loader) + step
            full = size == self.effective_local
            rows.append(
                dict(
                    epoch=epoch,
                    step=step,
                    iteration=iteration,
                    rank=self.rank,
                    full_batch=full,
                    measured=full and iteration >= self.config.warmup_updates,
                    local_samples=size,
                    micro_count=micro_count,
                    sync_micro_count=sync_micros,
                    t_data_ms=data_ms,
                    t_compute_ms=max(0.0, compute_ms),
                    t_sync_ms=sync.milliseconds,
                    t_opt_ms=opt_ms,
                    t_other_ms=max(0.0, other_ms),
                    t_update_ms=update_ms,
                    ar_count=sync.count,
                    ar_bytes=sync.nbytes,
                    loss=value,
                )
            )
            loss_sum += value * size
            samples += size
            previous = time.perf_counter()
        self.peak_allocated = max(
            self.peak_allocated, torch.cuda.max_memory_allocated(self.device)
        )
        self.peak_reserved = max(
            self.peak_reserved, torch.cuda.max_memory_reserved(self.device)
        )
        write_rows(self.folder / f"steps_rank{self.rank}.csv", rows, append=epoch > 1)
        (self.folder / f"sampling_rank{self.rank}.json").write_text(
            json.dumps(self.sampling_hashes, indent=2) + "\n"
        )
        totals = torch.tensor(
            [loss_sum, samples], dtype=torch.float64, device=self.device
        )
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
            time_column="t_update_ms",
            samples_column="local_samples",
        )
        started = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            if self.rank == 0:
                measured = evaluate_model(self.model, validation)
                if not math.isfinite(measured["loss"]) or not math.isfinite(train_loss):
                    raise RuntimeError("训练或验证损失不是有限值")
                if measured["loss"] < best_loss:
                    best_loss, best_epoch = measured["loss"], epoch
                performance = epoch_metrics.read(epoch)
                history.append(
                    dict(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=measured["loss"],
                        val_acc=measured["accuracy"],
                        **performance,
                    )
                )
                pd.DataFrame(history).to_csv(self.folder / "epochs.csv", index=False)
                print(
                    f"[epoch] {self.identifier} {epoch}/20 train={train_loss:.5f} "
                    f"val={measured['loss']:.5f} acc={measured['accuracy']:.4%}\n"
                    f"        {format_epoch_metrics(performance)}",
                    flush=True,
                )
            dist.barrier()
        (self.folder / f"memory_rank{self.rank}.json").write_text(
            json.dumps(
                dict(
                    allocated_bytes=self.peak_allocated,
                    reserved_bytes=self.peak_reserved,
                ),
                indent=2,
            )
            + "\n"
        )
        dist.barrier()
        final = None
        if self.rank == 0:
            test = DataLoader(
                self.split.subset("test"),
                batch_size=self.config.eval_batch,
                shuffle=False,
                num_workers=0,
            )
            tested = evaluate_model(self.model, test)
            size = self.model.size_info()
            frame = pd.concat(
                [
                    pd.read_csv(self.folder / f"steps_rank{rank}.csv")
                    for rank in range(self.ws)
                ],
                ignore_index=True,
            )
            performance = summarize_steps(
                frame,
                self.ws,
                self.batch,
                self.accumulation,
                size["gradient_bytes"],
                self.config.warmup_updates,
            )
            peaks = [
                json.loads((self.folder / f"memory_rank{rank}.json").read_text())
                for rank in range(self.ws)
            ]
            final = dict(
                experiment=EXPERIMENT,
                run_id=self.identifier,
                dataset=self.config.dataset,
                model="M",
                micro_batch=self.batch,
                accumulation=self.accumulation,
                seed=self.seed,
                ws=self.ws,
                global_batch=self.config.global_batch_for(self.batch),
                epochs_ran=20,
                test_epoch=20,
                selection="last",
                learning_rate=self.config.learning_rate,
                warmup_updates=self.config.warmup_updates,
                steps_per_epoch=len(self.loader),
                train_samples=len(self.split.train_indices),
                val_samples=len(self.split.validation_indices),
                test_samples=tested["samples"],
                best_val_epoch=best_epoch,
                train_loss=train_loss,
                val_loss=measured["loss"],
                val_acc=measured["accuracy"],
                test_loss=tested["loss"],
                test_acc=tested["accuracy"],
                peak_allocated_mb=max(p["allocated_bytes"] for p in peaks) / 1e6,
                peak_reserved_mb=max(p["reserved_bytes"] for p in peaks) / 1e6,
                wall_time_s=time.perf_counter() - started,
                **size,
                **performance,
            )
            save_checkpoint(
                self.folder / "checkpoint.pt",
                self.model,
                self.config,
                self.batch,
                self.seed,
                self.identifier,
                {
                    key: final[key]
                    for key in ("val_loss", "val_acc", "test_loss", "test_acc")
                },
            )
            final["checkpoint_path"] = str((self.folder / "checkpoint.pt").resolve())
            (self.folder / "final.json").write_text(json.dumps(final, indent=2) + "\n")
            print(
                f"[final] {self.identifier} update={final['avg_update_ms']:.3f}ms throughput={final['train_throughput']:.1f} sync={final['sync_share']:.2%} test={final['test_acc']:.4%}",
                flush=True,
            )
        dist.barrier()
        return final


def train_worker(
    rank: int,
    port: int,
    config: Config,
    batch: int,
    seed: int,
    folder: Path,
    data_root: Path,
) -> None:
    initialize(rank, config.world_size, port, config.host_threads)
    try:
        Trainer(config, batch, seed, rank, folder, data_root).train()
    finally:
        cleanup()
