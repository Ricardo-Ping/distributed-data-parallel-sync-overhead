"""E5：各GPU执行H次独立SGD更新，再平均参数；momentum始终保持本地。"""

from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from Data.datasets import DATA_ROOT
from E5.config import EXPERIMENT, run_id
from E5.data import load_split, train_loader
from E5.distributed import initialize, cleanup, ParameterAverager
from E5.evaluate import evaluate_model
from E5.metrics import summarize_rounds, write_rows, write_local_steps
from E5.model import MLP, save_checkpoint


class Trainer:
    def __init__(
        self,
        config,
        level,
        variant,
        seed,
        rank,
        folder,
        data_root=DATA_ROOT,
        *,
        split=None,
    ):
        self.config, self.level, self.variant, self.seed, self.rank = (
            config,
            level,
            variant,
            seed,
            rank,
        )
        if variant != "local_sgd":
            raise ValueError("E5仅运行Local SGD")
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda", rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(seed)
        self.split = load_split(config, data_root) if split is None else split
        self.loader = train_loader(self.split, config, rank, 2, seed, level)
        self.model = MLP(self.split.raw.in_dim).to(self.device)
        self.averager = ParameterAverager(self.model)
        self.averager.initialize_parameters()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        self.identifier = run_id(config, level, variant, seed)
        self.sampling_hashes = {}
        self.peak_allocated = 0
        self.round_iteration = 0
        self.last_drift = None
        runtime = dict(
            experiment=EXPERIMENT,
            run_id=self.identifier,
            device=str(self.device),
            backend=dist.get_backend(),
            rank=rank,
            world_size=2,
            host_threads=torch.get_num_threads(),
            dtype="float32",
            allow_tf32=False,
            level=level,
            variant=variant,
            micro_batch=config.micro_batch,
            global_batch=config.global_batch_for(level),
            timing_mode="cuda_synchronized_local_sgd_round",
            momentum_policy="local_unsynchronized",
            epoch_end_policy="flush_partial_round",
            initial_parameter_broadcasts=1,
            timing_schema_version=2,
            local_step_logging="actual_each_local_optimizer_update; parameter sync charged to boundary step",
        )
        (self.folder / f"runtime_rank{rank}.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )
        if rank == 0:
            self.split.save(self.folder, config)

    def local_update(self, features, labels):
        start = time.perf_counter()
        x, y = features.to(self.device), labels.to(self.device)
        torch.cuda.synchronize(self.device)
        data_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        loss = torch.nn.functional.cross_entropy(self.model(x), y)
        loss.backward()
        torch.cuda.synchronize(self.device)
        compute_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(self.device)
        opt_ms = (time.perf_counter() - start) * 1000
        return data_ms, compute_ms, opt_ms, loss.detach()

    def train_epoch(self, epoch):
        self.model.train()
        self.loader.sampler.set_epoch(epoch - 1)
        ids = np.asarray(list(self.loader.sampler), dtype=np.int64)
        self.sampling_hashes[str(epoch)] = hashlib.sha256(ids.tobytes()).hexdigest()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        full_steps = len(self.loader.dataset) // (2 * self.config.micro_batch)
        # loader.dataset在各rank均包含全局训练集，因此完整局部批步数需除以2。
        diagnostic_end = (full_steps // self.level) * self.level
        iterator = iter(self.loader)
        rows = []
        local_rows = []
        loss_sum = 0.0
        samples = 0
        self.last_drift = None
        for round_index, first in enumerate(range(0, len(self.loader), self.level)):
            count = min(self.level, len(self.loader) - first)
            torch.cuda.synchronize(self.device)
            begin = time.perf_counter()
            data_ms = compute_ms = opt_ms = 0.0
            losses = []
            local_samples = 0
            boundary = begin
            round_steps = []
            for offset in range(count):
                before = time.perf_counter()
                features, labels = next(iterator)
                fetch_ms = (time.perf_counter() - before) * 1000
                d, c, o, loss = self.local_update(features, labels)
                ended = time.perf_counter()
                step_ms = (ended - boundary) * 1000
                step_data = fetch_ms + d
                step_index = first + offset
                step_global = (epoch - 1) * len(self.loader) + step_index
                round_steps.append(
                    dict(
                        epoch=epoch,
                        step=step_index,
                        iteration=step_global,
                        round_iteration=self.round_iteration,
                        rank=self.rank,
                        measured=len(labels) == self.config.micro_batch
                        and step_global >= self.config.warmup_updates,
                        local_samples=len(labels),
                        t_data_ms=step_data,
                        t_compute_ms=c,
                        t_sync_ms=0.0,
                        t_opt_ms=o,
                        t_other_ms=max(0.0, step_ms - step_data - c - o),
                        t_update_ms=step_ms,
                        local_throughput=0.0,
                        sync_happened=False,
                        sync_type="none",
                        ar_count=0,
                        ar_bytes=0,
                        loss=loss,
                    )
                )
                boundary = ended
                data_ms += step_data
                compute_ms += c
                opt_ms += o
                local_samples += len(labels)
                losses.append((loss, len(labels)))
            diagnose = diagnostic_end > 0 and first + count == diagnostic_end
            sync = self.averager.average(diagnose=diagnose)
            end = time.perf_counter()
            total_ms = (end - begin) * 1000 - sync["diagnostic_ms"]
            last = round_steps[-1]
            last["t_update_ms"] += (end - boundary) * 1000 - sync["diagnostic_ms"]
            last["t_sync_ms"] = sync["t_sync_ms"]
            last["sync_happened"] = True
            last["sync_type"] = "parameter_allreduce"
            last["ar_count"] = 1
            last["ar_bytes"] = sync["ar_bytes"]
            last["t_other_ms"] = (
                last["t_update_ms"]
                - last["t_data_ms"]
                - last["t_compute_ms"]
                - last["t_opt_ms"]
                - last["t_sync_ms"]
            )
            if any(row["t_other_ms"] < -1e-7 for row in round_steps):
                raise RuntimeError("本地训练步分段无法加和")
            other_ms = total_ms - data_ms - compute_ms - opt_ms - sync["t_sync_ms"]
            if other_ms < -1e-7:
                raise RuntimeError("轮次时间分段无法加和")
            full = (
                count == self.level
                and local_samples == self.level * self.config.micro_batch
            )
            global_first = (epoch - 1) * len(self.loader) + first
            value = sum(v.item() * n for v, n in losses) / local_samples
            for row in round_steps:
                row["loss"] = row["loss"].item()
                row["local_throughput"] = (
                    row["local_samples"] * 1000 / row["t_update_ms"]
                )
                row["t_other_ms"] = max(0.0, row["t_other_ms"])
            local_rows.extend(round_steps)
            rows.append(
                dict(
                    epoch=epoch,
                    round=round_index,
                    iteration=self.round_iteration,
                    first_step=first,
                    global_first_step=global_first,
                    rank=self.rank,
                    steps_count=count,
                    local_samples=local_samples,
                    full_round=full,
                    measured=full and global_first >= self.config.warmup_updates,
                    t_data_ms=data_ms,
                    t_compute_ms=compute_ms,
                    t_sync_ms=sync["t_sync_ms"],
                    t_opt_ms=opt_ms,
                    t_other_ms=max(0.0, other_ms),
                    t_round_ms=total_ms,
                    diagnostic_ms=sync["diagnostic_ms"],
                    t_round_with_diagnostic_ms=(end - begin) * 1000,
                    t_pack_ms=sync["t_pack_ms"],
                    t_collective_ms=sync["t_collective_ms"],
                    t_copy_ms=sync["t_copy_ms"],
                    sync_arrival_s=sync["sync_arrival_s"],
                    ar_count=1,
                    ar_bytes=sync["ar_bytes"],
                    drift_measured=diagnose,
                    drift_relative_l2=sync["drift_relative_l2"],
                    loss=value,
                )
            )
            if diagnose:
                self.last_drift = sync["drift_relative_l2"]
            self.round_iteration += 1
            loss_sum += value * local_samples
            samples += local_samples
        self.peak_allocated = max(
            self.peak_allocated, torch.cuda.max_memory_allocated(self.device)
        )
        write_rows(self.folder / f"rounds_rank{self.rank}.csv", rows, append=epoch > 1)
        write_local_steps(
            self.folder / f"steps_rank{self.rank}.csv", local_rows, append=epoch > 1
        )
        (self.folder / f"sampling_rank{self.rank}.json").write_text(
            json.dumps(self.sampling_hashes, indent=2) + "\n"
        )
        totals = torch.tensor(
            [loss_sum, samples], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(totals)
        return (totals[0] / totals[1]).item()

    def train(self):
        val_loader = DataLoader(
            self.split.subset("validation"),
            batch_size=self.config.eval_batch,
            shuffle=False,
        )
        history = []
        started = time.perf_counter()
        train_wall = 0.0
        target_time = None
        target_epoch = None
        for epoch in range(1, 21):
            before = time.perf_counter()
            train_loss = self.train_epoch(epoch)
            train_wall += time.perf_counter() - before
            if self.rank == 0:
                val = evaluate_model(self.model, val_loader)
                elapsed = time.perf_counter() - started
                if not math.isfinite(train_loss + val["loss"]):
                    raise RuntimeError("训练或验证非有限")
                if (
                    target_time is None
                    and val["accuracy"] >= self.config.target_accuracy
                ):
                    target_time, target_epoch = elapsed, epoch
                history.append(
                    dict(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val["loss"],
                        val_acc=val["accuracy"],
                        elapsed_wall_s=elapsed,
                        elapsed_training_wall_s=train_wall,
                        drift_pre_sync_relative_l2=self.last_drift,
                    )
                )
                pd.DataFrame(history).to_csv(self.folder / "epochs.csv", index=False)
                print(
                    f"[epoch] {self.identifier} {epoch}/20 val={val['loss']:.5f} acc={val['accuracy']:.4%}",
                    flush=True,
                )
            dist.barrier()
        (self.folder / f"memory_rank{self.rank}.json").write_text(
            json.dumps(
                dict(
                    allocated_bytes=self.peak_allocated, includes_drift_diagnostics=True
                )
            )
            + "\n"
        )
        dist.barrier()
        final = None
        if self.rank == 0:
            tested = evaluate_model(
                self.model,
                DataLoader(
                    self.split.subset("test"),
                    batch_size=self.config.eval_batch,
                    shuffle=False,
                ),
            )
            frame = pd.concat(
                [pd.read_csv(self.folder / f"rounds_rank{r}.csv") for r in (0, 1)],
                ignore_index=True,
            )
            size = self.model.size_info()
            performance = summarize_rounds(
                frame, self.config, self.level, size["gradient_bytes"]
            )
            final = dict(
                experiment=EXPERIMENT,
                run_id=self.identifier,
                dataset=self.config.dataset,
                model="M",
                level=self.level,
                variant=self.variant,
                sync_interval=self.level,
                micro_batch=self.config.micro_batch,
                seed=self.seed,
                ws=2,
                global_batch=self.config.global_batch_for(self.level),
                epochs_ran=20,
                test_epoch=20,
                selection="last",
                timing_schema_version=2,
                train_samples=len(self.split.train_indices),
                val_samples=len(self.split.validation_indices),
                test_samples=tested["samples"],
                train_loss=train_loss,
                val_loss=val["loss"],
                val_acc=val["accuracy"],
                test_loss=tested["loss"],
                test_acc=tested["accuracy"],
                training_stage_wall_s=history[-1]["elapsed_wall_s"],
                target_accuracy=self.config.target_accuracy,
                target_reached=target_time is not None,
                time_to_target_s=target_time,
                target_epoch=target_epoch,
                wall_time_s=time.perf_counter() - started,
                **size,
                **performance,
            )
            save_checkpoint(
                self.folder / "checkpoint.pt",
                self.model,
                self.config,
                self.level,
                self.variant,
                self.seed,
                self.identifier,
                {
                    key: final[key]
                    for key in ("val_loss", "val_acc", "test_loss", "test_acc")
                },
            )
            (self.folder / "final.json").write_text(json.dumps(final, indent=2) + "\n")
            print(
                f"[final] {self.identifier} Q={final['train_throughput']:.1f} sync={final['sync_share']:.2%} target={target_epoch}",
                flush=True,
            )
        dist.barrier()
        return final


def train_worker(rank, port, config, level, variant, seed, folder, data_root):
    initialize(rank, 2, port, config.host_threads)
    try:
        Trainer(config, level, variant, seed, rank, folder, data_root).train()
    finally:
        cleanup()
