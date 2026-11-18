"""E3 单模型单 GPU：相同全局更新窗口，独立完成训练、验证与末轮测试。"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from Data.datasets import DATA_ROOT
from E3.config import EXPERIMENT
from E3.data import load_split
from E3.distributed import initialize
from E3.evaluate import evaluate_model
from E3.model import MLP
from E3.single_gpu.config import reference_config, saved_config
from E3.single_gpu.data import global_loader, sampling_record
from E3.single_gpu.metrics import summarize, write_rows


class Trainer:
    def __init__(self, case, folder, data_root=DATA_ROOT, *, split=None):
        self.case, self.folder = case, Path(folder)
        self.config = reference_config(case.dataset)
        self.device = initialize(0, 1, 0, self.config.host_threads)
        if dist.is_initialized():
            raise RuntimeError("单 GPU 基线必须在独立单进程中执行，不建立分布式进程组")
        torch.manual_seed(case.seed)
        self.split = load_split(self.config, data_root) if split is None else split
        self.loader = global_loader(self.split, case)
        self.model = MLP(self.split.raw.in_dim).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "config.json").write_text(
            json.dumps(saved_config(case), indent=2) + "\n"
        )
        self.split.save(self.folder, self.config)
        runtime = {
            "experiment": EXPERIMENT,
            "run_id": case.identifier,
            "device": str(self.device),
            "physical_gpus": 1,
            "training_processes": 1,
            "logical_replicas": 1,
            "world_size": 1,
            "backend": None,
            "cross_gpu_communication": False,
            "dtype": "float32",
            "allow_tf32": False,
            "host_threads": torch.get_num_threads(),
            "micro_batch": case.micro_batch,
            "accumulation": case.accumulation,
            "global_batch": case.global_batch,
            "timing_mode": "cuda_synchronized_global_window",
        }
        (self.folder / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        self.micro_iteration = 0
        self.sampling, self.peak_allocated = {}, 0

    def backward_window(self, features, labels):
        size, data_ms, compute_ms = len(labels), 0.0, 0.0
        losses = []
        for offset in range(0, size, self.case.micro_batch):
            end = min(offset + self.case.micro_batch, size)
            started = time.perf_counter()
            x, y = (
                features[offset:end].to(self.device),
                labels[offset:end].to(self.device),
            )
            torch.cuda.synchronize(self.device)
            data_ms += (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            loss = torch.nn.functional.cross_entropy(self.model(x), y)
            if self.case.accumulation == 1:
                loss.backward()
            else:
                (loss * ((end - offset) / size)).backward()
            torch.cuda.synchronize(self.device)
            compute_ms += (time.perf_counter() - started) * 1000
            losses.append((loss.detach(), end - offset))
        return data_ms, compute_ms, losses

    def train_epoch(self, epoch):
        self.model.train()
        self.loader.sampler.set_epoch(epoch - 1)
        self.sampling[str(epoch)] = sampling_record(self.loader.sampler)
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        rows, loss_sum, samples = [], 0.0, 0
        torch.cuda.synchronize(self.device)
        previous = time.perf_counter()
        for unit, (features, labels) in enumerate(self.loader):
            started = previous
            fetch_ms = (time.perf_counter() - started) * 1000
            data_ms, compute_ms, losses = self.backward_window(features, labels)
            opt_started = time.perf_counter()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.device)
            ended = time.perf_counter()
            opt_ms, total_ms = (ended - opt_started) * 1000, (ended - started) * 1000
            data_ms += fetch_ms
            other_ms = total_ms - data_ms - compute_ms - opt_ms
            if other_ms < -1e-7:
                raise RuntimeError("单卡时间分段无法加和")
            size = len(labels)
            value = sum(value.item() * count for value, count in losses) / size
            iteration = (epoch - 1) * len(self.loader) + unit
            full = size == self.case.global_batch
            measured = (
                full
                and iteration >= self.case.warmup_updates
                and self.micro_iteration >= self.case.warmup_micro_steps
            )
            rows.append(
                {
                    "epoch": epoch,
                    "unit": unit,
                    "iteration": iteration,
                    "first_update": unit,
                    "global_first_update": iteration,
                    "steps_count": 1,
                    "global_samples": size,
                    "micro_count": len(losses),
                    "micro_start": self.micro_iteration,
                    "full_batch": full,
                    "measured": measured,
                    "t_data_ms": data_ms,
                    "t_compute_ms": compute_ms,
                    "t_local_average_ms": 0.0,
                    "t_sync_ms": 0.0,
                    "t_opt_ms": opt_ms,
                    "t_other_ms": max(0.0, other_ms),
                    "t_unit_ms": total_ms,
                    "diagnostic_ms": 0.0,
                    "t_unit_with_diagnostic_ms": total_ms,
                    "optimizer_calls": 1,
                    "local_average_count": 0,
                    "ar_count": 0,
                    "ar_bytes": 0,
                    "drift_measured": False,
                    "drift_relative_l2": None,
                    "drift_replica0_relative_l2": None,
                    "drift_replica1_relative_l2": None,
                    "loss": value,
                }
            )
            self.micro_iteration += len(losses)
            loss_sum += value * size
            samples += size
            previous = time.perf_counter()
        self.peak_allocated = max(
            self.peak_allocated, torch.cuda.max_memory_allocated(self.device)
        )
        write_rows(self.folder / "steps.csv", rows, append=epoch > 1)
        (self.folder / "sampling.json").write_text(
            json.dumps(self.sampling, indent=2) + "\n"
        )
        return loss_sum / samples

    def train(self):
        validation = DataLoader(
            self.split.subset("validation"),
            batch_size=self.config.eval_batch,
            shuffle=False,
        )
        history, started = [], time.perf_counter()
        target_time = target_epoch = None
        for epoch in range(1, 21):
            train_loss = self.train_epoch(epoch)
            val = evaluate_model(self.model, validation)
            elapsed = time.perf_counter() - started
            if not math.isfinite(train_loss + val["loss"]):
                raise RuntimeError("训练或验证非有限")
            target = getattr(self.config, "target_accuracy", None)
            if target is not None and target_time is None and val["accuracy"] >= target:
                target_time, target_epoch = elapsed, epoch
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_acc": val["accuracy"],
                    "elapsed_wall_s": elapsed,
                }
            )
            pd.DataFrame(history).to_csv(self.folder / "epochs.csv", index=False)
            print(
                f"[epoch] {self.case.identifier} {epoch}/20 val={val['loss']:.5f} acc={val['accuracy']:.4%}",
                flush=True,
            )
        (self.folder / "memory.json").write_text(
            json.dumps({"allocated_bytes": self.peak_allocated}) + "\n"
        )
        tested = evaluate_model(
            self.model,
            DataLoader(
                self.split.subset("test"),
                batch_size=self.config.eval_batch,
                shuffle=False,
            ),
        )
        frame = pd.read_csv(self.folder / "steps.csv")
        final = {
            "experiment": EXPERIMENT,
            "run_id": self.case.identifier,
            "baseline_role": self.case.role,
            "dataset": self.case.dataset,
            "reference_level": self.case.reference_level,
            "seed": self.case.seed,
            "ws": 1,
            "physical_gpus": 1,
            "training_processes": 1,
            "logical_replicas": 1,
            "micro_batch": self.case.micro_batch,
            "accumulation": self.case.accumulation,
            "global_batch": self.case.global_batch,
            "epochs_ran": 20,
            "test_epoch": 20,
            "selection": "last",
            "train_samples": len(self.split.train_indices),
            "val_samples": len(self.split.validation_indices),
            "test_samples": tested["samples"],
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "test_loss": tested["loss"],
            "test_acc": tested["accuracy"],
            "peak_allocated_mb": self.peak_allocated / 1e6,
            "training_stage_wall_s": history[-1]["elapsed_wall_s"],
            "target_accuracy": getattr(self.config, "target_accuracy", None),
            "target_epoch": target_epoch,
            "target_reached": target_time is not None,
            "time_to_target_s": target_time,
            "wall_time_s": time.perf_counter() - started,
            **self.model.size_info(),
            **summarize(frame, self.case),
        }
        metrics = {
            key: final[key] for key in ("val_loss", "val_acc", "test_loss", "test_acc")
        }
        saved = {
            "format_version": 2,
            "experiment": EXPERIMENT,
            "run_id": self.case.identifier,
            "baseline_role": self.case.role,
            "epoch": 20,
            "selection": "last",
            "dataset": self.case.dataset,
            "model": "M",
            "in_dim": self.model.in_dim,
            "seed": self.case.seed,
            "ws": 1,
            "micro_batch": self.case.micro_batch,
            "accumulation": self.case.accumulation,
            "global_batch": self.case.global_batch,
            "training_config": asdict(self.config),
            "baseline_case": asdict(self.case),
            "metrics": metrics,
            "state_dict": {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            },
        }
        torch.save(saved, self.folder / "checkpoint.pt")
        (self.folder / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        print(
            f"[final] {self.case.identifier} Q={final['train_throughput']:.1f} GPU=1",
            flush=True,
        )
        return final


def worker(case, folder, data_root):
    Trainer(case, folder, data_root).train()
