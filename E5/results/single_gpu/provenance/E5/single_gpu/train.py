"""单 GPU 串行运行两个 Local SGD 副本，逐步更新，各自保留 momentum。"""
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
from E5.data import load_split
from E5.distributed import initialize
from E5.evaluate import evaluate_model
from E5.model import MLP
from E5.single_gpu.config import reference_config, saved_config
from E5.single_gpu.data import serial_loaders, serial_sampling_record
from E5.single_gpu.local_average import LocalAverager
from E5.single_gpu.metrics import summarize, write_rows, write_local_steps


class SerialTrainer:
    def __init__(self, case, folder, data_root=DATA_ROOT, *, split=None):
        if case.role != "serial_local_sgd":
            raise ValueError("SerialTrainer 只运行串行双副本")
        self.case, self.folder = case, Path(folder)
        self.config = reference_config(case.dataset)
        self.device = initialize(0, 1, 0, self.config.host_threads)
        if dist.is_initialized():
            raise RuntimeError("单 GPU 串行基线不建立进程组")
        torch.manual_seed(case.seed)
        self.split = load_split(self.config, data_root) if split is None else split
        self.loaders = serial_loaders(self.split, case)
        self.models = [MLP(self.split.raw.in_dim).to(self.device) for _ in (0, 1)]
        self.models[1].load_state_dict(self.models[0].state_dict())
        self.averager = LocalAverager(self.models)
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=self.config.learning_rate,
                                           momentum=self.config.momentum) for model in self.models]
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "config.json").write_text(json.dumps(saved_config(case), indent=2) + "\n")
        self.split.save(self.folder, self.config)
        runtime = {"experiment": "E5", "run_id": case.identifier, "device": str(self.device),
                   "physical_gpus": 1, "training_processes": 1, "logical_replicas": 2,
                   "world_size": 1, "backend": None, "cross_gpu_communication": False,
                   "dtype": "float32", "allow_tf32": False, "host_threads": torch.get_num_threads(),
                   "micro_batch_per_replica": case.micro_batch, "sync_interval": case.reference_level,
                   "global_batch_per_logical_step": case.global_batch, "momentum_policy": "local_unsynchronized",
                   "epoch_end_policy": "flush_partial_round", "timing_mode": "cuda_synchronized_serial_round",
                   "timing_schema_version": 2}
        (self.folder / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        self.sampling, self.peak_allocated = {}, 0
        self.iteration = self.micro_iteration = 0

    def local_update(self, replica, features, labels):
        started = time.perf_counter()
        x, y = features.to(self.device), labels.to(self.device)
        torch.cuda.synchronize(self.device)
        data_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        loss = torch.nn.functional.cross_entropy(self.models[replica](x), y)
        loss.backward()
        torch.cuda.synchronize(self.device)
        compute_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        self.optimizers[replica].step()
        self.optimizers[replica].zero_grad(set_to_none=True)
        torch.cuda.synchronize(self.device)
        return data_ms, compute_ms, (time.perf_counter() - started) * 1000, loss.detach()

    def train_epoch(self, epoch):
        for model, loader, optimizer in zip(self.models, self.loaders, self.optimizers):
            model.train()
            loader.sampler.set_epoch(epoch - 1)
            optimizer.zero_grad(set_to_none=True)
        self.sampling[str(epoch)] = serial_sampling_record(self.loaders)
        torch.cuda.reset_peak_memory_stats(self.device)
        full_steps = len(self.loaders[0].dataset) // self.case.global_batch
        h = self.case.reference_level
        diagnostic_end = (full_steps // h) * h
        iterators = [iter(loader) for loader in self.loaders]
        rows, local_rows, loss_sum, samples = [], [], 0.0, 0
        for unit, first in enumerate(range(0, len(self.loaders[0]), h)):
            count = min(h, len(self.loaders[0]) - first)
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            data_ms = compute_ms = opt_ms = 0.0
            losses, global_samples, round_steps = [], 0, []
            boundary = started
            for offset in range(count):
                for replica in (0, 1):
                    before = time.perf_counter()
                    features, labels = next(iterators[replica])
                    fetch_ms = (time.perf_counter() - before) * 1000
                    data, compute, optimize, loss = self.local_update(replica, features, labels)
                    ended_step = time.perf_counter()
                    step_ms = (ended_step - boundary) * 1000
                    step_data = data + fetch_ms
                    step = first + offset
                    round_steps.append({"epoch": epoch, "step": step,
                        "iteration": (epoch-1)*len(self.loaders[0])+step, "replica": replica,
                        "round_iteration": self.iteration, "local_samples": len(labels),
                        "t_data_ms": step_data, "t_compute_ms": compute, "t_opt_ms": optimize,
                        "t_local_average_ms": 0.0, "t_sync_ms": 0.0,
                        "t_other_ms": max(0.0, step_ms-step_data-compute-optimize),
                        "t_update_ms": step_ms, "local_throughput": 0.0,
                        "local_average_count": 0, "ar_count": 0, "ar_bytes": 0, "loss": loss})
                    boundary = ended_step
                    data_ms += step_data
                    compute_ms += compute
                    opt_ms += optimize
                    global_samples += len(labels)
                    losses.append((loss, len(labels)))
            diagnose = diagnostic_end > 0 and first + count == diagnostic_end
            averaged = self.averager.average(diagnose=diagnose)
            ended = time.perf_counter()
            total_ms = (ended - started) * 1000 - averaged["diagnostic_ms"]
            last = round_steps[-1]
            last["t_update_ms"] += (ended-boundary)*1000-averaged["diagnostic_ms"]
            last["t_local_average_ms"] = averaged["local_average_ms"]
            last["local_average_count"] = 1
            last["t_other_ms"] = last["t_update_ms"]-last["t_data_ms"]-last["t_compute_ms"]-last["t_opt_ms"]-last["t_local_average_ms"]
            if any(row["t_other_ms"] < -1e-7 for row in round_steps):
                raise RuntimeError("串行本地训练步分段无法加和")
            other_ms = total_ms - data_ms - compute_ms - opt_ms - averaged["local_average_ms"]
            if other_ms < -1e-7:
                raise RuntimeError("串行轮次分段无法加和")
            value = sum(value.item() * size for value, size in losses) / global_samples
            for row in round_steps:
                row["loss"] = row["loss"].item()
                row["local_throughput"] = row["local_samples"]*1000/row["t_update_ms"]
                row["t_other_ms"] = max(0.0,row["t_other_ms"])
            local_rows.extend(round_steps)
            full = count == h and global_samples == self.case.global_batch * h
            global_first = (epoch - 1) * len(self.loaders[0]) + first
            drift = averaged["drift"]
            mean_drift = (drift[0] + drift[1]) / 2 if diagnose else None
            rows.append({"epoch": epoch, "unit": unit, "iteration": self.iteration,
                         "first_update": first, "global_first_update": global_first, "steps_count": count,
                         "global_samples": global_samples, "micro_count": 2 * count, "micro_start": self.micro_iteration,
                         "full_batch": full, "measured": full and global_first >= self.case.warmup_updates,
                         "t_data_ms": data_ms, "t_compute_ms": compute_ms,
                         "t_local_average_ms": averaged["local_average_ms"], "t_sync_ms": 0.0,
                         "t_opt_ms": opt_ms, "t_other_ms": max(0.0, other_ms), "t_unit_ms": total_ms,
                         "diagnostic_ms": averaged["diagnostic_ms"], "t_unit_with_diagnostic_ms": (ended - started) * 1000,
                         "post_unit_ms": 0.0,
                         "optimizer_calls": 2 * count, "local_average_count": 1, "ar_count": 0, "ar_bytes": 0,
                         "drift_measured": diagnose, "drift_relative_l2": mean_drift,
                         "drift_replica0_relative_l2": drift[0], "drift_replica1_relative_l2": drift[1], "loss": value})
            self.iteration += 1
            self.micro_iteration += 2 * count
            loss_sum += value * global_samples
            samples += global_samples
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated(self.device))
        write_rows(self.folder / "rounds.csv", rows, append=epoch > 1)
        write_local_steps(self.folder / "local_steps.csv", local_rows, append=epoch > 1)
        (self.folder / "sampling.json").write_text(json.dumps(self.sampling, indent=2) + "\n")
        return loss_sum / samples

    def train(self):
        validation = DataLoader(self.split.subset("validation"), batch_size=self.config.eval_batch, shuffle=False)
        history, started = [], time.perf_counter()
        target_time = target_epoch = None
        for epoch in range(1, 21):
            train_loss = self.train_epoch(epoch)
            val = evaluate_model(self.models[0], validation)
            elapsed = time.perf_counter() - started
            if not math.isfinite(train_loss + val["loss"]):
                raise RuntimeError("串行训练或验证非有限")
            if target_time is None and val["accuracy"] >= self.config.target_accuracy:
                target_time, target_epoch = elapsed, epoch
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                            "val_acc": val["accuracy"], "elapsed_wall_s": elapsed})
            pd.DataFrame(history).to_csv(self.folder / "epochs.csv", index=False)
            print(f"[epoch] {self.case.identifier} {epoch}/20 val={val['loss']:.5f} acc={val['accuracy']:.4%}", flush=True)
        (self.folder / "memory.json").write_text(json.dumps({"allocated_bytes": self.peak_allocated,
                                                             "includes_two_replicas_and_diagnostics": True}) + "\n")
        tested = evaluate_model(self.models[0], DataLoader(self.split.subset("test"), batch_size=self.config.eval_batch, shuffle=False))
        frame = pd.read_csv(self.folder / "rounds.csv")
        final = {"experiment": "E5", "run_id": self.case.identifier, "baseline_role": self.case.role,
                 "dataset": self.case.dataset, "reference_level": self.case.reference_level, "seed": self.case.seed,
                 "ws": 1, "physical_gpus": 1, "training_processes": 1, "logical_replicas": 2,
                 "micro_batch": self.case.micro_batch, "accumulation": 1, "global_batch": self.case.global_batch,
                 "sync_interval": self.case.reference_level, "epochs_ran": 20, "test_epoch": 20, "selection": "last",
                 "timing_schema_version": 2,
                 "train_samples": len(self.split.train_indices), "val_samples": len(self.split.validation_indices),
                 "test_samples": tested["samples"], "train_loss": train_loss, "val_loss": val["loss"],
                 "val_acc": val["accuracy"], "test_loss": tested["loss"], "test_acc": tested["accuracy"],
                 "peak_allocated_mb": self.peak_allocated / 1e6, "training_stage_wall_s": history[-1]["elapsed_wall_s"],
                 "target_accuracy": self.config.target_accuracy, "target_reached": target_time is not None,
                 "time_to_target_s": target_time, "target_epoch": target_epoch,
                 "wall_time_s": time.perf_counter() - started, **self.models[0].size_info(), **summarize(frame, self.case)}
        equal = all(torch.equal(a, b) for a, b in zip(self.models[0].parameters(), self.models[1].parameters()))
        if not equal:
            raise RuntimeError("保存时两个副本未平均为共同参数")
        metrics = {key: final[key] for key in ("val_loss", "val_acc", "test_loss", "test_acc")}
        saved = {"format_version": 2, "experiment": "E5", "run_id": self.case.identifier,
                 "baseline_role": self.case.role, "epoch": 20, "selection": "last", "dataset": self.case.dataset,
                 "model": "M", "in_dim": self.models[0].in_dim, "seed": self.case.seed, "ws": 1,
                 "logical_replicas": 2, "replica_parameters_equal": equal,
                 "training_config": asdict(self.config), "baseline_case": asdict(self.case), "metrics": metrics,
                 "state_dict": {key: value.detach().clone() for key, value in self.models[0].state_dict().items()},
                 "optimizer_states": [optimizer.state_dict() for optimizer in self.optimizers]}
        torch.save(saved, self.folder / "checkpoint.pt")
        (self.folder / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        print(f"[final] {self.case.identifier} Q={final['train_throughput']:.1f} GPU=1 replicas=2", flush=True)
        return final


def worker(case, folder, data_root):
    if case.role == "ordinary_sgd":
        from E5.single_gpu.global_train import Trainer
        Trainer(case, folder, data_root).train()
    else:
        SerialTrainer(case, folder, data_root).train()
