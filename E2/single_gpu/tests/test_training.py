from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DistributedSampler

from Data.datasets import ImageData
from E2.config import EXPERIMENT
from E2.data import Split, stratified_indices
from E2.model import MLP, load_checkpoint
from E2.single_gpu import train as training
from E2.single_gpu.config import POINTS, cases_for, make_case, reference_config
from E2.single_gpu.data import global_loader, sampling_record


def synthetic(count=4560):
    generator = torch.Generator().manual_seed(917)
    labels = torch.arange(count) % 10
    raw = ImageData(
        "mnist",
        torch.randn(count, 784, generator=generator),
        labels,
        torch.randn(32, 784, generator=generator),
        torch.arange(32) % 10,
    )
    return Split(raw, *stratified_indices(labels, 0.1, 1234))


def test_plan_is_read_only_and_framework_is_independent(tmp_path):
    jobs = cases_for()
    expected = len(POINTS) * 3 * 3
    assert len(jobs) == expected == len({str(case.folder()) for case in jobs})
    assert len({case.identifier for case in jobs}) == expected
    output = tmp_path / "read_only"
    command = [
        sys.executable,
        "-m",
        f"{EXPERIMENT}.single_gpu.run",
        "--list",
        "--output-dir",
        str(output),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    listed = subprocess.run(
        command, env=env, text=True, capture_output=True, check=True
    )
    assert listed.stdout.count("GPUs=1") == expected
    assert not output.exists()
    for source in Path(EXPERIMENT, "single_gpu").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0][1:].isdigit()
            ):
                assert node.module.startswith(EXPERIMENT + ".")


def test_global_sample_order_recovers_both_distributed_ranks():
    split = synthetic(1140)
    config = reference_config("mnist")
    case = make_case(config, POINTS[0], 1234)
    loader = global_loader(split, case)
    for epoch in (0, 7, 19):
        loader.sampler.set_epoch(epoch)
        record = sampling_record(loader.sampler)
        order = np.asarray(list(loader.sampler))
        for rank in (0, 1):
            sampler = DistributedSampler(
                loader.dataset, num_replicas=2, rank=rank, shuffle=True, seed=1234
            )
            sampler.set_epoch(epoch)
            assert np.array_equal(order[rank::2], list(sampler))
            assert record[f"reference_rank{rank}_sha256"]


@pytest.mark.parametrize("point", POINTS)
def test_weighted_global_gradient_and_momentum_update_on_cuda(tmp_path, point):
    config = reference_config("mnist")
    case = make_case(config, point, 1234)
    trainer = training.Trainer(case, tmp_path / str(point), split=synthetic())
    reference = MLP(784).to("cuda:0")
    reference.load_state_dict(trainer.model.state_dict())
    optimizer = torch.optim.SGD(
        reference.parameters(), lr=config.learning_rate, momentum=config.momentum
    )
    micro64, global64 = [
        MLP(784).to(device="cuda:0", dtype=torch.float64) for _ in (0, 1)
    ]
    for model in (micro64, global64):
        model.load_state_dict(trainer.model.state_dict())
    optimizers64 = [
        torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        for model in (micro64, global64)
    ]
    generator = torch.Generator().manual_seed(41)
    for size in (case.global_batch, 8):
        features = torch.randn(size, 784, generator=generator)
        labels = torch.arange(size) % 10
        trainer.optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        _, _, losses = trainer.backward_window(features, labels)
        torch.nn.functional.cross_entropy(
            reference(features.cuda()), labels.cuda()
        ).backward()
        difference = sum(
            (a.grad.double() - b.grad.double()).square().sum()
            for a, b in zip(trainer.model.parameters(), reference.parameters())
        )
        norm = sum(b.grad.double().square().sum() for b in reference.parameters())
        assert (difference / norm).sqrt().item() < 2e-3
        assert len(losses) == math.ceil(size / case.micro_batch)
        assert sum(n for _, n in losses) == size
        trainer.optimizer.step()
        optimizer.step()
        difference = sum(
            (a.double() - b.double()).square().sum()
            for a, b in zip(trainer.model.parameters(), reference.parameters())
        )
        norm = sum(b.double().square().sum() for b in reference.parameters())
        assert (difference / norm).sqrt().item() < 2e-5
        # 另用GPU FP64验证数学上的窗口加权与momentum语义，避免要求FP32逐元素相同。
        for opt in optimizers64:
            opt.zero_grad(set_to_none=True)
        x64, y64 = features.to("cuda:0", dtype=torch.float64), labels.cuda()
        for offset in range(0, size, case.micro_batch):
            end = min(offset + case.micro_batch, size)
            loss64 = torch.nn.functional.cross_entropy(
                micro64(x64[offset:end]), y64[offset:end]
            )
            (loss64 * ((end - offset) / size)).backward()
        torch.nn.functional.cross_entropy(global64(x64), y64).backward()
        for actual, expected in zip(micro64.parameters(), global64.parameters()):
            torch.testing.assert_close(
                actual.grad, expected.grad, rtol=1e-10, atol=1e-12
            )
        for opt in optimizers64:
            opt.step()
        for actual, expected in zip(micro64.parameters(), global64.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_twenty_epochs_last_model_and_validation_test_separation(tmp_path, monkeypatch):
    config = reference_config("mnist")
    case = make_case(config, POINTS[-1], 1234)
    split = synthetic()
    trainer = training.Trainer(case, tmp_path / "whole", split=split)
    events = []
    original = training.evaluate_model

    def evaluated(model, loader):
        assert next(model.parameters()).device.type == "cuda"
        result = original(model, loader)
        events.append(len(loader.dataset))
        if len(loader.dataset) == len(split.validation_indices):
            result["loss"] = 0.0 if len(events) == 1 else result["loss"] + 1
        return result

    monkeypatch.setattr(training, "evaluate_model", evaluated)
    final = trainer.train()
    assert events == [len(split.validation_indices)] * 20 + [32]
    assert final["total_updates"] == 20 * math.ceil(
        len(split.train_indices) / case.global_batch
    )
    assert final["total_train_samples"] == 20 * len(split.train_indices)
    assert final["total_gradient_allreduces"] == 0
    saved_model, saved = load_checkpoint(trainer.folder / "checkpoint.pt")
    assert (saved["epoch"], saved["selection"], saved["ws"]) == (20, "last", 1)
    for actual, expected in zip(saved_model.parameters(), trainer.model.parameters()):
        assert torch.equal(actual, expected)
    frame = pd.read_csv(trainer.folder / "steps.csv")
    assert frame.ar_count.eq(0).all() and frame.t_sync_ms.eq(0).all()
    assert len(json.loads((trainer.folder / "sampling.json").read_text())) == 20
