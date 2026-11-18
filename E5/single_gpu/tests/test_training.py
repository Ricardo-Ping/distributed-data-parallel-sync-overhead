from __future__ import annotations

import ast
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Data.datasets import ImageData
from E5.data import Split, stratified_indices
from E5.distributed import cleanup, initialize
from E5.model import MLP, load_checkpoint
from E5.single_gpu import global_train, train
from E5.single_gpu.config import POINTS, cases_for, make_case, reference_config
from E5.single_gpu.data import (
    global_loader,
    sampling_record,
    serial_loaders,
    serial_sampling_record,
)


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
    cases = cases_for()
    assert len(cases) == 54 == len({str(case.folder()) for case in cases})
    assert sum(case.role == "serial_local_sgd" for case in cases) == 45
    assert sum(case.role == "ordinary_sgd" for case in cases) == 9
    output = tmp_path / "read_only"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "E5.single_gpu.run",
            "--list",
            "--output-dir",
            str(output),
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.count("GPUs=1") == 54 and not output.exists()
    for source in Path("E5/single_gpu").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0][1:].isdigit()
            ):
                assert node.module.startswith("E5.")


def test_serial_and_ordinary_loaders_have_identical_global_sample_order():
    config, split = reference_config("mnist"), synthetic(1140)
    for h in POINTS:
        serial = serial_loaders(split, make_case(config, h, 1234))
        ordinary = global_loader(split, make_case(config, 1, 1234, "ordinary_sgd"))
        for epoch in (0, 7, 19):
            for loader in [*serial, ordinary]:
                loader.sampler.set_epoch(epoch)
            assert serial_sampling_record(serial) == sampling_record(ordinary.sampler)


@pytest.mark.parametrize("h", POINTS)
def test_local_sgd_updates_and_momentum_match_independent_cuda_reference(tmp_path, h):
    config = reference_config("mnist")
    trainer = train.SerialTrainer(
        make_case(config, h, 1234), tmp_path / str(h), split=synthetic()
    )
    references = [MLP(784).to("cuda:0") for _ in (0, 1)]
    for actual, expected in zip(trainer.models, references):
        expected.load_state_dict(actual.state_dict())
    optimizers = [
        torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        for model in references
    ]
    syncs = 0
    for index, batches in enumerate(zip(*trainer.loaders)):
        for replica, (features, labels) in enumerate(batches):
            trainer.local_update(replica, features, labels)
            loss = torch.nn.functional.cross_entropy(
                references[replica](features.cuda()), labels.cuda()
            )
            loss.backward()
            optimizers[replica].step()
            optimizers[replica].zero_grad(set_to_none=True)
        boundary = (index + 1) % h == 0 or index + 1 == len(trainer.loaders[0])
        if boundary:
            before = [
                [
                    optimizer.state[p]["momentum_buffer"].clone()
                    for p in model.parameters()
                ]
                for optimizer, model in zip(trainer.optimizers, trainer.models)
            ]
            averaged = trainer.averager.average(diagnose=index + 1 == 16)
            assert averaged["cross_gpu_allreduces"] == 0
            with torch.no_grad():
                for first, second in zip(
                    references[0].parameters(), references[1].parameters()
                ):
                    mean = (first + second) / 2
                    first.copy_(mean)
                    second.copy_(mean)
            for replica in (0, 1):
                for actual, expected, original in zip(
                    trainer.models[replica].parameters(),
                    references[replica].parameters(),
                    before[replica],
                ):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                    momentum = trainer.optimizers[replica].state[actual][
                        "momentum_buffer"
                    ]
                    assert torch.equal(momentum, original)
                    torch.testing.assert_close(
                        momentum,
                        optimizers[replica].state[expected]["momentum_buffer"],
                        rtol=2e-5,
                        atol=2e-6,
                    )
            syncs += 1
    assert syncs == math.ceil(len(trainer.loaders[0]) / h)
    for first, second in zip(
        trainer.models[0].parameters(), trainer.models[1].parameters()
    ):
        assert torch.equal(first, second)
    differences = [
        (
            trainer.optimizers[0].state[a]["momentum_buffer"]
            - trainer.optimizers[1].state[b]["momentum_buffer"]
        )
        .abs()
        .max()
        .item()
        for a, b in zip(trainer.models[0].parameters(), trainer.models[1].parameters())
    ]
    assert max(differences) > 0


@pytest.mark.parametrize("role", ("serial_local_sgd", "ordinary_sgd"))
def test_twenty_epochs_last_model_and_validation_test_separation(
    tmp_path, monkeypatch, role
):
    config = reference_config("mnist")
    case = make_case(config, 16 if role == "serial_local_sgd" else 1, 1234, role)
    split = synthetic()
    module, cls = (
        (train, train.SerialTrainer)
        if role == "serial_local_sgd"
        else (global_train, global_train.Trainer)
    )
    trainer = cls(case, tmp_path / role, split=split)
    original, events = module.evaluate_model, []

    def evaluated(model, loader):
        result = original(model, loader)
        assert result["device"] == "cuda:0"
        events.append(len(loader.dataset))
        if len(loader.dataset) == len(split.validation_indices):
            result["loss"] = 0.0 if len(events) == 1 else result["loss"] + 1
        return result

    monkeypatch.setattr(module, "evaluate_model", evaluated)
    final = trainer.train()
    assert events == [len(split.validation_indices)] * 20 + [32]
    updates = 20 * math.ceil(len(split.train_indices) / case.global_batch)
    assert (
        final["total_updates"] == updates
        and final["total_optimizer_calls"] == updates * case.replicas
    )
    assert final["total_train_samples"] == 20 * len(split.train_indices)
    assert final["total_model_allreduces"] == 0
    actual_model = trainer.models[0] if role == "serial_local_sgd" else trainer.model
    model, saved = load_checkpoint(trainer.folder / "checkpoint.pt")
    for actual, expected in zip(model.parameters(), actual_model.parameters()):
        assert torch.equal(actual, expected)
    assert (saved["epoch"], saved["selection"], saved["ws"]) == (20, "last", 1)
    frame = pd.read_csv(trainer.folder / case.units_filename)
    assert frame.ar_count.eq(0).all() and frame.t_sync_ms.eq(0).all()
    assert frame.post_unit_ms.ge(0).all()
    assert frame.drift_measured.sum() == (20 if role == "serial_local_sgd" else 0)
    if role == "serial_local_sgd":
        local = pd.read_csv(trainer.folder / "local_steps.csv")
        assert len(local) == updates * 2
        assert local.groupby("iteration").replica.nunique().eq(2).all()
        sums = local.groupby("round_iteration").t_update_ms.sum()
        torch.testing.assert_close(
            torch.tensor(sums.to_numpy(), device="cuda:0"),
            torch.tensor(frame.t_unit_ms.to_numpy(), device="cuda:0"),
            rtol=1e-9,
            atol=1e-7,
        )


def nccl_worker(rank, port, folder):
    from E5.train import Trainer as DistributedTrainer

    folder = Path(folder)
    config, split = reference_config("mnist"), synthetic()
    serial = (
        train.SerialTrainer(make_case(config, 4, 1234), folder / "serial", split=split)
        if rank == 0
        else None
    )
    initialize(rank, 2, port, 4)
    try:
        actual = DistributedTrainer(
            config, 4, "local_sgd", 1234, rank, folder / "dual", split=split
        )
        source = iter(actual.loader)
        references = [iter(loader) for loader in serial.loaders] if rank == 0 else None
        for index in range(len(actual.loader)):
            actual.local_update(*next(source))
            if rank == 0:
                for replica in (0, 1):
                    serial.local_update(replica, *next(references[replica]))
            if (index + 1) % 4 == 0 or index + 1 == len(actual.loader):
                actual.averager.average()
                if rank == 0:
                    serial.averager.average()
                    for first, second in zip(
                        actual.model.parameters(), serial.models[0].parameters()
                    ):
                        torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6)
                dist.barrier()
        torch.save(
            {
                "parameters": actual.model.state_dict(),
                "optimizer": actual.optimizer.state_dict(),
            },
            folder / f"dual_rank{rank}.pt",
        )
        if rank == 0:
            torch.save(
                {
                    "parameters": [model.state_dict() for model in serial.models],
                    "optimizers": [
                        optimizer.state_dict() for optimizer in serial.optimizers
                    ],
                },
                folder / "serial.pt",
            )
        dist.barrier()
    finally:
        cleanup()


def test_serial_baseline_matches_real_two_gpu_nccl_algorithm(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(nccl_worker, args=(port, str(tmp_path)), nprocs=2, join=True)
    expected = torch.load(
        tmp_path / "serial.pt", map_location="cuda:0", weights_only=True
    )
    for rank in (0, 1):
        actual = torch.load(
            tmp_path / f"dual_rank{rank}.pt", map_location="cuda:0", weights_only=True
        )
        for name, value in actual["parameters"].items():
            torch.testing.assert_close(
                value, expected["parameters"][rank][name], rtol=2e-5, atol=2e-6
            )
        for key, state in actual["optimizer"]["state"].items():
            torch.testing.assert_close(
                state["momentum_buffer"],
                expected["optimizers"][rank]["state"][key]["momentum_buffer"],
                rtol=2e-5,
                atol=2e-6,
            )
