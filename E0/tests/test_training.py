"""E0 的 GPU 端到端测试：尾批平均、末轮模型和测试隔离。"""

from __future__ import annotations

import copy
import csv
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from Data.datasets import ImageData
from E0.config import Config
from E0.data import Split, stratified_indices
from E0.distributed import cleanup, initialize
from E0.metrics import CSVWriter, RUN_FIELDS
from E0.model import load_checkpoint
from E0.train import Trainer


def small_split() -> Split:
    generator = torch.Generator().manual_seed(9)
    raw = ImageData(
        "mnist",
        torch.randn(300, 12, generator=generator),
        torch.arange(300) % 10,
        torch.randn(16, 12, generator=generator),
        torch.arange(16) % 10,
    )
    return Split(raw, *stratified_indices(raw.y_train, 0.1, 1234))


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _tail_worker(rank: int, world_size: int, port: int, output: str) -> None:
    initialize(rank, world_size, port)
    try:
        trainer = Trainer(
            Config("mnist", 0.01),
            1234,
            rank,
            world_size,
            Path(output),
            split=small_split(),
        )
        reference = copy.deepcopy(trainer.model)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
        features, labels = trainer.loader.dataset.tensors
        order = torch.randperm(
            len(labels), generator=torch.Generator().manual_seed(1234)
        )
        expected_sizes = [256, 14]
        updates = []
        original_step = trainer.optimizer.step

        def checked_step():
            offset = sum(updates)
            size = expected_sizes[len(updates)]
            indices = order[offset : offset + size]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(
                reference(features[indices].to(trainer.device)),
                labels[indices].to(trainer.device),
            ).backward()
            for actual, expected in zip(
                trainer.model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(
                    actual.grad, expected.grad, rtol=1e-4, atol=2e-6
                )
            original_step()
            optimizer.step()
            for actual, expected in zip(
                trainer.model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
            updates.append(size)

        trainer.optimizer.step = checked_step
        writer = (
            CSVWriter(trainer.folder / "runs.csv", RUN_FIELDS) if rank == 0 else None
        )
        try:
            trainer.train_epoch(0, writer)
        finally:
            if writer is not None:
                writer.close()
        assert updates == expected_sizes
    finally:
        cleanup()


@pytest.mark.fast
@pytest.mark.parametrize("world_size", [1, 2])
def test_tail_updates_match_global_batch_on_gpu(tmp_path, world_size):
    mp.spawn(
        _tail_worker,
        args=(world_size, free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    with (tmp_path / "mnist/runs.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["samples"]) * world_size for row in rows] == [256, 14]
    assert all(row["device"] == "cuda:0" for row in rows)
    assert all(
        row["backend"] == ("nccl" if world_size == 2 else "none") for row in rows
    )


@pytest.mark.fast
def test_twenty_epochs_validate_then_test_and_save_last_model(tmp_path, monkeypatch):
    import E0.train as training

    initialize(0, 1, 0)
    try:
        trainer = Trainer(
            Config("mnist", 0.01), 1234, 0, 1, tmp_path, split=small_split()
        )
        evaluate = training.evaluate_model
        seen, states = [], []

        def monitored(model, loader):
            metrics = evaluate(model, loader)
            assert metrics["device"] == "cuda:0"
            seen.append(len(loader.dataset))
            if len(loader.dataset) == 30:
                states.append(copy.deepcopy(model.state_dict()))
                metrics["loss"] = float(
                    len(states)
                )  # 最优验证轮为1，末轮仍须测试与保存。
            else:
                assert len(states) == 20
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, states[-1][key], rtol=0, atol=0)
            return metrics

        monkeypatch.setattr(training, "evaluate_model", monitored)
        final = trainer.train()
        assert seen == [30] * 20 + [16]
        assert final["epochs_ran"] == final["test_epoch"] == 20
        assert final["best_epoch"] == 1 and final["last_val_loss"] == 20
        model, saved = load_checkpoint(final["checkpoint_path"])
        assert saved["epoch"] == 20 and saved["selection"] == "last"
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, states[-1][key], rtol=0, atol=0)
        with (tmp_path / "mnist/epochs.csv").open() as file:
            assert all(row["test_acc"] == "" for row in csv.DictReader(file))
    finally:
        cleanup()
