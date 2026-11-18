"""E3 CUDA/NCCL完整窗口梯度、尾批与末轮测试检查。"""

import copy
import importlib
import json
import socket
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from Data.datasets import ImageData
from E3.config import Config
from E3.data import Split, stratified_indices
from E3.distributed import cleanup, initialize
from E3.model import load_checkpoint
from E3.train import Trainer


def small_split():
    generator = torch.Generator().manual_seed(7)
    raw = ImageData(
        "mnist",
        torch.randn(560, 12, generator=generator),
        torch.arange(560) % 10,
        torch.randn(16, 12, generator=generator),
        torch.arange(16) % 10,
    )
    return Split(raw, *stratified_indices(raw.y_train, 0.1, 1234))


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def gradient_worker(rank, port, output, batch):
    initialize(rank, 2, port)
    try:
        trainer = Trainer(
            Config("mnist", 0.01), batch, 1234, rank, Path(output), split=small_split()
        )
        reference = copy.deepcopy(trainer.model)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
        features, labels = trainer.loader.dataset.tensors
        order = torch.randperm(
            len(labels), generator=torch.Generator().manual_seed(1234)
        )
        updated = []
        original = trainer.optimizer.step
        global_batch = trainer.config.global_batch_for(batch)

        def checked_step():
            start = len(updated) * global_batch
            ids = order[start : start + global_batch]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(
                reference(features[ids].to(trainer.device)),
                labels[ids].to(trainer.device),
            ).backward()
            for actual, expected in zip(
                trainer.model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(
                    actual.grad, expected.grad, rtol=5e-4, atol=3e-6
                )
            original()
            optimizer.step()
            for actual, expected in zip(
                trainer.model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(actual, expected, rtol=5e-4, atol=3e-6)
            updated.append(len(ids))

        trainer.optimizer.step = checked_step
        trainer.train_epoch(1)
        assert sum(updated) == len(labels)
    finally:
        cleanup()


@pytest.mark.parametrize("batch", [128, 64, 32, 16])
def test_all_updates_and_unequal_tail_match_global_batch_reference(tmp_path, batch):
    mp.spawn(
        gradient_worker, args=(free_port(), str(tmp_path), batch), nprocs=2, join=True
    )
    for rank in range(2):
        frame = pd.read_csv(tmp_path / f"steps_rank{rank}.csv")
        assert frame.ar_count.eq(1).all() and frame.sync_micro_count.eq(1).all()
        assert frame.local_samples.sum() * 2 == 504
        assert not frame.full_batch.iloc[-1]
        runtime = json.loads((tmp_path / f"runtime_rank{rank}.json").read_text())
        assert runtime["device"] == f"cuda:{rank}" and runtime["backend"] == "nccl"


def end_to_end_worker(rank, port, output):
    training = importlib.import_module("E3.train")
    initialize(rank, 2, port)
    try:
        trainer = Trainer(
            Config("mnist", 0.01), 16, 1234, rank, Path(output), split=small_split()
        )
        seen, states = [], []
        original = training.evaluate_model

        def monitored(model, loader):
            measured = original(model, loader)
            assert measured["device"] == "cuda:0"
            seen.append(len(loader.dataset))
            if len(loader.dataset) == 56:
                states.append(copy.deepcopy(model.state_dict()))
                measured["loss"] = len(states)
            else:
                assert len(states) == 20
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, states[-1][key], rtol=0, atol=0)
            return measured

        training.evaluate_model = monitored
        final = trainer.train()
        if rank == 0:
            assert seen == [56] * 20 + [16] and final["best_val_epoch"] == 1
            assert final["total_train_samples"] == 504 * 20
            model, saved = load_checkpoint(final["checkpoint_path"])
            assert saved["epoch"] == 20
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, states[-1][key], rtol=0, atol=0)
    finally:
        cleanup()


def test_twenty_epochs_validate_then_test_last_model(tmp_path):
    mp.spawn(end_to_end_worker, args=(free_port(), str(tmp_path)), nprocs=2, join=True)
