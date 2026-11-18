import copy
import socket
from pathlib import Path
import pandas as pd
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
from Data.datasets import ImageData
from E5.config import Config
from E5.data import Split, stratified_indices
from E5.distributed import initialize, cleanup
from E5.train import Trainer


def split_data(size=4560):
    gen = torch.Generator().manual_seed(7)
    raw = ImageData(
        "mnist",
        torch.randn(size, 12, generator=gen),
        torch.arange(size) % 10,
        torch.randn(16, 12, generator=gen),
        torch.arange(16) % 10,
    )
    return Split(raw, *stratified_indices(raw.y_train, 0.1, 1234))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def local_worker(rank, port, folder, level):
    initialize(rank, 2, port, 4)
    try:
        trainer = Trainer(
            Config("mnist", 0.01, 0.97),
            level,
            "local_sgd",
            1234,
            rank,
            Path(folder),
            split=split_data(),
        )
        refs = [copy.deepcopy(trainer.model) for _ in (0, 1)]
        optimizers = [
            torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9) for m in refs
        ]
        features, labels = trainer.loader.dataset.tensors
        order = torch.randperm(
            len(labels), generator=torch.Generator().manual_seed(1234)
        )
        calls = []
        averages = []
        original_update = trainer.local_update
        original_average = trainer.averager.average

        def update(x, y):
            result = original_update(x, y)
            start = len(calls) * 256
            ids = order[start : start + 256]
            for r in (0, 1):
                local = ids[r::2]
                optimizers[r].zero_grad(set_to_none=True)
                F.cross_entropy(
                    refs[r](features[local].to(trainer.device)),
                    labels[local].to(trainer.device),
                ).backward()
                optimizers[r].step()
            for a, b in zip(trainer.model.parameters(), refs[rank].parameters()):
                torch.testing.assert_close(a, b, rtol=5e-5, atol=3e-6)
            calls.append(len(ids))
            return result

        def average(**kwargs):
            result = original_average(**kwargs)
            with torch.no_grad():
                for a, b in zip(refs[0].parameters(), refs[1].parameters()):
                    mean = (a + b) / 2
                    a.copy_(mean)
                    b.copy_(mean)
            for a, b in zip(trainer.model.parameters(), refs[rank].parameters()):
                torch.testing.assert_close(a, b, rtol=5e-5, atol=3e-6)
            for a, b in zip(trainer.model.parameters(), refs[rank].parameters()):
                torch.testing.assert_close(
                    trainer.optimizer.state[a]["momentum_buffer"],
                    optimizers[rank].state[b]["momentum_buffer"],
                    rtol=5e-5,
                    atol=3e-6,
                )
            averages.append(len(calls))
            return result

        trainer.local_update = update
        trainer.averager.average = average
        trainer.train_epoch(1)
        assert sum(calls) == 4104 and len(calls) == 17
        assert averages == list(range(level, 17, level)) + [17]
    finally:
        cleanup()


@pytest.mark.parametrize("level", [1, 2, 4, 8, 16])
def test_local_updates_parameter_average_and_unsynchronized_momentum(tmp_path, level):
    mp.spawn(
        local_worker, args=(free_port(), str(tmp_path), level), nprocs=2, join=True
    )
    for rank in (0, 1):
        f = pd.read_csv(tmp_path / f"rounds_rank{rank}.csv")
        assert f.local_samples.sum() == 2052 and f.steps_count.sum() == 17
        assert f.ar_count.eq(1).all() and f.drift_measured.sum() == 1
        local = pd.read_csv(tmp_path / f"steps_rank{rank}.csv")
        assert len(local) == 17 and local.local_samples.sum() == 2052
        expected = [(step + 1) % level == 0 or step == 16 for step in range(17)]
        assert np.array_equal(local.sync_happened, expected)
        assert np.array_equal(local.ar_count, np.asarray(expected, dtype=int))
        assert np.allclose(
            local.local_throughput, local.local_samples * 1000 / local.t_update_ms
        )
        sums = local.groupby("round_iteration").sum(numeric_only=True)
        assert np.allclose(sums.t_update_ms, f.t_round_ms, rtol=1e-9, atol=1e-7)
        for field in (
            "t_data_ms",
            "t_compute_ms",
            "t_sync_ms",
            "t_opt_ms",
            "t_other_ms",
            "ar_count",
            "ar_bytes",
        ):
            assert np.allclose(sums[field], f[field], rtol=1e-9, atol=1e-7)


def last_worker(rank, port, folder):
    import E5.train as training

    initialize(rank, 2, port, 4)
    try:
        trainer = Trainer(
            Config("mnist", 0.01, 0.97),
            1,
            "local_sgd",
            1234,
            rank,
            Path(folder),
            split=split_data(560),
        )
        original = training.evaluate_model
        seen = []
        states = []

        def evaluate(model, loader):
            metrics = original(model, loader)
            assert metrics["device"] == "cuda:0"
            seen.append(len(loader.dataset))
            if len(loader.dataset) == 56:
                states.append(copy.deepcopy(model.state_dict()))
                metrics["loss"] = len(states)
            else:
                assert len(states) == 20
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, states[-1][k], rtol=0, atol=0)
            return metrics

        training.evaluate_model = evaluate
        final = trainer.train()
        if rank == 0:
            assert seen == [56] * 20 + [16] and final["epochs_ran"] == 20
    finally:
        cleanup()


def test_only_last_epoch_is_tested(tmp_path):
    mp.spawn(last_worker, args=(free_port(), str(tmp_path)), nprocs=2, join=True)
