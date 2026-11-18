import copy
import socket
from pathlib import Path
import pandas as pd
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from Data.datasets import ImageData
from E4.config import Config
from E4.data import Split, stratified_indices
from E4.distributed import initialize, cleanup
from E4.train import Trainer


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


def gradient_worker(rank, port, folder, level, variant):
    initialize(rank, 2, port, 4)
    try:
        trainer = Trainer(
            Config("mnist", 0.01),
            level,
            variant,
            1234,
            rank,
            Path(folder),
            split=split_data(),
        )
        reference = copy.deepcopy(trainer.model)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
        features, labels = trainer.loader.dataset.tensors
        order = torch.randperm(
            len(labels), generator=torch.Generator().manual_seed(1234)
        )
        original = trainer.optimizer.step
        updated = []
        gb = trainer.config.global_batch_for(level)

        def checked_step():
            reference.load_state_dict(trainer.model.state_dict())
            optimizer.load_state_dict(copy.deepcopy(trainer.optimizer.state_dict()))
            optimizer.zero_grad(set_to_none=True)
            ids = order[len(updated) * gb : (len(updated) + 1) * gb]
            F.cross_entropy(
                reference(features[ids].to(trainer.device)),
                labels[ids].to(trainer.device),
            ).backward()
            actual = torch.cat([p.grad.flatten() for p in trainer.model.parameters()])
            expected = torch.cat([p.grad.flatten() for p in reference.parameters()])
            assert ((actual - expected).norm() / expected.norm()).item() < 2e-3
            original()
            optimizer.step()
            for a, b in zip(trainer.model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, rtol=5e-4, atol=3e-6)
            updated.append(len(ids))

        trainer.optimizer.step = checked_step
        trainer.train_epoch(1)
        assert sum(updated) == len(labels) == 4104
    finally:
        cleanup()


@pytest.mark.parametrize("level", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("variant", ["every_micro", "window_end"])
def test_weighted_gradients_single_update_and_short_tail(tmp_path, level, variant):
    mp.spawn(
        gradient_worker,
        args=(free_port(), str(tmp_path), level, variant),
        nprocs=2,
        join=True,
    )
    for rank in (0, 1):
        f = pd.read_csv(tmp_path / f"steps_rank{rank}.csv")
        assert f.local_samples.sum() == 2052 and f.micro_count.sum() == 17
        assert f.ar_count.sum() == (17 if variant == "every_micro" else len(f))
        assert f.micro_count.iloc[-1] == 1 and f.local_samples.iloc[-1] == 4


def last_worker(rank, port, folder):
    import E4.train as training

    initialize(rank, 2, port, 4)
    try:
        trainer = Trainer(
            Config("mnist", 0.01),
            1,
            "window_end",
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
