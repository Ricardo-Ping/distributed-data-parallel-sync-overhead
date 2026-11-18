"""真正的CUDA/NCCL检查：完整梯度归约、等价更新、末轮测试隔离。"""

import copy
import json
import socket
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from E1.config import Config, PRESETS
from E1.distributed import cleanup, cuda_device, initialize
from E1.model import MLP, load_checkpoint
from E1.train import Trainer
from E1.tests.test_protocol import small_split


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _gradient_worker(rank, ws, port, output, preset):
    initialize(rank, ws, port)
    try:
        trainer = Trainer(
            Config("mnist", 0.01),
            preset,
            1234,
            rank,
            ws,
            Path(output),
            split=small_split(),
        )
        reference = copy.deepcopy(trainer.model)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
        features, labels = trainer.loader.dataset.tensors
        order = torch.randperm(
            len(labels), generator=torch.Generator().manual_seed(1234)
        )
        updated = []
        original_step = trainer.optimizer.step

        def checked_step():
            ids = order[len(updated) * 128 * ws : (len(updated) + 1) * 128 * ws]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(
                reference(features[ids].to(trainer.device)),
                labels[ids].to(trainer.device),
            ).backward()
            for actual, expected in zip(
                trainer.model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(
                    actual.grad, expected.grad, rtol=2e-4, atol=2e-6
                )
            original_step()
            optimizer.step()
            for actual, expected in zip(
                trainer.model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
            updated.append(True)

        trainer.optimizer.step = checked_step
        trainer.train_epoch(1)
        assert len(updated) == (2 if ws == 1 else 1)
    finally:
        cleanup()


@pytest.mark.parametrize("preset,ws", [("S", 1), ("S", 2), ("L", 2)])
def test_fixed_local_batch_updates_match_global_reference_on_gpu(tmp_path, preset, ws):
    mp.spawn(
        _gradient_worker,
        args=(ws, free_port(), str(tmp_path), preset),
        nprocs=ws,
        join=True,
    )
    for rank in range(ws):
        steps = pd.read_csv(tmp_path / f"steps_rank{rank}.csv")
        assert steps.samples.eq(128).all() and steps.ar_count.eq(int(ws > 1)).all()
        runtime = json.loads((tmp_path / f"runtime_rank{rank}.json").read_text())
        assert runtime["device"] == f"cuda:{rank}"
        assert runtime["backend"] == ("nccl" if ws > 1 else "none")


def test_validation_each_epoch_then_test_once_and_load_last_model(
    tmp_path, monkeypatch
):
    import E1.train as training

    initialize(0, 1, 0)
    try:
        trainer = Trainer(
            Config("mnist", 0.01), "S", 1234, 0, 1, tmp_path, split=small_split()
        )
        evaluate = training.evaluate_model
        seen, states = [], []

        def checked_evaluation(model, loader):
            measured = evaluate(model, loader)
            seen.append(len(loader.dataset))
            assert measured["device"] == "cuda:0"
            if len(loader.dataset) == 30:
                states.append(copy.deepcopy(model.state_dict()))
                measured["loss"] = float(len(states))
            else:
                assert len(states) == 20
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, states[-1][key], rtol=0, atol=0)
            return measured

        monkeypatch.setattr(training, "evaluate_model", checked_evaluation)
        final = trainer.train()
        assert seen == [30] * 20 + [16] and final["best_val_epoch"] == 1
        assert final["total_updates"] == 40 and final["measured_updates"] == 30
        loaded, saved = load_checkpoint(final["checkpoint_path"])
        assert saved["epoch"] == 20
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, states[-1][key], rtol=0, atol=0)
    finally:
        cleanup()


def test_three_sizes_keep_architecture_and_parameter_payload_consistent():
    cuda_device()
    for dataset_dim in (784, 3072):
        previous = 0
        for preset, (first, second) in PRESETS.items():
            model = MLP(dataset_dim, preset).cuda()
            info = model.size_info()
            expected = (
                (dataset_dim + 1) * first + (first + 1) * second + (second + 1) * 10
            )
            assert (
                info["parameters"] == expected
                and info["gradient_bytes"] == 4 * expected
            )
            assert first == 2 * second and info["parameters"] > previous
            previous = info["parameters"]
            assert model(torch.randn(128, dataset_dim, device="cuda")).shape == (
                128,
                10,
            )
