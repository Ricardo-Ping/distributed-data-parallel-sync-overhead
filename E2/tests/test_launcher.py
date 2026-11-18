"""启动失败仅在训练开始前重试；续跑禁止覆盖不完整结果。"""

import importlib
import json
from dataclasses import asdict

import pytest

from E2.config import Config

launcher = importlib.import_module("E2.run")


def test_address_in_use_retries_before_training_only(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    attempts = []

    def spawn(*args, **kwargs):
        attempts.append(kwargs["args"][0])
        if len(attempts) == 1:
            raise launcher.mp.ProcessRaisedException("EADDRINUSE", 0, 42)

    monkeypatch.setattr(launcher.mp, "spawn", spawn)
    launcher.launch(Config("mnist", 0.1), 128, 1234, tmp_path, tmp_path)
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "message,started", [("EADDRINUSE", True), ("CUDA error", False)]
)
def test_other_failures_and_started_training_are_never_retried(
    tmp_path, monkeypatch, message, started
):
    if started:
        (tmp_path / "runtime_rank0.json").write_text("{}")
    attempts = []

    def spawn(*args, **kwargs):
        attempts.append(1)
        raise launcher.mp.ProcessRaisedException(message, 0, 42)

    monkeypatch.setattr(launcher.mp, "spawn", spawn)
    with pytest.raises(launcher.mp.ProcessRaisedException):
        launcher.launch(Config("mnist", 0.1), 128, 1234, tmp_path, tmp_path)
    assert len(attempts) == 1


def test_resume_refuses_partial_folder_and_skips_completed_folder(tmp_path):
    config = Config("mnist", 0.1)
    folder = tmp_path / "one_run"
    assert not launcher.completed(folder, config, 128, 1234)
    folder.mkdir()
    with pytest.raises(FileExistsError):
        launcher.completed(folder, config, 128, 1234)
    for name in (
        "checkpoint.pt",
        "epochs.csv",
        "steps_rank0.csv",
        "steps_rank1.csv",
        "runtime_rank0.json",
        "runtime_rank1.json",
        "sampling_rank0.json",
        "sampling_rank1.json",
        "memory_rank0.json",
        "memory_rank1.json",
        "dataset.json",
        "split_indices.npz",
    ):
        (folder / name).touch()
    (folder / "config.json").write_text(json.dumps(asdict(config)))
    final = dict(
        experiment="E2",
        dataset="mnist",
        micro_batch=128,
        seed=1234,
        epochs_ran=20,
        test_epoch=20,
        selection="last",
        ws=2,
        global_batch=256,
        accumulation=1,
        model="M",
    )
    (folder / "final.json").write_text(json.dumps(final))
    assert launcher.completed(folder, config, 128, 1234)
    final["epochs_ran"] = 19
    (folder / "final.json").write_text(json.dumps(final))
    with pytest.raises(ValueError):
        launcher.completed(folder, config, 128, 1234)
