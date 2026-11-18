from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from E0.check_results import validate_gpu_runtime, validate_traces
from E0.check_results import load_final_runs as e0_final_runs


def complete_fixture(tmp_path):
    train, validation = np.arange(6), np.arange(6, 8)
    np.savez_compressed(
        tmp_path / "split_indices.npz", train=train, validation=validation
    )
    metadata = {
        "train_samples": 6,
        "val_samples": 2,
        "original_train_samples": 8,
        "train_indices_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(validation.tobytes()).hexdigest(),
    }
    steps = pd.DataFrame(
        [
            {
                "run_id": "example",
                "epoch": epoch,
                "step": 0,
                "samples": 6,
                "t_step_ms": 10.0,
                "t_sync_ms": 0,
                "loss": 1.0,
                "ar_count": 0,
            }
            for epoch in range(20)
        ]
    )
    history = pd.DataFrame(
        [
            {
                "run_id": "example",
                "epoch": epoch,
                "val_loss": 1.0 + epoch / 100,
                "val_acc": 0.5 - epoch / 1000,
                "train_loss": 0.8,
                "test_acc": None,
            }
            for epoch in range(20)
        ]
    )
    checkpoint = {
        "format_version": 1,
        "run_id": "example",
        "epoch": 20,
        "selection": "last",
        "metrics": {
            "val_loss": 1.19,
            "val_acc": 0.481,
            "test_loss": 1.2,
            "test_acc": 0.47,
        },
    }
    checkpoint_path = tmp_path / "example_epoch20.pt"
    torch.save(checkpoint, checkpoint_path)
    finals = pd.DataFrame(
        [
            {
                "run_id": "example",
                "epochs_ran": 20,
                "test_epoch": 20,
                "training_mode": "fixed_epochs",
                "stopped_early": False,
                "checkpoint_path": str(checkpoint_path),
                "ws": 1,
                "local_batch": 256,
                "total_train_samples": 120,
                "total_updates": 20,
                "train_time_s": 0.2,
                "avg_update_ms": 10.0,
                "avg_sync_ms": 0.0,
                "sync_share": 0.0,
                "train_throughput": 600.0,
                "best_epoch": 1,
                "best_val_loss": 1.0,
                "best_val_acc": 0.5,
                "train_loss": 0.8,
                "last_val_loss": 1.19,
                "last_val_acc": 0.481,
                "test_loss": 1.2,
                "test_acc": 0.47,
                "model": "M",
                "k": 1,
                "compress": "none",
                "data": "mnist",
                "lr": 0.1,
                "seed": 1234,
            }
        ]
    )
    steps.to_csv(tmp_path / "runs.csv", index=False)
    history.to_csv(tmp_path / "epochs.csv", index=False)
    return metadata, steps, history, finals, checkpoint


@pytest.mark.fast
@pytest.mark.parametrize(
    "corruption",
    [
        "truncated_steps",
        "changed_split",
        "missing_index",
        "early_stopping",
        "wrong_test_epoch",
        "last_val",
        "test_metric",
        "missing_checkpoint",
        "checkpoint_epoch",
        "sync_share",
    ],
)
def test_report_cross_checks_raw_records_and_split_indices(tmp_path, corruption):
    metadata, steps, _, finals, checkpoint = complete_fixture(tmp_path)
    validate_traces(tmp_path, metadata, finals)
    if corruption == "truncated_steps":
        steps.iloc[:-1].to_csv(tmp_path / "runs.csv", index=False)
    elif corruption == "changed_split":
        np.savez_compressed(
            tmp_path / "split_indices.npz",
            train=np.arange(6)[::-1],
            validation=np.arange(6, 8),
        )
    elif corruption == "missing_index":
        np.savez_compressed(
            tmp_path / "split_indices.npz",
            train=np.arange(5),
            validation=np.arange(6, 8),
        )
    elif corruption == "early_stopping":
        finals.loc[0, "training_mode"] = "early_stopping"
    elif corruption == "wrong_test_epoch":
        finals.loc[0, "test_epoch"] = 1
    elif corruption == "last_val":
        finals.loc[0, "last_val_loss"] = 1.0
    elif corruption == "test_metric":
        finals.loc[0, "test_acc"] = 0.99
    elif corruption == "missing_checkpoint":
        (tmp_path / "example_epoch20.pt").unlink()
    elif corruption == "checkpoint_epoch":
        checkpoint["epoch"] = 1
        torch.save(checkpoint, tmp_path / "example_epoch20.pt")
    else:
        finals.loc[0, "sync_share"] = 0.5
    with pytest.raises(ValueError):
        validate_traces(tmp_path, metadata, finals)


@pytest.mark.fast
@pytest.mark.parametrize(
    "field,value",
    [
        ("training_mode", "early_stopping"),
        ("epochs_ran", 19),
        ("test_epoch", 1),
        ("stopped_early", True),
    ],
)
def test_e0_reader_rejects_old_selection_protocol(tmp_path, field, value):
    _, _, _, finals, _ = complete_fixture(tmp_path)
    exp = tmp_path / "e0"
    exp.mkdir()
    finals.to_csv(exp / "final.csv", index=False)
    assert len(e0_final_runs(tmp_path, "e0")) == 1
    finals.loc[0, field] = value
    finals.to_csv(exp / "final.csv", index=False)
    with pytest.raises(ValueError, match="固定 20"):
        e0_final_runs(tmp_path, "e0")


@pytest.mark.fast
@pytest.mark.parametrize(
    "corruption",
    ["invalid_device", "invalid_backend", "async_timing", "wrong_rank_device"],
)
def test_gpu_records_reject_invalid_runtime(tmp_path, corruption):
    import json

    _, steps, history, finals, checkpoint = complete_fixture(tmp_path)
    runtime = {
        "device": "cuda:0",
        "backend": "none",
        "gpu_name": "example GPU",
        "host_threads": 4,
        "timing_mode": "cuda_synchronized_wall",
    }
    for frame in (steps, history, finals):
        for key, value in runtime.items():
            frame[key] = value
    steps["ar_bytes"] = 0
    steps["t_data_ms"], steps["t_fwd_bwd_ms"], steps["t_opt_ms"] = 1.0, 8.0, 1.0
    steps.to_csv(tmp_path / "runs.csv", index=False)
    history.to_csv(tmp_path / "epochs.csv", index=False)
    checkpoint.update(
        training_config={"device": "cuda"}, state_dict={"weight": torch.ones(1)}
    )
    torch.save(checkpoint, tmp_path / "example_epoch20.pt")
    runtime.update(run_id="example", rank=0, local_rank=0, world_size=1, seed=1234)
    folder = tmp_path / "runtime"
    folder.mkdir()
    path = folder / "example_rank0.json"
    path.write_text(json.dumps(runtime))
    validate_gpu_runtime(tmp_path, finals)
    if corruption == "invalid_device":
        finals.loc[0, "device"] = "meta"
    elif corruption == "invalid_backend":
        finals.loc[0, "backend"] = "invalid"
    elif corruption == "async_timing":
        finals.loc[0, "timing_mode"] = "unsynchronized_wall"
    else:
        runtime["device"] = "cuda:1"
        path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError):
        validate_gpu_runtime(tmp_path, finals)
