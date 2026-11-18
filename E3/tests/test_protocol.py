import ast
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from E3.config import BATCHES, Config, run_folder
from E3.metrics import summarize_steps
from E3.run import check_output, run, schedule


def test_fixed_two_gpus_and_batch_plan_is_read_only(tmp_path):
    config = Config("mnist", 0.1)
    assert [config.global_batch_for(batch) for batch in BATCHES] == [256, 256, 256, 256]
    assert [config.accumulation_for(batch) for batch in BATCHES] == [1, 2, 4, 8]
    assert len(set(schedule(config))) == 12
    run([config], tmp_path / "unused", list_only=True)
    assert not (tmp_path / "unused").exists()
    for values in ({"world_size": 1}, {"epochs": 2}, {"device": "meta"}):
        with pytest.raises(ValueError):
            replace(config, **values)
    run_folder(tmp_path, config, 32, 1234).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        check_output(config, [(32, 1234)], tmp_path)


def test_critical_rank_full_update_filter_and_constant_communication_per_update():
    rows = []
    for iteration, size in enumerate((128, 128, 120)):
        for rank in (0, 1):
            extra = float(rank)
            rows.append(
                dict(
                    iteration=iteration,
                    rank=rank,
                    full_batch=size == 128,
                    measured=iteration == 1,
                    local_samples=size,
                    micro_count=1,
                    sync_micro_count=1,
                    t_data_ms=1,
                    t_compute_ms=2,
                    t_sync_ms=1,
                    t_opt_ms=1,
                    t_other_ms=extra,
                    t_update_ms=5 + extra,
                    ar_count=1,
                    ar_bytes=400,
                    loss=1,
                )
            )
    frame = pd.DataFrame(rows)
    values = summarize_steps(frame, 2, 128, 1, 400, 1)
    assert values["measured_updates"] == 1 and values["total_train_samples"] == 752
    assert values["avg_update_ms"] == 6 and values["train_throughput"] == pytest.approx(
        256000 / 6
    )
    assert values["local_compute_throughput"] == 64000
    for key, value in [
        ("ar_count", 2),
        ("sync_micro_count", 2),
        ("ar_bytes", 200),
        ("measured", True),
    ]:
        corrupt = frame.copy()
        corrupt.loc[0, key] = value
        with pytest.raises(ValueError):
            summarize_steps(corrupt, 2, 128, 1, 400, 1)


def test_runtime_is_independent_of_other_experiment_frameworks():
    for path in Path(__file__).resolve().parents[1].glob("*.py"):
        tree = ast.parse(path.read_text())
        modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert all(
            not module.startswith(("E0", "E1", "E2", "src")) for module in modules
        ), path
