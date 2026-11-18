"""固定局部批量、原始计时口径和独立目录的协议检查。"""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from Data.datasets import ImageData
from E1.config import Config, PRESETS, run_folder
from E1.data import Split, stratified_indices, train_loader
from E1.metrics import paired_scaling, summarize_steps
from E1.run import check_output, run, schedule


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


def test_schedule_local_batch_and_read_only_list(tmp_path):
    config = Config("mnist", 0.1)
    jobs = schedule(config)
    assert len(jobs) == len(set(jobs)) == 18
    assert {(model, seed, ws) for model, seed, ws in jobs} == {
        (model, seed, ws) for model in PRESETS for seed in config.seeds for ws in (1, 2)
    }
    run([config], tmp_path / "unused", list_only=True)
    assert not (tmp_path / "unused").exists()
    split = small_split()
    loaders = [train_loader(split, config, rank, 2, 1234) for rank in (0, 1)]
    assert all(loader.batch_size == 128 and loader.drop_last for loader in loaders)
    assert set(loaders[0].sampler).isdisjoint(set(loaders[1].sampler))
    assert len(train_loader(split, config, 0, 1, 1234)) == 2
    assert len(loaders[0]) == 1


def test_invalid_protocol_and_existing_partial_run_fail(tmp_path):
    config = Config("mnist", 0.1)
    for values in ({"local_batch": 256}, {"epochs": 2}, {"device": "meta"}):
        with pytest.raises(ValueError):
            replace(config, **values)
    run_folder(tmp_path, "mnist", "S", 1, 1234).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        check_output(config, [("S", 1234, 1)], tmp_path)


def test_critical_rank_is_one_complete_row_and_warmup_is_excluded():
    rows = []
    for iteration in range(3):
        for rank in (0, 1):
            data, compute, sync, opt = (
                (2, 4, 3, 1) if rank == iteration % 2 else (1, 3, 4, 1)
            )
            rows.append(
                dict(
                    iteration=iteration,
                    rank=rank,
                    measured=iteration > 0,
                    samples=128,
                    t_data_ms=data,
                    t_compute_ms=compute,
                    t_sync_ms=sync,
                    t_opt_ms=opt,
                    t_step_ms=data + compute + sync + opt,
                    ar_count=1,
                    ar_bytes=400,
                    loss=1.0,
                )
            )
    frame = pd.DataFrame(rows)
    values = summarize_steps(frame, 2, 128, 400)
    assert values["total_updates"] == 3 and values["measured_updates"] == 2
    assert values["avg_update_ms"] == 10 and values["avg_sync_ms"] == 3
    assert values["sync_share"] == 0.3 and values["sync_compute_ratio"] == 0.75
    assert values["train_throughput"] == 25600
    for field, value in (
        ("samples", 127),
        ("ar_count", 2),
        ("ar_bytes", 399),
        ("t_step_ms", 99),
    ):
        corrupt = frame.copy()
        corrupt.loc[0, field] = value
        with pytest.raises(ValueError):
            summarize_steps(corrupt, 2, 128, 400)
    with pytest.raises(ValueError):
        summarize_steps(frame.iloc[:-1], 2, 128, 400)


def test_scaling_is_paired_per_seed_and_efficiency_divides_by_gpu_count():
    frame = pd.DataFrame(
        [
            dict(
                dataset="mnist",
                model="S",
                seed=seed,
                ws=ws,
                train_throughput=value,
                avg_update_ms=256000 / value if ws == 2 else 128000 / value,
            )
            for seed, a, b in [(1234, 100, 150), (1235, 1000, 1000)]
            for ws, value in [(1, a), (2, b)]
        ]
    )
    paired = paired_scaling(frame)
    assert paired.throughput_scaling.tolist() == [1.5, 1]
    assert paired.weak_scaling_efficiency.mean() == 0.625


def test_e1_does_not_import_other_experiments_or_old_framework():
    import ast

    for path in Path(__file__).resolve().parents[1].glob("*.py"):
        tree = ast.parse(path.read_text())
        modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert all(not name.startswith(("E0", "src")) for name in modules), path
