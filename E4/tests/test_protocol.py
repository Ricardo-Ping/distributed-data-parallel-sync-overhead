import ast
import json
from dataclasses import asdict, replace
from pathlib import Path
import pytest
from E4.config import Config
from E4.run import completed, run, schedule


def test_fixed_plan_and_read_only_listing(tmp_path):
    c = Config("mnist", 0.1)
    assert [c.global_batch_for(k) for k in c.levels] == [256, 512, 1024, 2048, 4096]
    assert len(set(schedule(c))) == 30
    run([c], tmp_path / "unused", list_only=True)
    assert not (tmp_path / "unused").exists()
    for values in ({"world_size": 1}, {"epochs": 2}, {"device": "cpu"}):
        with pytest.raises(ValueError):
            replace(c, **values)


def test_framework_is_independent():
    for p in Path(__file__).resolve().parents[1].glob("*.py"):
        imports = [
            n.module or ""
            for n in ast.walk(ast.parse(p.read_text()))
            if isinstance(n, ast.ImportFrom)
        ]
        assert not any(
            m.startswith(("E0", "E1", "E2", "E3", "E5", "src")) for m in imports
        ), p


@pytest.mark.parametrize("missing_rank", [0, 1])
def test_resume_requires_both_memory_records(tmp_path, missing_rank):
    config = Config("mnist", 0.1)
    folder = tmp_path / "run"
    folder.mkdir()
    names = (
        "final.json",
        "checkpoint.pt",
        "config.json",
        "epochs.csv",
        "dataset.json",
        "split_indices.npz",
    )
    names += tuple(
        f"{kind}_rank{rank}.{suffix}"
        for rank in (0, 1)
        for kind, suffix in (
            ("runtime", "json"),
            ("sampling", "json"),
            ("steps", "csv"),
            ("memory", "json"),
        )
    )
    for name in names:
        (folder / name).touch()
    (folder / "config.json").write_text(json.dumps(asdict(config)))
    (folder / "final.json").write_text(
        json.dumps(
            dict(
                experiment="E4",
                dataset="mnist",
                level=1,
                variant="window_end",
                seed=1234,
                epochs_ran=20,
                test_epoch=20,
                selection="last",
                ws=2,
            )
        )
    )
    assert completed(folder, config, 1, "window_end", 1234)
    (folder / f"memory_rank{missing_rank}.json").unlink()
    with pytest.raises(FileExistsError, match="不完整"):
        completed(folder, config, 1, "window_end", 1234)
