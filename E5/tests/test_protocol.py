import ast
from dataclasses import replace
from pathlib import Path
import pytest
from E5.config import Config
from E5.run import run, schedule
from E5.profile_activity import union_duration, analyze_trace


def test_fixed_plan_and_read_only_listing(tmp_path):
    c = Config("mnist", 0.1, 0.97)
    assert [c.global_batch_for(h) for h in c.levels] == [256] * 5
    assert len(set(schedule(c))) == 15
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
            m.startswith(("E0", "E1", "E2", "E3", "E4", "src")) for m in imports
        ), p


def test_gpu_activity_union_excludes_cpu_and_other_gpu():
    assert union_duration([(0, 4), (2, 7), (9, 11)]) == 9
    events = [
        dict(ph="X", name="E5_measured_region", cat="user_annotation", ts=0, dur=100),
        dict(ph="X", name="gemm", cat="kernel", ts=10, dur=20, args={"device": 0}),
        dict(ph="X", name="nccl", cat="kernel", ts=25, dur=15, args={"device": 0}),
        dict(ph="X", name="other", cat="kernel", ts=0, dur=100, args={"device": 1}),
        dict(ph="X", name="host", cat="cpu_op", ts=0, dur=100),
    ]
    result = analyze_trace(events, 0)
    assert result["gpu_gap_fraction"] == pytest.approx(0.7)
    assert result["nccl_kernel_ms"] == pytest.approx(0.015)
