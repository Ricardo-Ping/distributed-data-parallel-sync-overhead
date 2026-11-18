"""E0 的只读列表、覆盖保护和独立配置。"""

import csv
import json
from dataclasses import asdict

import pytest

from E0.config import Config, E0_ROOT, load_config
from E0.distributed import cuda_device
from E0.metrics import CSVWriter, RUN_FIELDS
from E0.run import check_output, run, schedule


def test_paired_order_and_read_only_listing(tmp_path):
    config = load_config(E0_ROOT / "configs/mnist.yaml")
    assert schedule(config) == [
        (1234, 1),
        (1234, 2),
        (1235, 2),
        (1235, 1),
        (1236, 1),
        (1236, 2),
    ]
    run([config], tmp_path / "unused", list_only=True)
    assert not (tmp_path / "unused").exists()


def test_existing_incomplete_run_is_not_appended(tmp_path):
    (tmp_path / "mnist").mkdir()
    with (tmp_path / "mnist/runs.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id"])
        writer.writerow(["e0_baseline_k1_ws1_mM_b256_r0"])
    with pytest.raises(FileExistsError):
        check_output(Config("mnist", 0.1), [(1234, 1)], tmp_path)


def test_protocol_and_device_reject_invalid_values():
    with pytest.raises(ValueError):
        Config("mnist", 0.1, epochs=2)
    with pytest.raises(ValueError):
        cuda_device("meta")


def test_repeats_with_changed_configuration_are_rejected(tmp_path):
    folder = tmp_path / "mnist"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(asdict(Config("mnist", 0.1))))
    with pytest.raises(ValueError, match="不同的配置"):
        check_output(Config("mnist", 0.2), [(1235, 1)], tmp_path)


def test_old_csv_schema_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "runs.csv"
    original = "run_id,old_field\nprevious,7\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="字段"):
        CSVWriter(path, RUN_FIELDS)
    assert path.read_text() == original
