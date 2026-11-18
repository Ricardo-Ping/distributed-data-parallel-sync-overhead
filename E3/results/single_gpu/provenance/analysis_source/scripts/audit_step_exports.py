"""重新读取压缩CSV，独立核对逐步吞吐、同步字段、行数和文件哈希。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def truth(column):
    text = column.astype(str).str.lower()
    if not text.isin(("true", "false", "1", "0")).all():
        raise ValueError("布尔字段无法解析")
    return text.isin(("true", "1"))


def audit(experiment):
    folder = ROOT / experiment / "results/step_metrics"
    record = json.loads((folder / "audit.json").read_text())
    index = pd.read_csv(folder / "run_index.csv")
    totals = {}
    for filename, sample, throughput, expected_key in (
        ("rank_steps.csv.gz", "local_samples", "local_throughput_sps", "rank_step_rows"),
        ("global_units.csv.gz", "global_samples", "global_throughput_sps", "global_unit_rows"),
    ):
        path = folder / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["derived_files_sha256"][filename]:
            raise ValueError("压缩逐步文件哈希变化")
        rows, sources, counts = 0, set(), {}
        types = {"record_measured_flag": "string"} if filename.startswith("rank") else None
        for frame in pd.read_csv(path, chunksize=50000, dtype=types):
            for key in (sample, "latency_ms", throughput, "sync_ms", "sync_share"):
                if not np.isfinite(frame[key]).all():
                    raise ValueError(f"{filename}/{key}存在无效数值")
            if not frame[sample].ge(1).all() or not frame.latency_ms.gt(0).all():
                raise ValueError("样本数或耗时不为正")
            np.testing.assert_allclose(frame[throughput], frame[sample] * 1000 / frame.latency_ms, rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(frame.sync_share, frame.sync_ms / frame.latency_ms, rtol=1e-10, atol=1e-12)
            if not frame.sync_share.between(0, 1 + 1e-8).all():
                raise ValueError("同步占比越界")
            if filename.startswith("rank"):
                ar = frame.ar_count
                np.testing.assert_array_equal(truth(frame.cross_gpu_sync), ar.gt(0))
                if not frame.loc[ar.eq(0), "logical_payload_bytes"].eq(0).all():
                    raise ValueError("无归约步出现跨卡载荷")
                single = frame.physical_gpus.eq(1)
                if not ar[single].eq(0).all() or not frame.loc[single, "sync_ms"].eq(0).all():
                    raise ValueError("单GPU步出现跨卡同步")
                if not frame.loc[ar.gt(0), "sync_type"].isin(("gradient_allreduce", "parameter_allreduce")).all():
                    raise ValueError("同步事件类型不符")
            else:
                truth(frame.selected_for_primary_summary)
            sources.update(frame.source_file.unique())
            for identity, count in frame.groupby(["dataset", "run_id"]).size().items():
                counts[identity] = counts.get(identity, 0) + int(count)
            rows += len(frame)
        assert rows == record[expected_key] == int(index[expected_key].sum())
        assert set(counts) == set(zip(index.dataset, index.run_id))
        for row in index.to_dict("records"):
            assert counts[row["dataset"], row["run_id"]] == row[expected_key]
        for source in sources:
            if not (ROOT / source).is_file():
                raise FileNotFoundError(source)
        totals[filename] = {"rows_checked": rows, "source_files": len(sources), "sha256": digest}
    result = {"experiment": experiment, "runs_checked": len(index), "csv_roundtrip_checked": True,
              "throughput_formula_checked": True, "synchronization_fields_checked": True,
              "all_sources_present": True, "files": totals}
    (folder / "roundtrip_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(experiment, len(index), "runs: CSV roundtrip and synchronization fields passed", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E0", "E1", "E2", "E3", "E4", "E5", "all"), default="all")
    args = parser.parse_args()
    for experiment in ("E0", "E1", "E2", "E3", "E4", "E5") if args.experiment == "all" else (args.experiment,):
        audit(experiment)
