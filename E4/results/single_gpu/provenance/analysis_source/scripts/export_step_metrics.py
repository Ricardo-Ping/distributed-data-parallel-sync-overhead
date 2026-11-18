"""只读原始CSV，导出显式逐步吞吐与同步指标；不执行训练或生成报告。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mnist", "fashionmnist", "cifar10")
EXPECTED_MAIN = {"E0": 18, "E1": 54, "E2": 45, "E3": 36, "E4": 90, "E5": 45}
RANK_COLUMNS = ["experiment", "dataset", "run_id", "source_kind", "baseline_role", "model", "seed",
                "physical_gpus", "logical_replicas", "rank", "replica", "epoch", "step", "iteration",
                "timing_unit", "local_samples", "latency_ms", "local_throughput_sps", "data_ms", "compute_ms",
                "optimizer_ms", "other_ms", "sync_type",
                "cross_gpu_sync", "sync_ms", "sync_share", "local_average_ms", "local_average_count",
                "ar_count", "logical_payload_bytes", "micro_count", "record_measured_flag", "source_file"]
GLOBAL_COLUMNS = ["experiment", "dataset", "run_id", "source_kind", "baseline_role", "model", "seed",
                  "physical_gpus", "logical_replicas", "epoch", "unit", "iteration", "timing_unit",
                  "optimizer_updates_per_replica", "global_samples", "latency_ms", "global_throughput_sps", "critical_rank",
                  "sync_ms", "sync_share", "local_average_ms", "ar_count_per_rank", "logical_payload_bytes_per_rank",
                  "timing_scope", "selected_for_primary_summary", "source_file"]


def meta(experiment, final, source_kind="main"):
    return {"experiment": experiment, "dataset": final.get("dataset", final.get("data")),
            "run_id": final["run_id"], "source_kind": source_kind,
            "baseline_role": final.get("baseline_role", "none"), "model": final.get("model", "M"),
            "seed": final["seed"], "physical_gpus": final.get("physical_gpus", final["ws"]),
            "logical_replicas": final.get("logical_replicas", final["ws"])}


def bools(values):
    return values.astype(str).str.lower().isin(("true", "1"))


def ranks(frame, info, source, rank=0, epoch_offset=0):
    elapsed = next(key for key in ("t_update_ms", "t_step_ms", "t_unit_ms") if key in frame)
    sample = next(key for key in ("local_samples", "samples", "global_samples") if key in frame)
    time, samples = frame[elapsed].to_numpy(), frame[sample].to_numpy()
    if not np.isfinite(time).all() or (time <= 0).any() or (samples < 1).any():
        raise ValueError(f"无效的逐步耗时或样本数：{source}")
    result = pd.DataFrame({key: [value] * len(frame) for key, value in info.items()})
    result["rank"] = rank
    result["replica"] = frame.replica.to_numpy() if "replica" in frame else rank
    result["epoch"] = frame.epoch.to_numpy() + epoch_offset
    result["step"] = frame["step"].to_numpy() if "step" in frame else frame["unit"].to_numpy()
    result["iteration"] = frame.iteration.to_numpy() if "iteration" in frame else np.arange(len(frame))
    local_sgd = info["experiment"] == "E5" and info["baseline_role"] != "ordinary_sgd"
    result["timing_unit"] = "local_optimizer_update" if local_sgd else "optimizer_update"
    result["local_samples"], result["latency_ms"] = samples, time
    result["local_throughput_sps"] = 1000 * samples / time
    for target, candidates in (("data_ms", ("t_data_ms",)), ("compute_ms", ("t_compute_ms", "t_fwd_bwd_ms")),
                               ("optimizer_ms", ("t_opt_ms",)), ("other_ms", ("t_other_ms",))):
        origin = next((key for key in candidates if key in frame), None)
        result[target] = frame[origin].to_numpy() if origin else None
    if "local_throughput" in frame and not np.allclose(result.local_throughput_sps, frame.local_throughput, rtol=1e-9):
        raise ValueError("已有逐步吞吐与原始计时不一致")
    ar = frame.ar_count.to_numpy()
    local_count = frame.local_average_count.to_numpy() if "local_average_count" in frame else np.zeros(len(frame), dtype=int)
    result["cross_gpu_sync"] = ar > 0
    cross_type = "parameter_allreduce" if local_sgd else "gradient_allreduce"
    result["sync_type"] = np.where(ar > 0, cross_type, np.where(local_count > 0, "local_parameter_average", "none"))
    result["sync_ms"] = frame.t_sync_ms.to_numpy()
    result["sync_share"] = result.sync_ms / result.latency_ms
    result["local_average_ms"] = frame.t_local_average_ms.to_numpy() if "t_local_average_ms" in frame else 0.0
    result["local_average_count"] = local_count
    result["ar_count"], result["logical_payload_bytes"] = ar, frame.ar_bytes.to_numpy()
    result["micro_count"] = frame.micro_count.to_numpy() if "micro_count" in frame else 1
    result["record_measured_flag"] = bools(frame.measured).to_numpy() if "measured" in frame else None
    result["source_file"] = str(Path(source).relative_to(ROOT))
    if result.duplicated(["iteration", "replica"]).any():
        raise ValueError("逐步记录重复")
    return result.reindex(columns=RANK_COLUMNS)


def units(frame, info, source, *, samples, time, unit, count=1, scope, epoch_offset=0, selected=None):
    latency, sample_counts = np.asarray(time), np.asarray(samples)
    if not np.isfinite(latency).all() or (latency <= 0).any() or (sample_counts < 1).any():
        raise ValueError(f"无效全局计时单元：{source}")
    result = pd.DataFrame({key: [value] * len(frame) for key, value in info.items()})
    result["epoch"] = frame.epoch.to_numpy() + epoch_offset
    result["unit"] = np.asarray(unit)
    result["iteration"] = frame.iteration.to_numpy() if "iteration" in frame else np.arange(len(frame))
    result["timing_unit"] = "local_sgd_sync_round" if "round" in scope else "optimizer_update"
    result["optimizer_updates_per_replica"] = count
    result["global_samples"], result["latency_ms"] = sample_counts, latency
    result["global_throughput_sps"] = sample_counts * 1000 / latency
    result["critical_rank"] = frame["rank"].to_numpy() if "rank" in frame else 0
    result["sync_ms"] = frame.t_sync_ms.to_numpy()
    result["sync_share"] = result.sync_ms / result.latency_ms
    result["local_average_ms"] = frame.t_local_average_ms.to_numpy() if "t_local_average_ms" in frame else 0.0
    result["ar_count_per_rank"] = frame.ar_count.to_numpy()
    result["logical_payload_bytes_per_rank"] = frame.ar_bytes.to_numpy()
    result["timing_scope"] = scope
    result["selected_for_primary_summary"] = True if selected is None else np.asarray(selected)
    source_name = str(Path(source).relative_to(ROOT))
    result["source_file"] = [source_name.replace("_rank0.csv", f"_rank{int(rank)}.csv") for rank in result.critical_rank] if scope.startswith("critical_rank") else source_name
    return result.reindex(columns=GLOBAL_COLUMNS)


def current_runs(experiment):
    result = []
    root = ROOT / experiment / "results"
    if experiment == "E0":
        for dataset in DATASETS:
            finals = pd.read_csv(root / dataset / "final.csv")
            for final in finals.to_dict("records"):
                result.append((final, root / dataset))
    else:
        for dataset in DATASETS:
            for path in sorted((root / dataset).rglob("final.json")):
                result.append((json.loads(path.read_text()), path.parent))
    if len(result) != EXPECTED_MAIN[experiment]:
        raise ValueError(f"{experiment}完整主实验应有{EXPECTED_MAIN[experiment]}次，实际{len(result)}次")
    return result


def export(experiment, include_single=False):
    root = ROOT / experiment / "results"
    output = root / "step_metrics"
    output.mkdir(exist_ok=True)
    audit, rank_rows, unit_rows = [], 0, 0
    with gzip.open(output / "rank_steps.csv.gz", "wt", encoding="utf-8", compresslevel=5) as rank_out, \
         gzip.open(output / "global_units.csv.gz", "wt", encoding="utf-8", compresslevel=5) as unit_out:
        def save(rank_frames, global_frame, info, final):
            nonlocal rank_rows, unit_rows
            count = 0
            for raw in rank_frames:
                raw.to_csv(rank_out, index=False, header=rank_rows == 0)
                rank_rows += len(raw)
                count += len(raw)
            global_frame.to_csv(unit_out, index=False, header=unit_rows == 0)
            unit_rows += len(global_frame)
            selected=global_frame[bools(global_frame.selected_for_primary_summary)]
            throughput=selected.global_samples.sum()*1000/selected.latency_ms.sum()
            if not np.isclose(throughput,final['train_throughput'],rtol=1e-8,atol=1e-6):
                raise ValueError(f"显式全局吞吐未还原原始运行汇总：{info['run_id']}")
            audit.append({**info, "rank_step_rows": count, "global_unit_rows": len(global_frame),
                          "explicit_throughput_checked": True,"original_run_throughput_recomputed":True})
        for final, folder in current_runs(experiment):
            info = meta(experiment, final)
            ws = int(final["ws"])
            if experiment == "E0":
                source = folder / "runs.csv"
                all_rows = pd.read_csv(source)
                raw = all_rows[all_rows.run_id.eq(final["run_id"])].sort_values(["epoch", "step"]).reset_index(drop=True)
                local = ranks(raw, info, source, epoch_offset=1)
                global_frame = units(raw, info, source, samples=raw.samples * ws, time=raw.t_step_ms,
                                     unit=raw.step, scope="rank0_observed", epoch_offset=1)
                save([local], global_frame, info, final)
                continue
            local_frames = []
            for rank in range(ws):
                source = folder / f"steps_rank{rank}.csv"
                if not source.is_file():
                    raise ValueError(f"缺少实际逐本地训练步：{source}")
                raw = pd.read_csv(source)
                local_frames.append(ranks(raw, info, source, rank))
            if experiment == "E5":
                source = folder / "rounds_rank0.csv"
                raw = pd.concat([pd.read_csv(folder / f"rounds_rank{rank}.csv") for rank in range(ws)], ignore_index=True)
                chosen = raw.loc[raw.groupby("iteration").t_round_ms.idxmax()].sort_values("iteration").reset_index(drop=True)
                global_frame = units(chosen, info, source, samples=chosen.local_samples * ws, time=chosen.t_round_ms,
                                     unit=chosen["round"], count=chosen.steps_count.to_numpy(), scope="critical_rank_sync_round",
                                     selected=chosen.measured)
            else:
                source = folder / "steps_rank0.csv"
                raw = pd.concat([pd.read_csv(folder / f"steps_rank{rank}.csv") for rank in range(ws)], ignore_index=True)
                time_column = "t_step_ms" if experiment == "E1" else "t_update_ms"
                chosen = raw.loc[raw.groupby("iteration")[time_column].idxmax()].sort_values("iteration").reset_index(drop=True)
                sample_column = "samples" if experiment == "E1" else "local_samples"
                global_frame = units(chosen, info, source, samples=chosen[sample_column] * ws, time=chosen[time_column],
                                     unit=chosen.step, scope="critical_rank_optimizer_update", selected=chosen.measured)
            save(local_frames, global_frame, info, final)
        if include_single and experiment in ("E2", "E3", "E4", "E5"):
            module = importlib.import_module(f"{experiment}.single_gpu.config")
            for case in module.cases_for():
                folder = case.folder()
                final = json.loads((folder / "final.json").read_text())
                info = meta(experiment, final, "single_gpu_baseline")
                filename = getattr(case, "units_filename", "steps.csv")
                source = folder / filename
                raw = pd.read_csv(source)
                if case.role == "serial_local_sgd":
                    local_source = folder / "local_steps.csv"
                    locals_frame = [ranks(pd.read_csv(local_source), info, local_source)]
                    scope = "single_gpu_serial_sync_round"
                else:
                    locals_frame = [ranks(raw, info, source)]
                    scope = "single_gpu_optimizer_update"
                global_frame = units(raw, info, source, samples=raw.global_samples, time=raw.t_unit_ms, unit=raw["unit"],
                                     count=raw.steps_count.to_numpy(), scope=scope, selected=raw.measured)
                save(locals_frame, global_frame, info, final)
    record = {"experiment": experiment, "runs": len(audit), "rank_step_rows": rank_rows, "global_unit_rows": unit_rows,
              "local_throughput_definition": "1000 * actual local samples / actual local optimizer-step ms",
              "global_throughput_definition": "1000 * actual global samples / recorded global timing-unit ms",
              "E0_scope": "original rank0 observations; original millisecond rounding preserved",
              "E5_scope": "each local optimizer step measured independently; H-step global throughput remains round-based",
              "original_files_modified": False, "checked": audit}
    record["derived_files_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                      for name in ("rank_steps.csv.gz", "global_units.csv.gz")}
    (output / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    pd.DataFrame(audit).to_csv(output / "run_index.csv", index=False)
    print(f"{experiment}: {len(audit)} runs, {rank_rows} actual optimizer-step records, {unit_rows} global units", flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*EXPECTED_MAIN, "all"), default="all")
    parser.add_argument("--include-single-gpu", action="store_true")
    args = parser.parse_args()
    for experiment in EXPECTED_MAIN if args.experiment == "all" else (args.experiment,):
        export(experiment, args.include_single_gpu)
