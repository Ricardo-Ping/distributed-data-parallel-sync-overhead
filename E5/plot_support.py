"""本实验主图的数据配对、曲线绘制与来源核验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from E5.config import DATASETS, EXPERIMENT, RESULTS_ROOT, SEEDS, load_config
from E5.single_gpu.config import (
    cases_for,
    reference_config,
    reference_folders,
    saved_config,
)

NAMES = {"mnist": "MNIST", "fashionmnist": "FashionMNIST", "cifar10": "CIFAR-10"}
MARKERS = ("o", "s", "^")
SIZES = (8, 6, 4)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison_frame(dual, results, single_results=None):
    results = Path(results)
    single_results = Path(single_results) if single_results else results / "single_gpu"
    singles, doubles, seen, sources = [], [], set(), {}
    for case in cases_for():
        if case.role == "ordinary_sgd":
            continue
        folder = case.folder(single_results)
        actual = json.loads((folder / "config.json").read_text())
        if actual != json.loads(json.dumps(saved_config(case))):
            raise ValueError(f"单GPU实际配置不匹配：{folder}")
        record = json.loads((folder / "final.json").read_text())
        if (
            record["run_id"],
            record["ws"],
            record["epochs_ran"],
            record["test_epoch"],
            record["selection"],
            record["global_batch"],
            record["seed"],
            record["baseline_role"],
        ) != (
            case.identifier,
            1,
            20,
            20,
            "last",
            case.global_batch,
            case.seed,
            case.role,
        ):
            raise ValueError(f"单GPU运行身份不匹配：{folder}")
        for field in (
            "total_gradient_allreduces",
            "total_model_allreduces",
            "total_logical_payload_bytes",
            "avg_sync_ms",
            "sync_share",
        ):
            if record[field] != 0:
                raise ValueError(f"单GPU出现跨卡通信：{folder}/{field}")
        point = case.global_batch if EXPERIMENT == "E2" else case.reference_level
        single = dict(
            record,
            series="single",
            plot_x=point,
            reference_level=case.reference_level,
            source_file=str(folder / "final.json"),
        )
        single["local_compute_throughput"] = (
            record["measured_samples"]
            / record["measured_updates"]
            * 1000
            / record["avg_compute_ms"]
        )
        single["sync_ms_per_sample"] = single["sync_us_per_sample"] = 0.0
        single["arrival_gap_ms"] = np.nan
        singles.append(single)
        for name in ("config.json", "final.json"):
            sources[str(folder / name)] = digest(folder / name)
        config = reference_config(case.dataset)
        for standard in reference_folders(config, case):
            reference = results / standard.relative_to(RESULTS_ROOT)
            if load_config(reference / "config.json") != config:
                raise ValueError(f"双GPU实际配置不匹配：{reference}")
            current = json.loads((reference / "final.json").read_text())
            memories = []
            for rank in (0, 1):
                memory_path = reference / f"memory_rank{rank}.json"
                memories.append(json.loads(memory_path.read_text())["allocated_bytes"])
                sources[str(memory_path)] = digest(memory_path)
            peak = max(memories) / 1e6
            if "peak_allocated_mb" in current and current["peak_allocated_mb"] != peak:
                raise ValueError("末轮汇总与实际训练显存记录不同")
            current["peak_allocated_mb"] = peak
            if current["run_id"] in seen:
                raise ValueError("双GPU运行重复配对")
            original = dual[dual.run_id.eq(current["run_id"])]
            if (
                len(original) != 1
                or current["ws"] != 2
                or current["global_batch"] != case.global_batch
            ):
                raise ValueError("双GPU网格不完整或全局Batch不匹配")
            for field in (
                "dataset",
                "seed",
                "train_samples",
                "val_samples",
                "test_samples",
                "parameters",
                "gradient_bytes",
                "measured_samples",
                "measured_updates",
                "total_train_samples",
                "total_updates",
            ):
                if current[field] != record[field]:
                    raise ValueError(f"单双GPU工作量不匹配：{reference}/{field}")
            doubles.append(
                dict(
                    current,
                    series=current.get("variant", "dual"),
                    plot_x=point,
                    reference_level=case.reference_level,
                    source_file=str(reference / "final.json"),
                )
            )
            seen.add(current["run_id"])
            for name in ("config.json", "final.json"):
                sources[str(reference / name)] = digest(reference / name)
    if seen != set(dual.run_id):
        raise ValueError("匹配的双GPU集合与主实验不同")
    frame = pd.DataFrame(singles + doubles)
    if frame.duplicated(["dataset", "series", "plot_x", "seed"]).any():
        raise ValueError("主图数据重复")
    for _, group in frame.groupby(["dataset", "series", "plot_x"]):
        if sorted(group.seed) != list(SEEDS):
            raise ValueError("主图配置缺少训练种子")
    frame.to_csv(results / "main_comparison.csv", index=False)
    (results / "main_comparison_sources.json").write_text(
        json.dumps(
            {
                "files_sha256": sources,
                "single_runs": len(singles),
                "dual_runs": len(doubles),
                "same_measured_workload": True,
            },
            indent=2,
        )
        + "\n"
    )
    return frame


def legend_handles(groups):
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=color,
            linewidth=2.2,
            linestyle="--" if name.startswith("single") else "-",
            label=label,
        )
        for name, color, label in groups
    ]
    handles += [
        Line2D(
            [],
            [],
            color="#666666",
            marker=marker,
            markerfacecolor="none",
            linestyle="none",
            label=f"种子 {seed}",
        )
        for seed, marker in zip(SEEDS, MARKERS)
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="#333333",
            marker="D",
            markerfacecolor="none",
            linewidth=2.2,
            label="粗线/菱形：三种子均值",
        )
    )
    return handles


def _line(axis, x, y, metadata, **style):
    (line,) = axis.plot(x, y, **style)
    line.set_gid(json.dumps(metadata, sort_keys=True))
    return line


def draw(axis, frame, dataset, metric, groups, scale=1, *, censored=False, dense=False):
    observed = []
    missing = []
    for name, color, label in groups:
        current = frame[frame.dataset.eq(dataset) & frame.series.eq(name)]
        if current.empty:
            raise ValueError(f"缺少绘图组：{dataset}/{name}")
        if current[metric].isna().all() and not censored:
            if EXPERIMENT == "E5" and metric == "arrival_gap_ms" and name == "single":
                missing.append(label)
                continue
            raise ValueError(f"缺少实测指标：{dataset}/{name}/{metric}")
        if not censored and current[metric].isna().any():
            raise ValueError(f"指标不完整：{dataset}/{name}/{metric}")
        means = current.groupby("plot_x")[metric].mean().sort_index()
        if censored:
            reached = current.groupby("plot_x").target_reached.all()
            means = means.where(reached)
        for seed, marker, size in zip(SEEDS, MARKERS, SIZES):
            values = current[current.seed.eq(seed)].sort_values("plot_x")
            meta = dict(
                dataset=dataset,
                series=name,
                seed=seed,
                metric=metric,
                scale=scale,
                statistic="seed",
            )
            _line(
                axis,
                values.plot_x,
                values[metric] * scale,
                meta,
                color=color,
                marker=marker,
                markerfacecolor="none",
                markersize=3
                if dense
                else size + (2 if name.startswith("single") else 0),
                markevery=4 if dense else None,
                linewidth=0.8,
                linestyle=":" if name.startswith("single") else "-",
                alpha=0.45,
                clip_on=False,
            )
            observed.extend(values[metric].dropna() * scale)
            if censored:
                missed = values[~values.target_reached]
                if not missed.empty:
                    meta = dict(
                        dataset=dataset,
                        series=name,
                        seed=seed,
                        metric="training_stage_wall_s",
                        scale=1,
                        statistic="censored",
                    )
                    _line(
                        axis,
                        missed.plot_x,
                        missed.training_stage_wall_s,
                        meta,
                        color=color,
                        marker="x",
                        linestyle="none",
                        markersize=7,
                        clip_on=False,
                    )
                    observed.extend(missed.training_stage_wall_s)
        meta = dict(
            dataset=dataset,
            series=name,
            metric=metric,
            scale=scale,
            statistic="reached_mean" if censored else "mean",
        )
        _line(
            axis,
            means.index,
            means * scale,
            meta,
            color=color,
            marker="D",
            markerfacecolor="none",
            markersize=5 if dense else (12 if name.startswith("single") else 8),
            markevery=4 if dense else None,
            linewidth=3.2 if name.startswith("single") else 2.2,
            linestyle="--" if name.startswith("single") else "-",
            clip_on=False,
        )
    if not observed:
        raise ValueError("没有可绘制的数据")
    low, high = min(observed), max(observed)
    if metric in ("test_acc", "val_acc"):
        padding = max(0.3, (high - low) * 0.18)
        axis.set_ylim(max(0, low - padding), min(100, high + padding))
    else:
        upper = max(high * (1.3 if censored else 1.15), 1e-9)
        axis.set_ylim(
            0,
            min(100, upper) if metric in ("sync_share", "gpu_gap_fraction") else upper,
        )
    axis.set_title(f"({'abc'[DATASETS.index(dataset)]}) {NAMES[dataset]}")
    axis.grid(alpha=0.17)
    if missing:
        axis.text(
            0.02,
            0.96,
            "仅双GPU定义该指标",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="#646C76",
            fontsize=9,
        )
    return missing


def save_figure(figure, path, source):
    """重新读取源CSV，核对每条种子/均值线及坐标范围后保存图片。"""
    source = Path(source)
    frame = pd.read_csv(source)
    points = curves = 0
    coverage = {}
    for axis in figure.axes:
        for line in axis.lines:
            if not line.get_gid():
                continue
            meta = json.loads(line.get_gid())
            key = (id(axis), meta["dataset"], meta["series"], meta["metric"])
            entry = coverage.setdefault(key, {"seeds": set(), "means": 0})
            if meta["statistic"] == "seed":
                entry["seeds"].add(meta["seed"])
            elif meta["statistic"] in ("mean", "reached_mean"):
                entry["means"] += 1
            group = frame[
                frame.dataset.eq(meta["dataset"]) & frame.series.eq(meta["series"])
            ]
            metric, scale = meta["metric"], meta["scale"]
            if meta["statistic"] in ("seed", "censored"):
                group = group[group.seed.eq(meta["seed"])].sort_values("plot_x")
                if meta["statistic"] == "censored":
                    group = group[~group.target_reached]
                x, y = group.plot_x.to_numpy(), group[metric].to_numpy() * scale
            else:
                mean = group.groupby("plot_x")[metric].mean().sort_index()
                if meta["statistic"] == "reached_mean":
                    mean = mean.where(group.groupby("plot_x").target_reached.all())
                x, y = mean.index.to_numpy(), mean.to_numpy() * scale
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_allclose(
                line.get_ydata(), y, rtol=1e-12, atol=1e-12, equal_nan=True
            )
            finite = np.asarray(y)[np.isfinite(y)]
            if finite.size and (
                finite.min() < axis.get_ylim()[0] or finite.max() > axis.get_ylim()[1]
            ):
                raise ValueError("曲线超出坐标范围")
            points += len(finite)
            curves += 1
    if not curves:
        raise ValueError("图片缺少可核验的种子和均值曲线")
    for key, entry in coverage.items():
        if entry["means"] or entry["seeds"]:
            if entry["seeds"] != set(SEEDS) or entry["means"] != 1:
                raise ValueError(f"图片缺少三个种子或唯一均值：{key}")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    return dict(
        figure=Path(path).name,
        source=source.name,
        source_sha256=digest(source),
        sha256=digest(path),
        curves=curves,
        points=points,
        all_points_inside_axes=True,
        all_three_seeds_and_means=True,
        source_values_checked=True,
    )
