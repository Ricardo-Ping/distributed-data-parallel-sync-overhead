"""E5单/双GPU对照图；保留严格同算法和普通SGD两种参考。"""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from E5.config import DATASETS, RESULTS_ROOT, SEEDS
from E5.single_gpu.config import reference_config

NAMES = {"mnist": "MNIST", "fashionmnist": "FashionMNIST", "cifar10": "CIFAR-10"}
GROUPS = (
    ("serial", "#646C76", "单GPU：串行双副本 Local SGD"),
    ("dual", "#2766A6", "双GPU：Local SGD"),
    ("ordinary", "#21876A", "单GPU：普通 SGD（batch256）"),
)


def render(results_dir=RESULTS_ROOT):
    results_dir = Path(results_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    for font in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        try:
            font_manager.findfont(font, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams.update({"font.family": [font], "axes.unicode_minus": False})
        break
    else:
        raise RuntimeError("需要中文字体")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    strict = pd.read_csv(results_dir / "scaling.csv")
    ordinary = pd.read_csv(results_dir / "ordinary_sgd_reference.csv")
    frame = strict.merge(
        ordinary,
        on=["dataset", "level", "seed"],
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    if len(frame) != 45 or not np.allclose(
        frame.dual_throughput, frame.dual_throughput_reference
    ):
        raise ValueError("严格对照与普通SGD参考的双卡记录不匹配")
    levels = [1, 2, 4, 8, 16]
    markers, sizes = ("o", "s", "^"), (8, 6, 4)
    seed_handles = [
        Line2D(
            [],
            [],
            color="#666666",
            marker=marker,
            markerfacecolor="none",
            linestyle="none",
            label=f"种子 {seed}",
        )
        for seed, marker in zip(SEEDS, markers)
    ]
    seed_handles.append(
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
    group_handles = [
        Line2D([], [], color=color, linewidth=2.2, label=label)
        for _, color, label in GROUPS
    ]
    expected_lines = {}

    def line(ax, x, values, *, gid, **kwargs):
        x, values = np.asarray(x, float), np.asarray(values, float)
        (plotted,) = ax.plot(x, values, **kwargs)
        plotted.set_gid(gid)
        expected_lines[gid] = (x.copy(), values.copy())

    def draw(ax, dataset, columns, scale=1, accuracy=False, references=()):
        data = frame[frame.dataset.eq(dataset)]
        observed = []
        for group, color, _ in GROUPS:
            if group not in columns:
                continue
            metric = columns[group]
            for seed, marker, size in zip(SEEDS, markers, sizes):
                current = data[data.seed.eq(seed)].sort_values("level")
                if list(current.level) != levels:
                    raise ValueError("缺少种子或H配置")
                values = current[metric].to_numpy() * scale
                observed.extend(values)
                line(
                    ax,
                    current.level,
                    values,
                    gid=f"{dataset}:{group}:{seed}:{metric}",
                    color=color,
                    marker=marker,
                    markerfacecolor="none",
                    markersize=size,
                    linewidth=0.8,
                    alpha=0.48,
                )
            mean = data.groupby("level")[metric].mean().sort_index()
            line(
                ax,
                mean.index,
                mean * scale,
                gid=f"{dataset}:{group}:mean:{metric}",
                color=color,
                marker="D",
                markerfacecolor="none",
                markersize=10,
                linewidth=2.2,
            )
        decorate(ax, dataset, observed, accuracy, references)

    def decorate(ax, dataset, observed, accuracy=False, references=()):
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels, list(map(str, levels)))
        ax.set_xticks([], minor=True)
        ax.set_xlabel("Local SGD 同步间隔 H")
        ax.set_title(f"({'abc'[DATASETS.index(dataset)]}) {NAMES[dataset]}")
        ax.grid(alpha=0.17)
        low, high = min(observed), max(observed)
        if accuracy:
            margin = max(0.3, (high - low) * 0.18)
            ax.set_ylim(max(0, low - margin), min(100, high + margin))
        else:
            ax.set_ylim(0, max(high * 1.15, max(references, default=0) * 1.1, 1e-9))
        for value in references:
            ax.axhline(value, color="#888888", linestyle="--", linewidth=0.8)

    audit = []
    folder = results_dir / "figs"
    folder.mkdir(parents=True, exist_ok=True)

    def save(fig, filename, title, handles, bottom=0.10):
        fig.suptitle(f"E5 {title}", fontsize=15)
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, bottom, 1, 0.95))
        points, count = 0, 0
        seen = set()
        for ax in fig.axes:
            for plotted in ax.lines:
                gid = plotted.get_gid()
                if gid is None:
                    continue
                x, y = expected_lines[gid]
                if not np.array_equal(x, plotted.get_xdata()) or not np.array_equal(
                    y, plotted.get_ydata()
                ):
                    raise ValueError("图中数值与输入指标不符")
                if (
                    not np.isfinite(y).all()
                    or (y < ax.get_ylim()[0]).any()
                    or (y > ax.get_ylim()[1]).any()
                ):
                    raise ValueError("图中存在无效或截断点")
                if gid in seen:
                    raise ValueError("重复的指标曲线")
                seen.add(gid)
                points += len(y)
                count += 1
        target = folder / filename
        fig.savefig(target, dpi=180, bbox_inches="tight")
        plt.close(fig)
        audit.append(
            {
                "figure": filename,
                "source": ["scaling.csv", "ordinary_sgd_reference.csv"],
                "points": points,
                "curves": count,
                "source_values_checked": True,
                "all_points_inside_axes": True,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        print(target, flush=True)
        expected_lines.clear()

    definitions = [
        (
            "F8_single_dual_throughput.png",
            "图8 两种单GPU参考与双GPU的吞吐及更新成本",
            group_handles,
            [
                (
                    {
                        "serial": "single_throughput",
                        "dual": "dual_throughput",
                        "ordinary": "ordinary_throughput",
                    },
                    "全局训练吞吐（样本/秒）",
                    1,
                    False,
                    (),
                ),
                (
                    {
                        "serial": "single_update_ms",
                        "dual": "dual_update_ms",
                        "ordinary": "ordinary_update_ms",
                    },
                    "每步256样本的摊销耗时（ms）",
                    1,
                    False,
                    (),
                ),
            ],
        ),
        (
            "F9_parallel_speedup.png",
            "图9 严格同算法加速比与普通SGD任务吞吐比",
            [
                Line2D(
                    [],
                    [],
                    color="#2766A6",
                    linewidth=2.2,
                    label="严格配对：双卡/单卡串行双副本",
                ),
                Line2D(
                    [],
                    [],
                    color="#21876A",
                    linewidth=2.2,
                    label="任务参考：双卡/单卡普通SGD（仅上排）",
                ),
            ],
            [
                (
                    {"dual": "speedup", "ordinary": "task_throughput_ratio"},
                    "吞吐比（H>1时绿色改变算法）",
                    1,
                    False,
                    (1, 2),
                ),
                (
                    {"dual": "compute_speedup"},
                    "严格配对计算段加速比 C1/C2",
                    1,
                    False,
                    (1, 2),
                ),
                (
                    {"dual": "parallel_efficiency"},
                    "严格配对双GPU并行效率（%）",
                    100,
                    False,
                    (100,),
                ),
            ],
        ),
        (
            "F10_single_dual_quality_memory.png",
            "图10 末轮质量与每卡峰值显存（普通SGD的9次结果按H复用）",
            group_handles,
            [
                (
                    {
                        "serial": "single_test_acc",
                        "dual": "dual_test_acc",
                        "ordinary": "ordinary_test_acc",
                    },
                    "末轮测试准确率（%）",
                    100,
                    True,
                    (),
                ),
                (
                    {
                        "serial": "single_peak_memory_mb",
                        "dual": "dual_peak_memory_mb",
                        "ordinary": "ordinary_peak_memory_mb",
                    },
                    "每卡峰值张量显存（MB）",
                    1,
                    False,
                    (),
                ),
            ],
        ),
    ]
    for filename, title, handles, metrics in definitions:
        fig, axes = plt.subplots(
            len(metrics), 3, figsize=(15, 3.6 * len(metrics) + 1), squeeze=False
        )
        for row, (columns, ylabel, scale, accuracy, references) in enumerate(metrics):
            for column, dataset in enumerate(DATASETS):
                draw(axes[row, column], dataset, columns, scale, accuracy, references)
                axes[row, column].set_ylabel(ylabel)
        save(
            fig,
            filename,
            title,
            [*handles, *seed_handles],
            0.10 if len(metrics) == 2 else 0.07,
        )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    for column, dataset in enumerate(DATASETS):
        draw(
            axes[0, column],
            dataset,
            {
                "serial": "single_stage_wall_s",
                "dual": "dual_stage_wall_s",
                "ordinary": "ordinary_stage_wall_s",
            },
        )
        axes[0, column].set_ylabel("20轮训练阶段墙钟（s）")
        data = frame[frame.dataset.eq(dataset)]
        observed = []
        for group, color, _ in GROUPS:
            prefix = {"serial": "single", "dual": "dual", "ordinary": "ordinary"}[group]
            metric = prefix + "_time_to_target_s"
            for seed, marker, size in zip(SEEDS, markers, sizes):
                current = data[data.seed.eq(seed)].sort_values("level")
                reached = current[prefix + "_target_reached"].astype(bool)
                for flag in (True, False):
                    subset = current[reached.eq(flag)]
                    if subset.empty:
                        continue
                    values = subset[
                        metric if flag else prefix + "_stage_wall_s"
                    ].to_numpy()
                    observed.extend(values)
                    line(
                        axes[1, column],
                        subset.level,
                        values,
                        gid=f"{dataset}:{group}:{seed}:{'reached' if flag else 'censored'}",
                        color=color,
                        marker=marker if flag else "v",
                        markerfacecolor="none",
                        markersize=size,
                        linestyle="none",
                        alpha=0.65,
                    )
            complete = data.groupby("level")[prefix + "_target_reached"].all()
            mean = data.groupby("level")[metric].mean()[complete]
            if not mean.empty:
                line(
                    axes[1, column],
                    mean.index,
                    mean,
                    gid=f"{dataset}:{group}:mean:{metric}",
                    color=color,
                    marker="D",
                    markerfacecolor="none",
                    markersize=10,
                    linewidth=2.2,
                )
        decorate(axes[1, column], dataset, observed)
        target = reference_config(dataset).target_accuracy * 100
        axes[1, column].set_title(
            f"({'abc'[column]}) {NAMES[dataset]}（验证目标≥{target:g}%）"
        )
        axes[1, column].set_ylabel("Time-to-Target（s，验证集）")
    censor_handle = Line2D(
        [],
        [],
        color="#777777",
        marker="v",
        markerfacecolor="none",
        linestyle="none",
        label="倒三角：未达标，显示训练阶段下界",
    )
    save(
        fig,
        "F11_wall_time_to_target.png",
        "图11 20轮训练阶段墙钟与 Time-to-Target",
        [*group_handles, *seed_handles, censor_handle],
        0.12,
    )
    sources = {
        name: hashlib.sha256((results_dir / name).read_bytes()).hexdigest()
        for name in ("scaling.csv", "ordinary_sgd_reference.csv")
    }
    (results_dir / "scaling_figure_validation.json").write_text(
        json.dumps({"source_sha256": sources, "figures": audit}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    render(parser.parse_args().results_dir)
