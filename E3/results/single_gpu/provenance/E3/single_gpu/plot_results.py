"""单/双GPU对照图：三个种子与配对均值，不生成报告文字。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from E3.config import DATASETS, EXPERIMENT, RESULTS_ROOT, SEEDS

NAMES = {"mnist": "MNIST", "fashionmnist": "FashionMNIST", "cifar10": "CIFAR-10"}


def render():
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
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    frame = pd.read_csv(RESULTS_ROOT / "scaling.csv")
    variants = sorted(frame.variant.unique())
    colors = ("#2766A6", "#C97424")
    labels = {"ddp": "双GPU（DDP）", "every_micro": "双GPU（每micro同步）",
              "window_end": "双GPU（窗口末同步）", "local_sgd": "双GPU（Local SGD）"}
    groups = [("single", "#646C76", "单GPU（匹配全局Batch）")]
    groups += [(variant, color, labels[variant]) for variant, color in zip(variants, colors)]
    markers, sizes = ("o", "s", "^"), (8, 6, 4)
    handles = [Line2D([], [], color=color, linewidth=2.2, label=label) for _, color, label in groups]
    handles += [Line2D([], [], color="#666666", marker=marker, markerfacecolor="none", linestyle="none", label=f"种子 {seed}")
                for seed, marker in zip(SEEDS, markers)]
    handles += [Line2D([], [], color="#333333", marker="D", markerfacecolor="none", linewidth=2.2, label="粗线/菱形：三种子均值")]

    def group_data(dataset, group):
        data = frame[frame.dataset.eq(dataset)]
        return data.drop_duplicates(["level", "seed"]) if group == "single" else data[data.variant.eq(group)]

    def draw(ax, dataset, single_metric, dual_metric=None, scale=1, ratios=False, accuracy=False, references=()):
        observed = []
        for group, color, _ in groups[1:] if ratios else groups:
            data = group_data(dataset, group)
            metric = single_metric if group == "single" or dual_metric is None else dual_metric
            for seed, marker, size in zip(SEEDS, markers, sizes):
                current = data[data.seed.eq(seed)].sort_values("level")
                if current.level.duplicated().any():
                    raise ValueError("绘图种子存在重复配置")
                values = current[metric] * scale
                observed.extend(values)
                line, = ax.plot(current.level, values, color=color, marker=marker, markerfacecolor="none",
                                markersize=size, linewidth=0.8, alpha=0.48)
                line.set_gid(f"seed:{group}:{seed}:{metric}")
            mean = data.groupby("level")[metric].mean().sort_index()
            line, = ax.plot(mean.index, mean * scale, color=color, marker="D", markerfacecolor="none",
                            markersize=10, linewidth=2.2)
            line.set_gid(f"mean:{group}:{metric}")
        ax.set_xscale("log", base=2)
        levels = sorted(frame.level.unique())
        ax.set_xticks(levels, list(map(str, levels)))
        ax.set_xticks([], minor=True)
        ax.set_xlabel("参考双GPU每卡 Micro-Batch")
        ax.set_title(f'({"abc"[DATASETS.index(dataset)]}) {NAMES[dataset]}')
        ax.grid(alpha=0.17)
        low, high = min(observed), max(observed)
        if accuracy:
            margin = max(0.3, (high - low) * 0.18)
            ax.set_ylim(max(0, low - margin), min(100, high + margin))
        else:
            ax.set_ylim(0, max(high * 1.15, max(references, default=0) * 1.1, 1e-9))
        for value in references:
            ax.axhline(value, color="#888888", linestyle="--", linewidth=0.8)

    figures = [
        ("P1_single_dual_throughput.png", "逐配置匹配全局Batch的单/双GPU吞吐与更新时间", False,
         [("single_throughput", "dual_throughput", "全局训练吞吐（样本/秒）", 1, False, ()),
          ("single_update_ms", "dual_update_ms", "一次参数更新耗时（ms）", 1, False, ())]),
        ("P2_parallel_speedup.png", "同种子配对加速比与并行效率", True,
         [("speedup", None, "吞吐加速比 Q2/Q1", 1, False, (1, 2)),
          ("compute_speedup", None, "计算段加速比 C1/C2", 1, False, (1, 2)),
          ("parallel_efficiency", None, "双GPU并行效率（%）", 100, False, (100,))]),
        ("P3_single_dual_quality_memory.png", "单/双GPU末轮质量与每卡峰值显存", False,
         [("single_test_acc", "dual_test_acc", "末轮测试准确率（%）", 100, True, ()),
          ("single_peak_memory_mb", "dual_peak_memory_mb", "每卡峰值张量显存（MB）", 1, False, ())]),
    ]
    folder = RESULTS_ROOT / "figs"
    audit = []
    for filename, title, ratios, metrics in figures:
        fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 3.6 * len(metrics) + 1), squeeze=False)
        for row, (single, dual, label, scale, accuracy, references) in enumerate(metrics):
            for column, dataset in enumerate(DATASETS):
                ax = axes[row, column]
                draw(ax, dataset, single, dual, scale, ratios, accuracy, references)
                ax.set_ylabel(label)
        fig.suptitle(f"{EXPERIMENT} {title}", fontsize=15)
        legend = handles[1:] if ratios else handles
        fig.legend(handles=legend, loc="lower center", ncol=3 if len(legend) <= 6 else 4, frameon=False)
        fig.tight_layout(rect=(0, 0.10 if len(metrics) == 2 else 0.07, 1, 0.95))
        point_count = 0
        for ax in fig.axes:
            for line in ax.lines:
                if not line.get_gid():
                    continue
                values = np.asarray(line.get_ydata(), dtype=float)
                if not np.isfinite(values).all() or values.min() < ax.get_ylim()[0] or values.max() > ax.get_ylim()[1]:
                    raise ValueError("图中存在无效或被截断的种子/均值点")
                point_count += len(values)
        fig.savefig(folder / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        audit.append({"figure": filename, "source": "scaling.csv", "points": point_count,
                      "all_points_inside_axes": True, "all_three_seeds_and_means": True})
        print(folder / filename, flush=True)
    import json
    (RESULTS_ROOT / "scaling_figure_validation.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    render()
