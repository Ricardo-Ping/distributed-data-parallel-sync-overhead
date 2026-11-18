"""只生成 E1 图表与机器统计表；实验报告由人工维护。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from E1.check_results import load_final_runs
from E1.config import DATASETS, PRESETS, RESULTS_ROOT, SEEDS
from E1.metrics import paired_scaling, seed_summary

NAMES = {"mnist": "MNIST", "fashionmnist": "FashionMNIST", "cifar10": "CIFAR-10"}


def render(results: Path = RESULTS_ROOT) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams.update({"font.family": [family], "axes.unicode_minus": False})
        break
    else:
        raise RuntimeError("需要中文字体，例如 fonts-noto-cjk")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    finals = load_final_runs(results)
    summary = seed_summary(finals)
    scaling = paired_scaling(finals)
    finals.to_csv(results / "final.csv", index=False)
    summary.to_csv(results / "seed_summary.csv", index=False)
    scaling.to_csv(results / "paired_scaling.csv", index=False)
    finals[["dataset", "model", "parameters", "gradient_bytes", "forward_flops_per_sample", "hidden1", "hidden2"]].drop_duplicates().to_csv(results / "model_sizes.csv", index=False)
    folder = results / "figs"
    folder.mkdir(exist_ok=True)
    colors, markers = {1: "#2563A6", 2: "#C76A13"}, ("o", "s", "^")
    x, width = np.arange(3), .34
    legends = [Patch(facecolor=colors[ws], label="单卡" if ws == 1 else "双卡") for ws in (1, 2)]
    seed_legends = [Line2D([], [], color="#20242A", marker=marker, linestyle="none", label=f"种子 {seed}")
                    for seed, marker in zip(SEEDS, markers)]

    def ordered(frame: pd.DataFrame, dataset: str, ws: int, metric: str) -> np.ndarray:
        return frame[(frame.dataset == dataset) & frame.ws.eq(ws)].groupby("model")[metric].mean().reindex(PRESETS).to_numpy()

    def dots(axis, dataset: str, ws: int, metric: str, positions, scale=1):
        for seed, marker, offset in zip(SEEDS, markers, (-.06, 0, .06)):
            values = finals[(finals.dataset == dataset) & finals.ws.eq(ws) & finals.seed.eq(seed)].set_index("model").loc[list(PRESETS), metric]
            axis.scatter(positions + offset, values.to_numpy() * scale, color="#20242A", marker=marker,
                         s=16, zorder=5, linewidths=.4)

    def grouped(axis, dataset: str, metric: str, scale=1, fmt=".2f"):
        for ws in (1, 2):
            positions = x + (-width/2 if ws == 1 else width/2)
            values = ordered(finals, dataset, ws, metric) * scale
            bars = axis.bar(positions, values, width=width, color=colors[ws], alpha=.8, zorder=2)
            axis.bar_label(bars, labels=[format(v, fmt) for v in values], padding=5, fontsize=8)
            dots(axis, dataset, ws, metric, positions, scale)
        axis.set_xticks(x, list(PRESETS))
        axis.set_ylim(bottom=0)
        axis.margins(y=.2)
        axis.set_xlabel("模型规模（仅改变隐藏层宽度）")
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.18)

    def save(figure, name):
        figure.canvas.draw()
        figure.savefig(folder / name, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(folder / name)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    phase_colors = ("#A9BACD", "#2563A6", "#C76A13", "#3E8A70")
    phases = (("avg_data_ms", "数据准备"), ("avg_compute_ms", "前向与反向"), ("avg_sync_ms", "梯度同步"), ("avg_opt_ms", "SGD更新"))
    for axis, dataset in zip(axes, DATASETS):
        for ws in (1, 2):
            positions, bottom = x + (-width/2 if ws == 1 else width/2), np.zeros(3)
            for (metric, _), color in zip(phases, phase_colors):
                values = ordered(finals, dataset, ws, metric)
                axis.bar(positions, values, bottom=bottom, width=width, color=color,
                         hatch="//" if ws == 2 else None, edgecolor="white", linewidth=.3)
                bottom += values
            dots(axis, dataset, ws, "avg_update_ms", positions)
        axis.set_xticks(x, list(PRESETS))
        axis.set_xlabel("模型规模；每组左柱单卡、右柱双卡")
        axis.set_ylabel("平均更新耗时（毫秒/次）")
        axis.set_title(NAMES[dataset])
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    handles = [Patch(facecolor=color, label=label) for (_, label), color in zip(phases, phase_colors)]
    handles += [Patch(facecolor="white", edgecolor="#555555", hatch="//", label="双卡"), *seed_legends]
    figure.suptitle("图1 模型规模与更新时间分段（柱为三种子均值，点为逐种子总耗时）", fontsize=14)
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, .14, 1, .94))
    save(figure, "F1_step_breakdown.png")

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, dataset in enumerate(DATASETS):
        for row, (metric, scale, label) in enumerate((("sync_share", 100, "同步占更新耗时（%）"),
                                                     ("sync_compute_ratio", 1, "同步 / 前向反向时间（倍）"))):
            axis = axes[row, column]
            grouped(axis, dataset, metric, scale)
            axis.set_title(NAMES[dataset])
            axis.set_ylabel(label)
    figure.suptitle("图2 模型规模对计算与同步比例的影响", fontsize=15)
    figure.legend(handles=[*legends, *seed_legends], loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, .07, 1, .94))
    save(figure, "F2_compute_sync_ratio.png")

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, dataset in enumerate(DATASETS):
        axis = axes[0, column]
        grouped(axis, dataset, "train_throughput", fmt=",.0f")
        axis.set_title(NAMES[dataset])
        axis.set_ylabel("分段训练吞吐（样本/秒）")
        axis = axes[1, column]
        group = scaling[scaling.dataset == dataset]
        means = group.groupby("model").weak_scaling_efficiency.mean().reindex(PRESETS).to_numpy()*100
        bars = axis.bar(x, means, width=.55, color="#3E8A70", alpha=.8)
        axis.bar_label(bars, fmt="%.1f", padding=5, fontsize=8)
        for seed, marker, offset in zip(SEEDS, markers, (-.1, 0, .1)):
            values = group[group.seed.eq(seed)].set_index("model").loc[list(PRESETS), "weak_scaling_efficiency"]*100
            axis.scatter(x+offset, values, marker=marker, color="#20242A", s=17, zorder=3)
        axis.axhline(100, color="#8C929C", linestyle="--", linewidth=1)
        axis.set_xticks(x, list(PRESETS))
        axis.set_ylim(0, max(115, group.weak_scaling_efficiency.max()*115))
        axis.set_xlabel("模型规模")
        axis.set_ylabel("双卡弱扩展效率（%）")
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    figure.suptitle("图3 固定每卡 batch=128：吞吐与双卡弱扩展效率", fontsize=15)
    figure.legend(handles=[*legends, *seed_legends], loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, .07, 1, .94))
    save(figure, "F3_throughput_scaling.png")

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, dataset in zip(axes, DATASETS):
        group = finals[(finals.dataset == dataset) & finals.ws.eq(2)]
        payloads = group.groupby("model").gradient_bytes.first().reindex(PRESETS).to_numpy()/1e6
        for metric, color, label in (("avg_compute_ms", "#2563A6", "前向与反向"), ("avg_sync_ms", "#C76A13", "梯度同步")):
            means = group.groupby("model")[metric].mean().reindex(PRESETS)
            axis.plot(payloads, means, color=color, marker="D", label=label)
            for seed, marker in zip(SEEDS, markers):
                values = group[group.seed.eq(seed)].set_index("model").loc[list(PRESETS), metric]
                axis.scatter(payloads, values, marker=marker, color=color, s=17, alpha=.65)
        axis.set_title(NAMES[dataset])
        axis.set_xscale("log")
        axis.set_xticks(payloads, [f"{preset}\n{payload:.2f}" for preset, payload in zip(PRESETS, payloads)])
        axis.set_xticks([], minor=True)
        axis.set_xlabel("FP32完整梯度载荷（MB，对数刻度）")
        axis.set_ylabel("双卡平均耗时（毫秒/次）")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.18)
    figure.suptitle("图4 梯度载荷与双卡计算、同步耗时（线为均值，点为三个种子）", fontsize=14)
    figure.legend(handles=[*axes[0].get_legend_handles_labels()[0], *seed_legends],
                  loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, .07, 1, .94))
    save(figure, "F4_payload_timing.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    render(parser.parse_args().results_dir)
