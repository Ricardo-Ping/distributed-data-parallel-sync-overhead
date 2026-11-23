"""E3仅生成图表和CSV，报告直接维护。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from E3.check_results import load_final_runs
from E3.config import DATASETS, EXPERIMENT, RESULTS_ROOT
from E3.metrics import seed_summary
from E3.plot_support import comparison_frame, draw, legend_handles, save_figure

NAMES = {"mnist": "MNIST", "fashionmnist": "FashionMNIST", "cifar10": "CIFAR-10"}


def render(results: Path = RESULTS_ROOT, single_results: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        try:
            font_manager.findfont(font, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams.update({"font.family": [font], "axes.unicode_minus": False})
        break
    else:
        raise RuntimeError("需要中文字体，例如fonts-noto-cjk")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    frame = load_final_runs(results)
    summary = seed_summary(frame)
    frame.to_csv(results / "final.csv", index=False)
    summary.to_csv(results / "seed_summary.csv", index=False)
    paired = []
    for dataset, group in frame.groupby("dataset"):
        base = group[group.micro_batch.eq(128)].set_index("seed")
        for row in group.to_dict("records"):
            paired.append(
                dict(
                    dataset=dataset,
                    micro_batch=row["micro_batch"],
                    accumulation=row["accumulation"],
                    seed=row["seed"],
                    throughput_relative=row["train_throughput"]
                    / base.loc[row["seed"], "train_throughput"],
                    update_time_relative=row["avg_update_ms"]
                    / base.loc[row["seed"], "avg_update_ms"],
                )
            )
    pd.DataFrame(paired).to_csv(results / "paired_relative.csv", index=False)
    folder = results / "figs"
    folder.mkdir(exist_ok=True)
    comparison = comparison_frame(frame, results, single_results)
    groups = (
        ("single", "#646C76", "单GPU（匹配全局Batch）"),
        ("dual", "#2766A6", "双GPU（DDP）"),
    )
    handles = legend_handles(groups)

    def plot(axis, dataset, metric, scale=1):
        draw(axis, comparison, dataset, metric, groups, scale)
        axis.set_xscale("log", base=2)
        ticks = sorted(comparison.plot_x.unique())
        axis.set_xticks(ticks, [f"{value}×{128 // value}" for value in ticks])
        axis.set_xlabel("参考双GPU：每卡 Micro-Batch × K（全局256）")
        axis.set_xticks([], minor=True)

    figures = [
        (
            "F1_throughput.png",
            "固定全局批量的不同实现与训练吞吐",
            (("train_throughput", "训练吞吐（样本/秒）", 1),),
        ),
        (
            "F2_update_compute.png",
            "更新时间与计算段效率",
            (
                ("avg_update_ms", "平均更新时间（毫秒/更新）", 1),
                ("local_compute_throughput", "每卡计算段吞吐（样本/计算秒）", 1),
            ),
        ),
        (
            "F3_sync_cost.png",
            "梯度同步的成本与摊销",
            (
                ("avg_sync_ms", "平均同步耗时（毫秒/更新）", 1),
                ("sync_share", "同步占更新时间（%）", 100),
                ("sync_ms_per_sample", "同步成本（微秒/全局样本）", 1000),
            ),
        ),
        (
            "F4_memory_accuracy.png",
            "训练显存与末轮测试结果",
            (
                ("peak_allocated_mb", "每卡训练峰值显存（MB）", 1),
                ("test_acc", "末轮测试准确率（%）", 100),
            ),
        ),
    ]
    audit = []
    for filename, title, metrics in figures:
        figure, axes = plt.subplots(
            len(metrics), 3, figsize=(15, 3.5 * len(metrics) + 1), squeeze=False
        )
        for row, (metric, label, scale) in enumerate(metrics):
            for column, dataset in enumerate(DATASETS):
                plot(axes[row, column], dataset, metric, scale)
                axes[row, column].set_ylabel(label)
        figure.suptitle(f"{EXPERIMENT} {title}：单/双GPU对照", fontsize=15)
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        figure.tight_layout(rect=(0, 0.10 if len(metrics) == 1 else 0.06, 1, 0.94))
        figure.canvas.draw()
        audit.append(
            save_figure(figure, folder / filename, results / "main_comparison.csv")
        )
        plt.close(figure)
        print(folder / filename)
    (results / "figure_validation.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--single-results-dir", type=Path)
    args = parser.parse_args()
    render(args.results_dir, args.single_results_dir)
