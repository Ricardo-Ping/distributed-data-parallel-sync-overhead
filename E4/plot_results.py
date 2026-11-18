"""E4仅生成CSV和统计图；实验报告直接维护。"""

import argparse
import json
from pathlib import Path
import pandas as pd
from E4.config import DATASETS, RESULTS_ROOT
from E4.check_results import load_final_runs
from E4.metrics import seed_summary
from E4.plot_support import comparison_frame, draw, legend_handles, save_figure

NAMES = {"mnist": "MNIST", "fashionmnist": "FashionMNIST", "cifar10": "CIFAR-10"}


def render(results=RESULTS_ROOT, single_results=None):
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
        raise RuntimeError("需要中文字体，例如Noto Sans CJK")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    frame = load_final_runs(results)
    frame.to_csv(results / "final.csv", index=False)
    seed_summary(frame).to_csv(results / "seed_summary.csv", index=False)
    paired = []
    for (ds, k, seed), g in frame.groupby(["dataset", "level", "seed"]):
        a = g[g.variant.eq("every_micro")].iloc[0]
        b = g[g.variant.eq("window_end")].iloc[0]
        paired.append(
            dict(
                dataset=ds,
                level=k,
                seed=seed,
                throughput_speedup=b.train_throughput / a.train_throughput,
                sync_time_ratio=b.avg_sync_ms / a.avg_sync_ms,
                allreduce_count_ratio=b.total_gradient_allreduces
                / a.total_gradient_allreduces,
            )
        )
    comparisons = pd.DataFrame(paired)
    comparisons.to_csv(results / "paired_comparison.csv", index=False)
    folder = results / "figs"
    folder.mkdir(exist_ok=True)
    comparison = comparison_frame(frame, results, single_results)
    paired_plot = comparisons.assign(series="paired", plot_x=comparisons.level)
    paired_plot.to_csv(results / "sync_optimization_plot.csv", index=False)
    groups = (
        ("single", "#646C76", "单GPU（匹配全局Batch）"),
        ("every_micro", "#2766A6", "双GPU（每个micro同步）"),
        ("window_end", "#C97424", "双GPU（窗口末同步）"),
    )
    handles = legend_handles(groups)

    def draw_panel(axis, dataset, metric, scale, is_pair=False):
        source = paired_plot if is_pair else comparison
        selected = (("paired", "#333333", "同K通信优化比"),) if is_pair else groups
        draw(axis, source, dataset, metric, selected, scale)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        axis.set_xticks([], minor=True)
        axis.set_xlabel(
            "累积次数 K（双GPU每卡micro=128）"
            if is_pair
            else "累积次数 K（单GPU micro=256；双GPU每卡128）"
        )
        if is_pair:
            axis.axhline(1, color="#888888", linestyle="--", linewidth=0.8)
            axis.set_ylim(0, max(axis.get_ylim()[1], 1.1))

    figures = [
        (
            "F1_calls_payload.png",
            "同步次数与累计逻辑载荷",
            [
                ("total_gradient_allreduces", "20轮梯度归约次数", 1),
                ("total_logical_payload_bytes", "每 rank 逻辑载荷（GB）", 1e-9),
            ],
            False,
        ),
        (
            "F2_sync_cost.png",
            "同步成本与摊销",
            [
                ("avg_sync_ms", "同步耗时（ms/参数更新）", 1),
                ("sync_share", "同步占窗口时间（%）", 100),
                ("sync_us_per_sample", "同步成本（微秒/全局样本）", 1),
            ],
            False,
        ),
        (
            "F3_throughput_update.png",
            "训练吞吐与完整累积窗口成本",
            [
                ("train_throughput", "训练吞吐（样本/秒）", 1),
                ("avg_update_ms", "完整参数更新时间（ms）", 1),
            ],
            False,
        ),
        (
            "F4_quality_memory.png",
            "末轮模型质量与训练显存",
            [
                ("test_acc", "末轮测试准确率（%）", 100),
                ("peak_allocated_mb", "每卡训练峰值显存（MB）", 1),
            ],
            False,
        ),
        (
            "F5_paired_optimization.png",
            "同K配对的通信优化收益",
            [
                ("throughput_speedup", "吞吐比：末次同步/每 micro 同步", 1),
                ("sync_time_ratio", "同步耗时比：末次/每 micro", 1),
            ],
            True,
        ),
    ]
    audit = []
    for filename, title, metrics, is_pair in figures:
        fig, axes = plt.subplots(
            len(metrics), 3, figsize=(15, 3.6 * len(metrics) + 1), squeeze=False
        )
        for row, (metric, label, scale) in enumerate(metrics):
            for col, ds in enumerate(DATASETS):
                draw_panel(axes[row, col], ds, metric, scale, is_pair)
                axes[row, col].set_ylabel(label)
        fig.suptitle("E4 " + title + ("" if is_pair else "：单/双GPU对照"), fontsize=15)
        legend = (
            legend_handles((("paired", "#333333", "同K通信优化比"),))[1:]
            if is_pair
            else handles
        )
        fig.legend(
            handles=legend, loc="lower center", ncol=4 if is_pair else 3, frameon=False
        )
        fig.tight_layout(rect=(0, 0.10 if len(metrics) == 2 else 0.07, 1, 0.95))
        source = results / (
            "sync_optimization_plot.csv" if is_pair else "main_comparison.csv"
        )
        audit.append(save_figure(fig, folder / filename, source))
        plt.close(fig)
        print(folder / filename)
    (results / "figure_validation.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--single-results-dir", type=Path)
    args = p.parse_args()
    render(args.results_dir, args.single_results_dir)
