"""E5主图：匹配算法的单/双GPU对照及独立Profiler诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from E5.config import DATASETS, RESULTS_ROOT
from E5.check_results import load_final_runs
from E5.metrics import seed_summary
from E5.plot_support import (
    NAMES,
    comparison_frame,
    digest,
    draw,
    legend_handles,
    save_figure,
)


def render(results=RESULTS_ROOT, single_results=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    results = Path(results)
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
    baseline = frame[frame.level.eq(1)]
    matched = frame.merge(
        baseline, on=["dataset", "seed"], suffixes=("", "_h1"), validate="many_to_one"
    )
    paired = matched[["dataset", "level", "seed"]].copy()
    paired["throughput_speedup"] = (
        matched.train_throughput / matched.train_throughput_h1
    )
    paired["sync_per_update_ratio"] = (
        matched.avg_sync_per_update_ms / matched.avg_sync_per_update_ms_h1
    )
    paired["training_stage_time_ratio"] = (
        matched.training_stage_wall_s / matched.training_stage_wall_s_h1
    )
    paired["test_accuracy_difference_pp"] = (
        matched.test_acc - matched.test_acc_h1
    ) * 100
    paired["both_targets_reached"] = matched.target_reached & matched.target_reached_h1
    paired["time_to_target_speedup"] = (
        matched.time_to_target_s_h1 / matched.time_to_target_s
    ).where(paired.both_targets_reached)
    paired.to_csv(results / "paired_comparison.csv", index=False)

    comparison = comparison_frame(frame, results, single_results)
    groups = (
        ("single", "#646C76", "单GPU（串行双副本）"),
        ("local_sgd", "#2766A6", "双GPU（Local SGD）"),
    )
    handles = legend_handles(groups)
    folder = results / "figs"
    folder.mkdir(exist_ok=True)
    audit = []

    def format_axis(axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        axis.set_xticks([], minor=True)
        axis.set_xlabel("Local SGD 同步间隔 H（每副本Batch=128）")

    figures = [
        (
            "F1_calls_payload.png",
            "跨卡参数归约次数与逻辑载荷",
            [
                ("total_model_allreduces", "20轮跨卡参数归约次数", 1),
                ("total_logical_payload_bytes", "每 rank 跨卡逻辑载荷（GB）", 1e-9),
            ],
        ),
        (
            "F2_throughput_update.png",
            "训练吞吐与更新摊销成本",
            [
                ("train_throughput", "全局训练吞吐（样本/秒）", 1),
                ("avg_update_ms", "每个逻辑更新步的摊销时间（ms）", 1),
            ],
        ),
        (
            "F3_sync_wait.png",
            "跨卡同步路径与两卡到达差",
            [
                ("avg_sync_ms", "跨卡参数平均耗时（ms/同步轮次）", 1),
                ("sync_share", "跨卡同步占轮次时间（%）", 100),
                ("arrival_gap_ms", "两 rank 同步点到达差（ms）", 1),
            ],
        ),
        (
            "F4_drift_quality.png",
            "同步前漂移、末轮质量与训练显存",
            [
                ("drift_mean_relative_l2", "同步前参数相对L2距离", 1),
                ("test_acc", "末轮测试准确率（%）", 100),
                ("peak_allocated_mb", "每卡训练峰值张量显存（MB）", 1),
            ],
        ),
    ]
    for filename, title, metrics in figures:
        fig, axes = plt.subplots(
            len(metrics), 3, figsize=(15, 3.6 * len(metrics) + 1), squeeze=False
        )
        for row, (metric, label, scale) in enumerate(metrics):
            for col, dataset in enumerate(DATASETS):
                axis = axes[row, col]
                draw(axis, comparison, dataset, metric, groups, scale)
                format_axis(axis)
                axis.set_ylabel(label)
        fig.suptitle("E5 " + title + "：单/双GPU对照", fontsize=15)
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.10 if len(metrics) == 2 else 0.07, 1, 0.95))
        audit.append(
            save_figure(fig, folder / filename, results / "main_comparison.csv")
        )
        plt.close(fig)
        print(folder / filename, flush=True)

    # 相同算法的训练/验证曲线按epoch对齐，单GPU与双GPU分行展示。
    histories, history_sources = [], {}
    for record in comparison.to_dict("records"):
        source = Path(record["source_file"]).parent / "epochs.csv"
        history = pd.read_csv(source)
        if history.epoch.tolist() != list(range(1, 21)):
            raise ValueError("收敛曲线必须包含20轮")
        history = history.assign(
            dataset=record["dataset"],
            seed=record["seed"],
            series=f"{record['series']}_h{record['reference_level']}",
            role=record["series"],
            level=record["reference_level"],
            plot_x=history.epoch,
            source_file=str(source),
        )
        histories.append(history)
        history_sources[str(source)] = digest(source)
    convergence = pd.concat(histories, ignore_index=True)
    convergence.to_csv(results / "convergence_plot.csv", index=False)
    provenance = results / "main_comparison_sources.json"
    saved = json.loads(provenance.read_text())
    saved["epoch_files_sha256"] = history_sources
    provenance.write_text(json.dumps(saved, indent=2) + "\n")
    hcolors = ("#2364A4", "#C97424", "#268467", "#8F5BA6", "#B64956")
    levels = (1, 2, 4, 8, 16)
    metrics = [
        ("single", "train_loss", "单GPU串行双副本\n训练交叉熵损失", 1),
        ("local_sgd", "train_loss", "双GPU Local SGD\n训练交叉熵损失", 1),
        ("single", "val_acc", "单GPU串行双副本\n验证准确率（%）", 100),
        ("local_sgd", "val_acc", "双GPU Local SGD\n验证准确率（%）", 100),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15, 15.3), squeeze=False)
    for row, (role, metric, label, scale) in enumerate(metrics):
        curves = tuple(
            (f"{role}_h{h}", color, f"H={h}") for h, color in zip(levels, hcolors)
        )
        for axis, dataset in zip(axes[row], DATASETS):
            draw(axis, convergence, dataset, metric, curves, scale, dense=True)
            axis.set_xticks([1, 5, 10, 15, 20])
            axis.set_xlabel("训练 epoch")
            axis.set_ylabel(label)
    convergence_handles = [
        Line2D([], [], color=color, linewidth=2.2, label=f"H={h}")
        for h, color in zip(levels, hcolors)
    ]
    convergence_handles += legend_handles((("local_sgd", "#666666", ""),))[1:]
    fig.suptitle("E5 单/双GPU训练与验证：细线为三个种子，粗线为逐轮均值", fontsize=14)
    fig.legend(handles=convergence_handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    audit.append(
        save_figure(fig, folder / "F5_validation.png", results / "convergence_plot.csv")
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    targets = []
    for axis, dataset in zip(axes, DATASETS):
        draw(axis, comparison, dataset, "time_to_target_s", groups, censored=True)
        format_axis(axis)
        group = comparison[comparison.dataset.eq(dataset)]
        upper = axis.get_ylim()[1]
        for h in levels:
            for offset, (name, color, _) in enumerate(groups):
                values = group[group.series.eq(name) & group.reference_level.eq(h)]
                reached = int(values.target_reached.sum())
                axis.text(
                    h,
                    upper * (0.96 - 0.075 * offset),
                    ("单" if name == "single" else "双") + f"{reached}/3",
                    color=color,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
                if name == "local_sgd":
                    targets.append(
                        dict(
                            dataset=dataset,
                            level=h,
                            reached=reached,
                            total=3,
                            mean_time_s=values.time_to_target_s.mean()
                            if reached == 3
                            else np.nan,
                            std_time_s=values.time_to_target_s.std()
                            if reached == 3
                            else np.nan,
                        )
                    )
        axis.set_ylabel("达到验证目标的耗时（秒）")
        axis.set_title(
            f"({'abc'[DATASETS.index(dataset)]}) {NAMES[dataset]}，目标{group.target_accuracy.iloc[0]:.0%}"
        )
    pd.DataFrame(targets).to_csv(results / "time_to_target_summary.csv", index=False)
    target_handles = handles + [
        Line2D(
            [],
            [],
            color="#555555",
            marker="x",
            linestyle="none",
            label="未达到：20轮观测时长",
        )
    ]
    fig.suptitle("E5 单/双GPU Time-to-Target（含验证、诊断与日志）", fontsize=14)
    fig.legend(handles=target_handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 0.92))
    audit.append(
        save_figure(
            fig, folder / "F6_time_to_target.png", results / "main_comparison.csv"
        )
    )
    plt.close(fig)

    profile_path = results / "profile_seed_summary.csv"
    if profile_path.exists():
        profile = pd.read_csv(profile_path)
        profile = profile.assign(series="local_sgd", plot_x=profile.level)
        profile.to_csv(results / "profile_plot.csv", index=False)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), squeeze=False)
        for row, (metric, label, scale) in enumerate(
            [
                ("gpu_gap_fraction", "无CUDA活动比例（%）", 100),
                ("nccl_kernel_ms", "64步内NCCL kernel活动时长（ms）", 1),
            ]
        ):
            for col, dataset in enumerate(DATASETS):
                axis = axes[row, col]
                draw(axis, profile, dataset, metric, groups[1:], scale)
                format_axis(axis)
                axis.set_ylabel(label)
        fig.suptitle("E5 双GPU独立Profiler诊断：每个种子先对两卡取平均", fontsize=14)
        fig.legend(
            handles=legend_handles(groups[1:]),
            loc="lower center",
            ncol=3,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.94))
        audit.append(
            save_figure(
                fig, folder / "F7_gpu_activity.png", results / "profile_plot.csv"
            )
        )
        plt.close(fig)
    (results / "figure_validation.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--single-results-dir", type=Path)
    args = parser.parse_args()
    render(args.results_dir, args.single_results_dir)
