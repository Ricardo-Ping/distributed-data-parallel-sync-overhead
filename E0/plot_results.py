"""终值以单卡/双卡并列柱状图展示三个种子及均值；逐轮曲线使用逐点95% t区间。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from E0.config import EXPERIMENTS, RESULTS_ROOT
from E0.metrics import SEEDS, SUMMARY_METRICS, summarize_seeds, three_seed_statistics

DISPLAY_NAMES = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "cifar10": "CIFAR-10",
}
GPU_LABELS = {1: "单卡", 2: "双卡"}


def configure_chinese_font() -> None:
    """选用中文字体；缺少字体时明确提示，避免输出方框或缺字的图。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for family in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "Droid Sans Fallback",
    ):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams.update({"font.family": [family], "axes.unicode_minus": False})
        return
    raise RuntimeError(
        "绘图需要中文字体，请安装 Noto Sans CJK（Linux 软件包 fonts-noto-cjk）"
    )


def load_results(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    finals, histories = [], []
    for dataset in EXPERIMENTS:
        final = pd.read_csv(results / dataset / "final.csv")
        required = {
            "run_id",
            "seed",
            "ws",
            "data",
            "device",
            "backend",
            "epochs_ran",
            "test_epoch",
            *SUMMARY_METRICS,
        }
        if required - set(final):
            raise ValueError(f"{dataset}: final.csv 缺少绘图所需字段")
        if len(final) != 6 or final.run_id.duplicated().any():
            raise ValueError(f"{dataset}: 需要六次独立运行")
        if set(zip(final.ws, final.seed)) != {
            (ws, seed) for ws in (1, 2) for seed in SEEDS
        }:
            raise ValueError(f"{dataset}: 必须包含单/双GPU下的三个种子")
        if not (
            final.data.eq(dataset)
            & final.device.eq("cuda:0")
            & final.epochs_ran.eq(20)
            & final.test_epoch.eq(20)
        ).all():
            raise ValueError(f"{dataset}: 需要固定20轮的GPU结果")
        if not (final.backend.eq(np.where(final.ws.eq(2), "nccl", "none"))).all():
            raise ValueError(f"{dataset}: 需要NCCL双GPU结果")
        if not np.isfinite(final[list(SUMMARY_METRICS)].to_numpy()).all():
            raise ValueError(f"{dataset}: 最终指标包含非有限值")
        epochs = pd.read_csv(results / dataset / "epochs.csv")
        if (
            set(epochs.run_id) != set(final.run_id)
            or epochs.duplicated(["run_id", "epoch"]).any()
        ):
            raise ValueError(f"{dataset}: 逐轮记录缺失或重复")
        for identifier, group in epochs.groupby("run_id"):
            if sorted(group.epoch) != list(range(20)):
                raise ValueError(f"{identifier}: 需要完整20轮记录")
        if epochs.test_acc.notna().any():
            raise ValueError("逐轮图不能包含测试集准确率")
        if not np.isfinite(
            epochs[["train_loss", "val_loss", "val_acc"]].to_numpy()
        ).all():
            raise ValueError(f"{dataset}: 逐轮指标包含非有限值")
        if "seed" in epochs:
            epochs = epochs.drop(columns="seed")
        epochs = epochs.merge(
            final[["run_id", "seed"]], on="run_id", validate="many_to_one"
        )
        epochs["epoch"] += 1
        finals.append(final.assign(dataset=dataset))
        histories.append(epochs.assign(dataset=dataset))
    return pd.concat(finals, ignore_index=True), pd.concat(histories, ignore_index=True)


def epoch_statistics(epochs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, world_size, epoch), group in epochs.groupby(
        ["dataset", "ws", "epoch"]
    ):
        if sorted(group.seed) != list(SEEDS):
            raise ValueError(f"{dataset}/GPU{world_size}/epoch{epoch}: 缺少三个种子")
        for metric in ("train_loss", "val_loss", "val_acc"):
            values = group[metric].to_numpy()
            rows.append(
                {
                    "dataset": dataset,
                    "ws": world_size,
                    "epoch": epoch,
                    "metric": metric,
                    **three_seed_statistics(values.tolist()),
                }
            )
    return pd.DataFrame(rows)


def render(results: Path = RESULTS_ROOT) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    configure_chinese_font()
    finals, epochs = load_results(results)
    summary = pd.DataFrame(summarize_seeds(finals.to_dict("records")))
    epoch_summary = epoch_statistics(epochs)
    finals.sort_values(["dataset", "seed", "ws"]).to_csv(
        results / "seed_results.csv", index=False
    )
    summary.to_csv(results / "seed_summary.csv", index=False)
    epoch_summary.to_csv(results / "epoch_summary.csv", index=False)
    figures = results / "figs"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    gpu_colors = {1: "#2563A6", 2: "#C76A13"}
    x = np.arange(4)
    bar_width = 0.34

    def seed_bars(
        axis, dataset: str, metric: str, scale: float = 1, fmt: str = ".3f"
    ) -> None:
        maximum = 0.0
        for world_size in (1, 2):
            values = (
                finals[(finals.dataset == dataset) & finals.ws.eq(world_size)]
                .set_index("seed")
                .loc[list(SEEDS), metric]
                .to_numpy()
                * scale
            )
            heights = np.append(values, values.mean())
            offset = -bar_width / 2 if world_size == 1 else bar_width / 2
            bars = axis.bar(
                x + offset,
                heights,
                width=bar_width,
                color=gpu_colors[world_size],
                edgecolor="white",
                linewidth=0.5,
                label=GPU_LABELS[world_size],
                zorder=3,
            )
            axis.bar_label(
                bars,
                labels=[format(value, fmt) for value in heights],
                padding=3,
                fontsize=8,
                color=gpu_colors[world_size],
            )
            maximum = max(maximum, float(heights.max()))
        axis.axvline(2.5, color="#8C929C", ls=":", lw=1)
        axis.set_xticks(x, [*map(str, SEEDS), "均值"])
        axis.set_xlim(-0.6, 3.6)
        axis.set_xlabel("随机种子 / 三种子均值")
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(0, maximum * 1.18 if maximum > 0 else 1)
        if metric == "test_acc":
            axis.set_ylim(0, 108)
            axis.set_yticks([0, 20, 40, 60, 80, 100])

    seed_legend = [
        Patch(facecolor=gpu_colors[ws], label=GPU_LABELS[ws]) for ws in (1, 2)
    ]

    def save(figure, name: str) -> None:
        # 先确定中文和对数刻度的文字位置，再计算导出边界。
        figure.canvas.draw()
        figure.savefig(figures / name, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"[figure] {figures / name}")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, dataset in zip(axes, EXPERIMENTS):
        seed_bars(axis, dataset, "train_throughput", fmt=",.0f")
        axis.set_title(DISPLAY_NAMES[dataset])
        axis.set_ylabel("训练吞吐（样本/秒）")
    figure.suptitle("训练吞吐：三个随机种子与均值", fontsize=15)
    figure.legend(
        handles=seed_legend, loc="lower center", ncol=2, frameon=False, fontsize=10
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.92))
    save(figure, "F1_throughput_seeds.png")

    figure, axes = plt.subplots(3, 3, figsize=(15, 11))
    metrics = (
        ("avg_update_ms", "平均更新耗时（毫秒/次）", 1, ".3f"),
        ("avg_sync_ms", "平均同步耗时（毫秒/次）", 1, ".3f"),
        ("sync_share", "同步占更新耗时（%）", 100, ".2f"),
    )
    for row, (metric, label, scale, fmt) in enumerate(metrics):
        for column, dataset in enumerate(EXPERIMENTS):
            axis = axes[row, column]
            seed_bars(axis, dataset, metric, scale, fmt)
            axis.set_title(DISPLAY_NAMES[dataset])
            axis.set_ylabel(label)
    figure.suptitle("更新耗时、同步耗时与同步占比：单卡与双卡对照", fontsize=15)
    figure.legend(handles=seed_legend, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    save(figure, "F2_timing_seeds.png")

    seed_colors = {1234: "#2563A6", 1235: "#C76A13", 1236: "#268467"}
    seed_styles = {1234: "-", 1235: "--", 1236: ":"}
    seed_markers = {1234: "o", 1235: "s", 1236: "^"}
    curve_legend = [
        Line2D(
            [],
            [],
            color=seed_colors[seed],
            ls=seed_styles[seed],
            marker=seed_markers[seed],
            label=f"随机种子 {seed}",
        )
        for seed in SEEDS
    ]
    curve_legend += [
        Line2D([], [], color="#20242A", lw=2.4, label="三个种子的均值"),
        Patch(facecolor="#8C929C", alpha=0.25, label="逐轮95% t置信区间（三个种子）"),
    ]
    for panel, dataset in zip(("a", "b", "c"), EXPERIMENTS):
        figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True, sharey="row")
        for row, (metric, label, scale) in enumerate(
            (
                ("train_loss", "训练损失（交叉熵）", 1),
                ("val_loss", "验证损失（交叉熵）", 1),
                ("val_acc", "验证准确率（%）", 100),
            )
        ):
            for column, world_size in enumerate((1, 2)):
                axis = axes[row, column]
                group = epochs[(epochs.dataset == dataset) & epochs.ws.eq(world_size)]
                pivot = (
                    group.pivot(index="epoch", columns="seed", values=metric).loc[
                        range(1, 21), list(SEEDS)
                    ]
                    * scale
                )
                statistics = (
                    epoch_summary[
                        (epoch_summary.dataset == dataset)
                        & epoch_summary.ws.eq(world_size)
                        & epoch_summary.metric.eq(metric)
                    ]
                    .set_index("epoch")
                    .loc[range(1, 21)]
                )
                lower, upper = statistics.ci95_low * scale, statistics.ci95_high * scale
                axis.fill_between(
                    statistics.index,
                    lower,
                    upper,
                    color="#8C929C",
                    alpha=0.22,
                    zorder=0,
                )
                axis.plot(
                    statistics.index,
                    statistics["mean"] * scale,
                    color="#20242A",
                    lw=2.1,
                    zorder=2,
                )
                for index, seed in enumerate(SEEDS):
                    axis.plot(
                        pivot.index,
                        pivot[seed],
                        color=seed_colors[seed],
                        ls=seed_styles[seed],
                        lw=1.1,
                        marker=seed_markers[seed],
                        markersize=3.5,
                        markevery=(index, 3),
                        alpha=0.9,
                        zorder=3,
                    )
                if metric == "train_loss" and (lower > 0).all():
                    axis.set_yscale("log")
                axis.set_title(GPU_LABELS[world_size])
                axis.set_ylabel(
                    label + ("\n对数刻度" if axis.get_yscale() == "log" else "")
                )
                axis.set_xlabel("训练轮次")
                axis.set_xticks([1, 5, 10, 15, 20])
                axis.grid(alpha=0.2)
        figure.suptitle(
            f"{DISPLAY_NAMES[dataset]}：20轮训练与验证", fontsize=15
        )
        figure.legend(handles=curve_legend, loc="lower center", ncol=3, frameon=False)
        figure.tight_layout(rect=(0, 0.07, 1, 0.96))
        save(figure, f"F3{panel}_learning_{dataset}.png")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, dataset in zip(axes, EXPERIMENTS):
        seed_bars(axis, dataset, "test_acc", scale=100, fmt=".2f")
        axis.set_title(DISPLAY_NAMES[dataset])
        axis.set_ylabel("第20轮模型的测试准确率（%）")
    figure.suptitle("最终测试准确率：三个随机种子与均值", fontsize=15)
    figure.legend(
        handles=seed_legend, loc="lower center", ncol=2, frameon=False, fontsize=10
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.92))
    save(figure, "F4_test_accuracy_seeds.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    render(args.results_dir)
