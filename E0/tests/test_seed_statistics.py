"""三种子展示必须保留每次观测，并以等权算术均值汇总。"""

import math

import pandas as pd
import pytest

from E0.metrics import SUMMARY_METRICS, summarize_seeds, three_seed_statistics
from E0.plot_results import epoch_statistics, load_results


def test_summary_uses_mean_sample_std_and_actual_range():
    # 均值4、中位数2，用不同的统计量检验图中均值标注。
    rows = [
        {"dataset": "mnist", "ws": 1, "seed": seed, "value": value}
        for seed, value in zip((1234, 1235, 1236), (1, 2, 9))
    ]
    summary = summarize_seeds(rows, metrics=("value",))[0]
    assert summary["mean"] == 4
    assert summary["std"] == pytest.approx(math.sqrt(19))  # 样本方差：38/(3-1)
    assert (summary["min"], summary["max"], summary["n"]) == (1, 9, 3)


@pytest.mark.parametrize("seeds", [(1234,), (1234, 1235), (1234, 1234, 1236)])
def test_incomplete_or_duplicate_seeds_are_rejected(seeds):
    rows = [{"dataset": "mnist", "ws": 2, "seed": seed, "value": 1} for seed in seeds]
    with pytest.raises(ValueError):
        summarize_seeds(rows, metrics=("value",))


def test_epoch_statistics_keep_observed_range_separate_from_ci():
    rows = [
        {
            "dataset": "mnist",
            "ws": 2,
            "epoch": 1,
            "seed": seed,
            "train_loss": loss,
            "val_loss": loss,
            "val_acc": 0.9,
        }
        for seed, loss in zip((1234, 1235, 1236), (1, 2, 9))
    ]
    statistics = epoch_statistics(pd.DataFrame(rows)).set_index("metric")
    assert statistics.loc["train_loss", "mean"] == 4
    assert statistics.loc["train_loss", "min"] == 1
    assert statistics.loc["train_loss", "max"] == 9
    assert statistics.loc["val_acc", "std"] == pytest.approx(0, abs=1e-15)


def test_three_seed_ci_uses_student_t_not_observed_range_or_normal_approximation():
    statistics = three_seed_statistics([1.0, 2.0, 3.0])
    # 独立查表值：t(0.975,2)=4.3026527299，样本标准差为1。
    expected_half_width = 4.3026527299 / math.sqrt(3)
    assert statistics["ci95_low"] == pytest.approx(2 - expected_half_width)
    assert statistics["ci95_high"] == pytest.approx(2 + expected_half_width)
    assert statistics["ci95_low"] < statistics["min"]
    assert statistics["ci95_high"] > statistics["max"]


def test_ci_is_recomputed_at_each_epoch_and_can_collapse_with_zero_variation():
    rows = [
        {
            "dataset": "mnist",
            "ws": 2,
            "epoch": epoch,
            "seed": seed,
            "train_loss": loss,
            "val_loss": loss,
            "val_acc": 0.9,
        }
        for epoch, values in ((1, (1, 2, 3)), (2, (10, 10, 10)))
        for seed, loss in zip((1234, 1235, 1236), values)
    ]
    summary = (
        epoch_statistics(pd.DataFrame(rows))
        .query("metric == 'train_loss'")
        .set_index("epoch")
    )
    assert summary.loc[1, "ci95_low"] < 1 and summary.loc[1, "ci95_high"] > 3
    assert summary.loc[2, "ci95_low"] == summary.loc[2, "ci95_high"] == 10
    assert summary.loc[1, "ci95_low"] != summary.loc[2, "ci95_low"]


def test_one_seed_does_not_have_a_repeat_based_confidence_interval():
    with pytest.raises(ValueError):
        three_seed_statistics([1.0])


def test_final_plots_use_seed_and_mean_bars_without_intervals(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from E0 import plot_results

    finals, epochs = [], []
    for dataset in ("mnist", "fashionmnist", "cifar10"):
        for ws in (1, 2):
            for index, seed in enumerate((1234, 1235, 1236)):
                finals.append(
                    {
                        **dict.fromkeys(SUMMARY_METRICS, 1.0),
                        "dataset": dataset,
                        "ws": ws,
                        "seed": seed,
                        "test_acc": 0.9 + index * 0.01,
                        "train_throughput": 1000 + 100 * index,
                    }
                )
                for epoch in range(1, 21):
                    epochs.append(
                        {
                            "dataset": dataset,
                            "ws": ws,
                            "seed": seed,
                            "epoch": epoch,
                            "train_loss": 1 / epoch + index * 0.002,
                            "val_loss": 1 / epoch + index * 0.003,
                            "val_acc": 0.8 + index * 0.01,
                        }
                    )
    monkeypatch.setattr(
        plot_results,
        "load_results",
        lambda results: (pd.DataFrame(finals), pd.DataFrame(epochs)),
    )
    fill_calls, error_positions, bar_heights = [], [], []
    fill, errorbar, bar = Axes.fill_between, Axes.errorbar, Axes.bar

    def checked_fill(self, x, *args, **kwargs):
        fill_calls.append(list(x))
        return fill(self, x, *args, **kwargs)

    def checked_errorbar(self, x, *args, **kwargs):
        error_positions.append(x)
        return errorbar(self, x, *args, **kwargs)

    def checked_bar(self, x, height, *args, **kwargs):
        bar_heights.append(list(height))
        return bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", checked_fill)
    monkeypatch.setattr(Axes, "errorbar", checked_errorbar)
    monkeypatch.setattr(Axes, "bar", checked_bar)
    plot_results.render(tmp_path)
    assert fill_calls and all(x == list(range(1, 21)) for x in fill_calls)
    assert not error_positions
    assert len(bar_heights) == 30  # 三个数据集 × 五个终值指标 × 单/双卡。
    assert all(
        len(values) == 4 and values[-1] == pytest.approx(sum(values[:3]) / 3)
        for values in bar_heights
    )


def test_plot_reader_rejects_incomplete_seed_matrix(tmp_path):
    folder = tmp_path / "mnist"
    folder.mkdir()
    rows = [
        {
            **dict.fromkeys(SUMMARY_METRICS, 1.0),
            "run_id": f"seed{seed}_ws{ws}",
            "seed": seed,
            "ws": ws,
            "data": "mnist",
            "device": "cuda:0",
            "backend": "nccl" if ws == 2 else "none",
            "epochs_ran": 20,
            "test_epoch": 20,
        }
        for seed in (1234, 1235)
        for ws in (1, 2)
    ]
    pd.DataFrame(rows).to_csv(folder / "final.csv", index=False)
    with pytest.raises(ValueError, match="六次"):
        load_results(tmp_path)
