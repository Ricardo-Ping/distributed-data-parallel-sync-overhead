"""只读核对单/双GPU报告表格、公式、图表来源与文内链接，不生成报告。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"MNIST": "mnist", "FashionMNIST": "fashionmnist", "CIFAR-10": "cifar10"}
FIELDS = (("single_throughput", 1e-3, 2), ("dual_throughput", 1e-3, 2),
          ("speedup", 1, 3), ("compute_speedup", 1, 3), ("parallel_efficiency", 100, 1))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_e5_main(root, text):
    section = text.split("## 6. 第2版实测结果", 1)[1].split("## 7.", 1)[0]
    summary = pd.read_csv(root / "seed_summary.csv")
    final = pd.read_csv(root / "final.csv")
    profiles = pd.read_csv(root / "profile_seed_summary.csv")
    counts = {"performance": 0, "time_to_target": 0, "profile": 0}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] not in NAMES:
            continue
        dataset, level = NAMES[cells[0]], int(cells[1])
        if len(cells) == 7:
            expected = []
            group = summary[summary.dataset.eq(dataset) & summary.level.eq(level)]
            for metric, scale, places in (("train_throughput", 1e-3, 2), ("avg_update_ms", 1, 2),
                                          ("sync_share", 100, 2), ("drift_mean_relative_l2", 1000, 3),
                                          ("test_acc", 100, 2)):
                row = group[group.metric.eq(metric)].iloc[0]
                expected.append(f'{row["mean"] * scale:.{places}f}±{row["std"] * scale:.{places}f}')
            assert cells[2:] == expected, (dataset, level, cells[2:], expected)
            counts["performance"] += 1
        elif len(cells) == 5:
            group = final[final.dataset.eq(dataset) & final.level.eq(level)].sort_values("seed")
            reached = group.target_reached.astype(bool)
            epochs = "、".join(str(int(value)) if pd.notna(value) else "未达" for value in group.target_epoch)
            value = group.time_to_target_s
            mean_std = f"{value.mean():.2f}±{value.std(ddof=1):.2f}" if reached.all() else "未全部达标"
            assert cells[2:] == [f"{int(reached.sum())}/3", epochs, mean_std]
            counts["time_to_target"] += 1
        elif len(cells) == 4:
            group = profiles[profiles.dataset.eq(dataset) & profiles.level.eq(level)]
            expected = [f'{group[key].mean() * scale:.2f}±{group[key].std(ddof=1) * scale:.2f}'
                        for key, scale in (("gpu_gap_fraction", 100), ("nccl_kernel_ms", 1))]
            assert cells[2:] == expected
            counts["profile"] += 1
    assert counts == {"performance": 15, "time_to_target": 15, "profile": 6}, counts
    return counts


def audit(experiment):
    root = ROOT / experiment / "results"
    report = root / "实验报告.md"
    text = report.read_text()
    frame = pd.read_csv(root / "scaling.csv")
    summary = pd.read_csv(root / "scaling_summary.csv")
    expected_pairs = {"E2": 45, "E3": 36, "E4": 90, "E5": 45}[experiment]
    assert len(frame) == expected_pairs
    assert not frame.duplicated(["dataset", "level", "variant", "seed"]).any()
    np.testing.assert_allclose(frame.speedup, frame.dual_throughput / frame.single_throughput, rtol=1e-12)
    np.testing.assert_allclose(frame.compute_speedup, frame.single_compute_ms / frame.dual_compute_ms, rtol=1e-12)
    np.testing.assert_allclose(frame.parallel_efficiency, frame.speedup / 2, rtol=1e-12)
    np.testing.assert_allclose(frame.time_saved_fraction, 1 - 1 / frame.speedup, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(frame.gpu_cost_ratio, 2 / frame.speedup, rtol=1e-12)
    for (dataset, level, variant), group in frame.groupby(["dataset", "level", "variant"]):
        assert sorted(group.seed) == [1234, 1235, 1236]
        available = summary[summary.dataset.eq(dataset) & summary.level.eq(level) & summary.variant.eq(variant)]
        for metric in available.metric:
            row = available[available.metric.eq(metric)].iloc[0]
            values = pd.to_numeric(group[metric], errors="coerce")
            if metric == "time_to_target_speedup" and not group.both_targets_reached.all():
                assert pd.isna(row["mean"]) and pd.isna(row["std"])
                continue
            assert int(row.n) == int(values.notna().sum())
            for key, value in (("mean", values.mean()), ("std", values.std(ddof=1)), ("min", values.min()), ("max", values.max())):
                assert math.isclose(row[key], value, rel_tol=1e-10, abs_tol=1e-10) or (pd.isna(row[key]) and pd.isna(value))

    marker = "逐配置匹配的单GPU对照"
    if marker not in text:
        raise ValueError(f"{report}缺少单GPU对照章节")
    section = text[text.index(marker):]
    checked = []
    ordinary_rows = 0
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if experiment == "E5" and len(cells) == 4 and cells[0] in NAMES:
            ordinary = pd.read_csv(root / "ordinary_sgd_reference.csv")
            group = ordinary[ordinary.dataset.eq(NAMES[cells[0]]) & ordinary.level.eq(int(cells[1]))]
            expected = [f'{group[key].mean()*scale:.{places}f}±{group[key].std(ddof=1)*scale:.{places}f}'
                        for key, scale, places in (("ordinary_throughput", 1e-3, 2), ("task_throughput_ratio", 1, 3))]
            assert cells[2:] == expected
            ordinary_rows += 1
            continue
        if len(cells) != 7 or cells[0] not in NAMES:
            continue
        dataset = NAMES[cells[0]]
        level = int(re.search(r"\d+", cells[1])[0])
        variant = next((v for v in ("every_micro", "window_end") if v in cells[1]), None)
        current = summary[summary.dataset.eq(dataset) & summary.level.eq(level)]
        if variant:
            current = current[current.variant.eq(variant)]
        else:
            if current.variant.nunique() != 1:
                raise ValueError("报告未区分同步方式")
            variant = current.variant.iloc[0]
        expected = []
        for metric, scale, places in FIELDS:
            row = current[current.metric.eq(metric)]
            assert len(row) == 1
            row = row.iloc[0]
            expected.append(f'{row["mean"] * scale:.{places}f}±{row["std"] * scale:.{places}f}')
        if cells[2:] != expected:
            raise ValueError(f"报告数值不匹配：{dataset}/{level}/{variant}: {cells[2:]} != {expected}")
        checked.append((dataset, level, variant))
    assert len(checked) == expected_pairs // 3 and len(set(checked)) == len(checked)

    links = []
    for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        target = match[1].strip().strip("<>")
        if target.startswith(("http:", "https:", "#", "mailto:")):
            continue
        path = (report.parent / target.split("#")[0]).resolve()
        if not path.exists() and path != (root / "scaling_report_validation.json").resolve():
            raise FileNotFoundError(f"文档链接不存在：{target}")
        links.append(target)
    figures = json.loads((root / "scaling_figure_validation.json").read_text())
    figure_rows = figures if isinstance(figures, list) else figures["figures"]
    for row in figure_rows:
        assert row["all_points_inside_axes"] and row["source_values_checked"]
        assert sha(root / "figs" / row["figure"]) == row["sha256"]
        if "source_sha256" in row:
            assert sha(root / row["source"]) == row["source_sha256"]
    if isinstance(figures, dict):
        for name, digest in figures["source_sha256"].items():
            assert sha(root / name) == digest
    output = {"experiment": experiment, "paired_rows_formula_checked": len(frame),
              "summary_rows_recomputed": len(summary), "report_scaling_rows_checked": len(checked),
              "local_links_checked": len(links), "figures_source_and_sha256_checked": len(figure_rows),
              "report_sha256": sha(report), "all_checks_passed": True}
    if experiment == "E5":
        output["E5_main_report_numeric_rows_checked"] = audit_e5_main(root, text)
        assert ordinary_rows == 6
        output["E5_ordinary_SGD_report_rows_checked"] = ordinary_rows
    (root / "scaling_report_validation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E2", "E3", "E4", "E5", "all"), default="all")
    args = parser.parse_args()
    for experiment in ("E2", "E3", "E4", "E5") if args.experiment == "all" else (args.experiment,):
        audit(experiment)
