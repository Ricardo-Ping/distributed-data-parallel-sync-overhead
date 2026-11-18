"""检查 E0 原始记录与 GPU 模型，不生成或改写实验说明文档。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from E0.config import PROJECT_ROOT, RESULTS_ROOT, EXPERIMENTS, resolve_checkpoint
from E0.evaluate import evaluate_checkpoint
from E0.metrics import summarize_seeds

CONFIG = ["model", "ws", "k", "local_batch", "compress", "data", "lr"]


def record_checkpoint(path: str | Path) -> Path:
    try:
        return resolve_checkpoint(path)
    except FileNotFoundError as exc:
        raise ValueError(f"记录中的模型文件不存在：{path}") from exc


def load_final_runs(results_dir: Path, exp: str) -> pd.DataFrame | None:
    """Read fixed-20-epoch final-model results, with one row per run."""
    path = results_dir / exp / "final.csv"
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    required = {
        *CONFIG,
        "run_id",
        "seed",
        "test_loss",
        "test_acc",
        "best_val_loss",
        "best_val_acc",
        "epochs_ran",
        "best_epoch",
        "train_throughput",
        "avg_update_ms",
        "avg_sync_ms",
        "training_mode",
        "test_epoch",
        "last_val_loss",
        "last_val_acc",
        "checkpoint_path",
        "stopped_early",
        "sync_share",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"{path} 缺少新版 E0 字段：{sorted(missing)}")
    if rows.empty or rows.run_id.isna().any() or rows.run_id.duplicated().any():
        raise ValueError(f"{path} 没有完整结果，或 run_id 为空/重复")
    if rows[list(required)].isna().any().any():
        raise ValueError(f"{path} 存在未完成的最终结果")
    if (
        not rows.training_mode.eq("fixed_epochs").all()
        or not rows.epochs_ran.eq(20).all()
        or not rows.test_epoch.eq(20).all()
        or not rows.stopped_early.astype(str)
        .str.lower()
        .isin({"false", "0", "0.0"})
        .all()
    ):
        raise ValueError(f"{path} 不是完整的固定 20 轮末轮模型结果")
    return rows


def validate_traces(folder: Path, metadata: dict, finals: pd.DataFrame) -> None:
    """用逐步、逐轮记录和真实索引交叉检查最终结果，拒绝残缺或混用的数据。"""
    with np.load(folder / "split_indices.npz") as saved:
        train, validation = saved["train"], saved["validation"]
    for indices, key in ((train, "train"), (validation, "val")):
        if len(indices) != metadata[f"{key}_samples"]:
            raise ValueError(f"{folder}: {key} 索引数量不一致")
        if (
            hashlib.sha256(indices.tobytes()).hexdigest()
            != metadata[f"{key}_indices_sha256"]
        ):
            raise ValueError(f"{folder}: {key} 索引哈希不一致")
    union = np.sort(np.concatenate([train, validation]))
    if not np.array_equal(union, np.arange(metadata["original_train_samples"])):
        raise ValueError(f"{folder}: 训练/验证索引重复或未覆盖完整官方训练集")
    steps = pd.read_csv(folder / "runs.csv")
    epochs = pd.read_csv(folder / "epochs.csv")
    if set(steps.run_id) != set(finals.run_id) or set(epochs.run_id) != set(
        finals.run_id
    ):
        raise ValueError(f"{folder}: 逐步/逐轮/最终记录包含不同的运行")
    if epochs.test_acc.notna().any():
        raise ValueError(f"{folder}: 逐轮记录不应查看测试集")
    for final in finals.itertuples():
        if (
            final.training_mode != "fixed_epochs"
            or final.epochs_ran != 20
            or final.test_epoch != 20
            or str(final.stopped_early).lower() not in {"false", "0", "0.0"}
        ):
            raise ValueError(f"{final.run_id}: 必须固定训练并测试第 20 轮模型")
        checkpoint_path = record_checkpoint(final.checkpoint_path)
        if not checkpoint_path.is_file():
            raise ValueError(f"{final.run_id}: 缺少模型检查点 {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location="cuda:0", weights_only=True
        )
        if (
            checkpoint.get("format_version") != 1
            or checkpoint.get("epoch") != 20
            or checkpoint.get("selection") != "last"
            or checkpoint.get("run_id") != final.run_id
        ):
            raise ValueError(f"{final.run_id}: 检查点不是该运行第 20 轮末轮模型")
        trace = steps[steps.run_id == final.run_id]
        history = epochs[epochs.run_id == final.run_id].sort_values("epoch")
        if history.epoch.tolist() != list(range(final.epochs_ran)):
            raise ValueError(f"{final.run_id}: 轮数缺失或重复")
        n_updates = math.ceil(metadata["train_samples"] / 256)
        for epoch, group in trace.groupby("epoch", sort=True):
            group = group.sort_values("step")
            if group.step.tolist() != list(range(n_updates)):
                raise ValueError(f"{final.run_id}, epoch={epoch}: 更新记录缺失或重复")
            expected_samples = [final.local_batch] * (n_updates - 1)
            expected_samples += [(metadata["train_samples"] % 256 or 256) // final.ws]
            if group.samples.tolist() != expected_samples:
                raise ValueError(f"{final.run_id}, epoch={epoch}: 样本数量或尾批不一致")
        if set(trace.epoch) != set(history.epoch):
            raise ValueError(f"{final.run_id}: 训练步与轮次不一致")
        best = history.loc[history.val_loss.idxmin()]
        comparisons = {
            "total_train_samples": trace.samples.sum() * final.ws,
            "total_updates": len(trace),
            "train_time_s": trace.t_step_ms.sum() / 1000,
            "avg_update_ms": trace.t_step_ms.mean(),
            "avg_sync_ms": trace.t_sync_ms.mean(),
            "sync_share": trace.t_sync_ms.sum() / trace.t_step_ms.sum(),
            "train_throughput": trace.samples.sum()
            * final.ws
            * 1000
            / trace.t_step_ms.sum(),
            "best_epoch": best.epoch + 1,
            "best_val_loss": best.val_loss,
            "best_val_acc": best.val_acc,
            "train_loss": history.iloc[-1].train_loss,
            "last_val_loss": history.iloc[-1].val_loss,
            "last_val_acc": history.iloc[-1].val_acc,
        }
        for field in ("test_loss", "test_acc"):
            comparisons[field] = checkpoint["metrics"][field]
        for final_field, checkpoint_field in (
            ("last_val_loss", "val_loss"),
            ("last_val_acc", "val_acc"),
        ):
            if not math.isclose(
                float(getattr(final, final_field)),
                float(checkpoint["metrics"][checkpoint_field]),
                rel_tol=1e-9,
                abs_tol=1e-7,
            ):
                raise ValueError(f"{final.run_id}: {final_field} 与检查点不一致")
        for field, value in comparisons.items():
            if not math.isclose(
                float(getattr(final, field)), float(value), rel_tol=1e-9, abs_tol=1e-7
            ):
                raise ValueError(f"{final.run_id}: {field} 与原始记录不一致")
        if not np.isfinite(trace[["t_step_ms", "t_sync_ms", "loss"]].to_numpy()).all():
            raise ValueError(f"{final.run_id}: 计时或训练损失出现非有限值")
        if final.ws == 1 and (
            trace.t_sync_ms.ne(0).any() or trace.ar_count.ne(0).any()
        ):
            raise ValueError(f"{final.run_id}: 单进程梯度同步应为零")


def validate_gpu_runtime(folder: Path, finals: pd.DataFrame) -> None:
    """核验各 rank 的 CUDA 设备、NCCL 后端与计时方式。"""
    fields = ["device", "backend", "gpu_name", "timing_mode"]
    if (
        any(field not in finals for field in fields)
        or finals[fields].isna().any().any()
    ):
        raise ValueError(f"{folder}: 缺少 GPU 运行记录")
    traces = pd.read_csv(folder / "runs.csv")
    histories = pd.read_csv(folder / "epochs.csv")
    if any(field not in traces or field not in histories for field in fields):
        raise ValueError(f"{folder}: 缺少逐步/逐轮设备记录")
    for final in finals.itertuples():
        backend = "nccl" if final.ws == 2 else "none"
        if (
            final.device != "cuda:0"
            or final.backend != backend
            or final.timing_mode != "cuda_synchronized_wall"
        ):
            raise ValueError(f"{final.run_id}: 需要 CUDA、NCCL 双进程和同步墙钟计时")
        for records in (traces, histories):
            group = records[records.run_id == final.run_id]
            for field in fields:
                if not group[field].eq(getattr(final, field)).all():
                    raise ValueError(f"{final.run_id}: {field} 与逐步/逐轮记录不一致")
        for rank in range(int(final.ws)):
            path = folder / "runtime" / f"{final.run_id}_rank{rank}.json"
            runtime = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "run_id": final.run_id,
                "rank": rank,
                "local_rank": rank,
                "world_size": final.ws,
                "seed": final.seed,
                "device": f"cuda:{rank}",
                "backend": backend,
                "timing_mode": final.timing_mode,
                "gpu_name": final.gpu_name,
            }
            if any(runtime.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{path}: rank/GPU/NCCL 运行信息不一致")
        checkpoint_path = record_checkpoint(final.checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cuda:0", weights_only=True
        )
        if checkpoint["training_config"].get("device") != "cuda":
            raise ValueError(f"{final.run_id}: 检查点未使用显式 CUDA 配置")
        trace = traces[traces.run_id == final.run_id]
        expected_bytes = sum(
            t.numel() * t.element_size() for t in checkpoint["state_dict"].values()
        )
        if final.ws == 2 and (
            not trace.ar_bytes.eq(expected_bytes).all()
            or not trace.ar_count.ge(1).all()
            or not trace.t_sync_ms.gt(0).all()
        ):
            raise ValueError(f"{final.run_id}: 双 GPU 每次更新应同步完整 FP32 梯度")
        if final.ws == 1 and not trace.ar_bytes.eq(0).all():
            raise ValueError(f"{final.run_id}: 单 GPU 梯度同步字节应为零")
        parts = trace[["t_data_ms", "t_fwd_bwd_ms", "t_sync_ms", "t_opt_ms"]]
        if (
            parts.lt(0).any().any()
            or not trace.t_step_ms.gt(0).all()
            or not np.allclose(parts.sum(axis=1), trace.t_step_ms, rtol=0, atol=0.0021)
        ):
            raise ValueError(f"{final.run_id}: 分段计时与总耗时不一致")


def validate_results(results: Path = RESULTS_ROOT, *, reevaluate: bool = True) -> dict:
    """检查三数据集18次记录，可选重载全部模型做独立 GPU 测试。"""
    evaluations, summaries, per_seed_statistics = [], [], []
    for dataset in EXPERIMENTS:
        folder = results / dataset
        rows = load_final_runs(results, dataset)
        if rows is None or len(rows) != 6 or rows.duplicated(["ws", "seed"]).any():
            raise ValueError(f"{dataset} 需要单/双 GPU × 三个种子的六次完整结果")
        if set(zip(rows.ws, rows.seed)) != {
            (ws, seed) for ws in (1, 2) for seed in (1234, 1235, 1236)
        }:
            raise ValueError(f"{dataset} 的 GPU/种子组合不完整")
        if not (rows.data.eq(dataset) & rows.model.eq("M") & rows.k.eq(1)).all():
            raise ValueError(f"{dataset} 的数据集、模型或累积设置不符合 E0")
        metadata = json.loads((folder / "dataset.json").read_text())
        sizes = (45000, 5000, 10000) if dataset == "cifar10" else (54000, 6000, 10000)
        if (
            tuple(
                metadata[key]
                for key in ("train_samples", "val_samples", "test_samples")
            )
            != sizes
        ):
            raise ValueError(f"{dataset} 的官方训练/验证/测试规模不正确")
        if metadata["split_seed"] != 1234 or metadata["validation_fraction"] != 0.1:
            raise ValueError(f"{dataset} 的验证划分协议不正确")
        for key in ("train_samples", "val_samples", "test_samples", "split_seed"):
            if not rows[key].eq(metadata[key]).all():
                raise ValueError(f"{dataset}: {key} 与划分记录不一致")
        validate_traces(folder, metadata, rows)
        validate_gpu_runtime(folder, rows)
        for final in rows.itertuples():
            checkpoint = resolve_checkpoint(final.checkpoint_path)
            relative = (
                str(checkpoint.relative_to(PROJECT_ROOT))
                if checkpoint.is_relative_to(PROJECT_ROOT)
                else str(checkpoint)
            )
            record = {
                "run_id": final.run_id,
                "checkpoint": relative,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            if reevaluate:
                measured = evaluate_checkpoint(
                    checkpoint, device=f"cuda:{int(final.ws) - 1}"
                )
                if (
                    measured["samples"] != 10000
                    or measured["accuracy"] != final.test_acc
                    or not math.isclose(
                        measured["loss"], final.test_loss, abs_tol=1e-6, rel_tol=1e-5
                    )
                ):
                    raise ValueError(f"{final.run_id}: 独立 GPU 测试与记录不符")
                record["gpu_test"] = measured
                print(
                    f"[verified] {final.run_id}: {measured['device']} acc={measured['accuracy']:.4%}",
                    flush=True,
                )
            evaluations.append(record)
        statistics = summarize_seeds(rows.assign(dataset=dataset).to_dict("records"))
        per_seed_statistics.extend(statistics)
        means = {(row["ws"], row["metric"]): row["mean"] for row in statistics}
        summaries.append(
            {
                "dataset": dataset,
                "single_gpu_samples_s": means[1, "train_throughput"],
                "dual_gpu_samples_s": means[2, "train_throughput"],
                "dual_single_ratio": means[2, "train_throughput"]
                / means[1, "train_throughput"],
                "single_test_acc": means[1, "test_acc"],
                "dual_test_acc": means[2, "test_acc"],
            }
        )
    return {
        "runs": len(evaluations),
        "gpu_reevaluated": reevaluate,
        "evaluations": evaluations,
        "aggregation": "arithmetic mean across seeds 1234, 1235, 1236",
        "dispersion": "sample standard deviation (ddof=1) and observed min-max",
        "confidence_interval": "95% Student-t mean interval: mean +/- t(0.975,df=2)*std/sqrt(3); independent approximately normal repetitions assumed",
        "seed_statistics": per_seed_statistics,
        "summary": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--records-only", action="store_true", help="只校验记录，不执行模型测试"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    record = validate_results(args.results_dir, reevaluate=not args.records_only)
    filename = "records_validation.json" if args.records_only else "validation.json"
    (args.results_dir / filename).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"E0: {record['runs']} 次运行通过核验；模型复测={record['gpu_reevaluated']}")


if __name__ == "__main__":
    main()
