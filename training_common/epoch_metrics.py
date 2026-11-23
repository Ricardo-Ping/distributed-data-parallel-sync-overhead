"""Read newly committed timing rows and report the current epoch."""

from pathlib import Path

import pandas as pd


class EpochMetrics:
    def __init__(self, paths, *, time_column, samples_column):
        self.paths = tuple(paths)
        self.time_column = time_column
        self.samples_column = samples_column
        self.positions = [0] * len(self.paths)

    def read(self, epoch):
        frames = []
        for index, path in enumerate(self.paths):
            with Path(path).open("rb") as stream:
                header = stream.readline().decode().strip().split(",")
                stream.seek(self.positions[index] or stream.tell())
                frames.append(pd.read_csv(stream, header=None, names=header))
                self.positions[index] = stream.tell()

        frame = pd.concat(frames, ignore_index=True)
        if frame.empty or not frame.epoch.eq(epoch).all():
            raise ValueError(f"第 {epoch} 轮缺少对应的训练步记录")
        if len(self.paths) > 1:
            # Carry forward all fields from the same critical rank per update.
            frame = frame.loc[frame.groupby("iteration")[self.time_column].idxmax()]
        steady = frame[frame.measured]
        updates = len(steady)
        samples = int(steady[self.samples_column].sum()) * len(self.paths)
        elapsed = float(steady[self.time_column].sum())
        compute, sync = (float(steady[key].sum()) for key in ("t_compute_ms", "t_sync_ms"))
        return {
            "measured_updates": updates,
            "measured_samples": samples,
            "train_throughput": 1000 * samples / elapsed if elapsed else None,
            "avg_update_ms": elapsed / updates if updates else None,
            "avg_compute_ms": compute / updates if updates else None,
            "avg_sync_ms": sync / updates if updates else None,
            "sync_share": sync / elapsed if elapsed else None,
            "sync_compute_ratio": sync / compute if compute else None,
            "sync_us_per_sample": 1000 * sync / samples if samples else None,
        }


def format_epoch_metrics(metrics, *, scale_experiment=False):
    if not metrics["measured_updates"]:
        return "performance=N/A (no measured updates)"

    common = (
        f"throughput={metrics['train_throughput']:.1f} samples/s  "
        f"update={metrics['avg_update_ms']:.3f}ms  "
        f"compute={metrics['avg_compute_ms']:.3f}ms  "
        f"sync={metrics['avg_sync_ms']:.3f}ms  "
        f"sync_share={metrics['sync_share']:.2%}"
    )
    if scale_experiment:
        ratio = metrics["sync_compute_ratio"]
        return common + (f"  S/C={ratio:.3f}" if ratio is not None else "  S/C=N/A")
    return common + f"  sync/sample={metrics['sync_us_per_sample']:.2f}us"
