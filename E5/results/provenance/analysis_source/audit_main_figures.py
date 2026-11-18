import hashlib
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

experiment = sys.argv[1]
root = Path(experiment) / 'results'
datasets = ('mnist', 'fashionmnist', 'cifar10')
seeds = (1234, 1235, 1236)
levels = (1, 2, 4, 8, 16)
records = []
save = Figure.savefig


def equal(actual, expected):
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


def audit(fig, filename, *args, **kwargs):
    name = Path(filename).name
    frame = pd.read_csv(root / 'final.csv')
    standard = {
        'E4': {
            'F1_calls_payload.png': [('total_gradient_allreduces', 1), ('total_logical_payload_bytes', 1e-9)],
            'F2_sync_cost.png': [('avg_sync_ms', 1), ('sync_share', 100), ('sync_us_per_sample', 1)],
            'F3_throughput_update.png': [('train_throughput', 1), ('avg_update_ms', 1)],
            'F4_quality_memory.png': [('test_acc', 100), ('peak_allocated_mb', 1)],
            'F5_paired_optimization.png': [('throughput_speedup', 1), ('sync_time_ratio', 1)],
        },
        'E5': {
            'F1_calls_payload.png': [('total_model_allreduces', 1), ('total_logical_payload_bytes', 1e-9)],
            'F2_throughput_update.png': [('train_throughput', 1), ('avg_update_ms', 1)],
            'F3_sync_wait.png': [('avg_sync_ms', 1), ('sync_share', 100), ('arrival_gap_ms', 1)],
            'F4_drift_quality.png': [('drift_mean_relative_l2', 1), ('test_acc', 100)],
            'F7_gpu_activity.png': [('gpu_gap_fraction', 100), ('nccl_kernel_ms', 1)],
        },
    }[experiment]
    points = seed_series = mean_series = 0
    for index, ax in enumerate(fig.axes):
        dataset = datasets[index % 3]
        if name in standard:
            metric, scale = standard[name][index // 3]
            if name == 'F5_paired_optimization.png':
                data = pd.read_csv(root / 'paired_comparison.csv')
            elif name == 'F7_gpu_activity.png':
                data = pd.read_csv(root / 'profile_seed_summary.csv')
            else:
                data = frame
            data = data[data.dataset.eq(dataset)]
            assert ax.get_xscale() == 'log'
            if experiment == 'E4':
                variants = ('paired',) if name == 'F5_paired_optimization.png' else ('every_micro', 'window_end')
                expected_gids = []
                for variant in variants:
                    group = data if variant == 'paired' else data[data.variant.eq(variant)]
                    for seed in seeds:
                        gid = f'seed:{variant}:{seed}'
                        expected_gids.append(gid)
                        line = next(l for l in ax.lines if l.get_gid() == gid)
                        equal(line.get_xdata(), levels)
                        equal(line.get_ydata(), group[group.seed.eq(seed)].sort_values('level')[metric] * scale)
                    gid = f'mean:{variant}'
                    expected_gids.append(gid)
                    line = next(l for l in ax.lines if l.get_gid() == gid)
                    equal(line.get_ydata(), group.groupby('level')[metric].mean().sort_index() * scale)
            else:
                expected_gids = [f'seed:{seed}' for seed in seeds] + ['mean']
                for seed in seeds:
                    line = next(l for l in ax.lines if l.get_gid() == f'seed:{seed}')
                    equal(line.get_xdata(), levels)
                    equal(line.get_ydata(), data[data.seed.eq(seed)].sort_values('level')[metric] * scale)
                equal(next(l for l in ax.lines if l.get_gid() == 'mean').get_ydata(),
                      data.groupby('level')[metric].mean().sort_index() * scale)
            assert sorted(l.get_gid() for l in ax.lines if l.get_gid()) == sorted(expected_gids)
            seed_series += sum(g.startswith('seed:') for g in expected_gids)
            mean_series += sum(g.startswith('mean') for g in expected_gids)
        elif name == 'F5_validation.png':
            assert len(ax.lines) == 20
            for i, level in enumerate(levels):
                histories = []
                for j, seed in enumerate(seeds):
                    history = pd.read_csv(root / dataset / f'h{level}' / f'seed{seed}' / 'epochs.csv')
                    metric, scale = ('train_loss', 1) if index < 3 else ('val_acc', 100)
                    histories.append(history[metric].to_numpy() * scale)
                    equal(ax.lines[i*4+j].get_xdata(), range(1,21))
                    equal(ax.lines[i*4+j].get_ydata(), histories[-1])
                equal(ax.lines[i*4+3].get_ydata(), np.mean(histories, axis=0))
            seed_series += 15
            mean_series += 5
        elif name == 'F6_time_to_target.png':
            assert len(ax.lines) == 4
            data = frame[frame.dataset.eq(dataset)]
            for i, seed in enumerate(seeds):
                group = data[data.seed.eq(seed)].sort_values('level')
                equal(ax.lines[i].get_ydata(), group.time_to_target_s)
                missed = group[~group.target_reached]
                equal(np.asarray(ax.collections[i].get_offsets()), missed[['level', 'training_stage_wall_s']].to_numpy())
            means = [group.time_to_target_s.mean() if group.target_reached.all() else np.nan for _, group in data.groupby('level')]
            equal(ax.lines[-1].get_ydata(), means)
            seed_series += 3
            mean_series += 1
        else:
            raise ValueError(name)
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        for line in ax.lines:
            # Exclude horizontal reference lines expressed in axes coordinates.
            if line.get_linestyle() == '--' and line.get_gid() is None:
                continue
            x,y = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            assert np.all((x[valid] >= xmin) & (x[valid] <= xmax))
            assert np.all((y[valid] >= ymin) & (y[valid] <= ymax))
            points += int(valid.sum())
        for collection in ax.collections:
            xy = np.asarray(collection.get_offsets())
            if xy.size:
                assert np.all((xy[:,0] >= xmin) & (xy[:,0] <= xmax))
                assert np.all((xy[:,1] >= ymin) & (xy[:,1] <= ymax))
    result = save(fig, filename, *args, **kwargs)
    records.append(dict(figure=name, seed_series=seed_series, mean_series=mean_series, points=points,
                        source_values_match=True, all_points_inside_axes=True,
                        sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest()))
    print('AUDITED', name, flush=True)
    return result


Figure.savefig = audit
importlib.import_module(f'{experiment}.plot_results').render()
(root / 'figure_validation.json').write_text(json.dumps(records, indent=2) + '\n')
