"""E4整次累积窗口计时与同步调用、逻辑载荷复算。"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd

PHASES=('t_data_ms','t_compute_ms','t_sync_ms','t_opt_ms','t_other_ms')
STEP_FIELDS=['epoch','step','iteration','micro_start','rank','full_batch','measured','local_samples',
             'micro_count','sync_micro_count',*PHASES,'t_update_ms','ar_count','ar_bytes','loss']
SUMMARY_METRICS=('train_throughput','avg_update_ms','avg_compute_ms','avg_sync_ms','sync_share',
                 'sync_us_per_sample','local_compute_throughput','peak_allocated_mb','test_acc','test_loss',
                 'total_gradient_allreduces','total_logical_payload_bytes','sync_calls_per_micro',
                 'avg_data_ms','avg_opt_ms','avg_other_ms')

def write_rows(path,rows,append=False):
    with Path(path).open('a' if append else 'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=STEP_FIELDS)
        if not append: writer.writeheader()
        writer.writerows(rows)

def summarize_steps(frame,config,level,variant,payload_bytes):
    if frame.empty or frame.duplicated(['iteration','rank']).any(): raise ValueError('记录为空或重复')
    if set(frame['rank'])!={0,1} or not frame.groupby('iteration')['rank'].nunique().eq(2).all():
        raise ValueError('rank记录不完整')
    if not np.isfinite(frame[[*PHASES,'t_update_ms','loss']]).all().all() or (frame[list(PHASES)]<0).any().any():
        raise ValueError('耗时或loss无效')
    if not np.allclose(frame[list(PHASES)].sum(axis=1),frame.t_update_ms,rtol=1e-10,atol=1e-9):
        raise ValueError('时间分段无法加和')
    micro=np.ceil(frame.local_samples/config.micro_batch).astype(int)
    counts=micro if variant=='every_micro' else np.ones(len(frame),dtype=int)
    full=frame.local_samples.eq(config.effective_local_batch(level))
    if not np.array_equal(frame.micro_count,micro) or not np.array_equal(frame.ar_count,counts):
        raise ValueError('micro数或同步次数不符合协议')
    if not np.array_equal(frame.sync_micro_count,counts) or not np.array_equal(frame.ar_bytes,counts*payload_bytes):
        raise ValueError('同步micro数或载荷不符')
    if not np.array_equal(frame.full_batch,full) or not np.array_equal(frame.measured,full&frame.micro_start.ge(config.warmup_micro_steps)):
        raise ValueError('完整窗口、预热标记不符')
    for field in ('local_samples','micro_count','ar_count','micro_start','measured'):
        if not frame.groupby('iteration')[field].nunique().eq(1).all(): raise ValueError('rank窗口不一致')
    selected=frame.loc[frame.groupby('iteration').t_update_ms.idxmax()].sort_values('iteration')
    starts=selected.micro_count.cumsum().shift(fill_value=0)
    if not np.array_equal(selected.micro_start,starts): raise ValueError('micro计数不连续')
    steady=selected[selected.measured]
    if steady.empty: raise ValueError('无稳态完整窗口')
    u=len(steady); n=int(steady.local_samples.sum())*2
    t=float(steady.t_update_ms.sum()); c=float(steady.t_compute_ms.sum()); s=float(steady.t_sync_ms.sum())
    return dict(total_updates=len(selected),total_train_samples=int(selected.local_samples.sum())*2,
                total_micro_steps=int(selected.micro_count.sum()),total_gradient_allreduces=int(selected.ar_count.sum()),
                total_logical_payload_bytes=int(selected.ar_bytes.sum()),measured_updates=u,measured_samples=n,
                measured_time_s=t/1000,train_throughput=1000*n/t,avg_update_ms=t/u,avg_compute_ms=c/u,
                avg_sync_ms=s/u,sync_share=s/t,sync_us_per_sample=s*1000/n,
                local_compute_throughput=(n/2)*1000/c,sync_calls_per_micro=steady.ar_count.sum()/steady.micro_count.sum(),
                avg_data_ms=steady.t_data_ms.mean(),avg_opt_ms=steady.t_opt_ms.mean(),avg_other_ms=steady.t_other_ms.mean())

def seed_summary(frame):
    rows=[]
    for (dataset,level,variant),group in frame.groupby(['dataset','level','variant']):
        if sorted(group.seed)!=[1234,1235,1236]: raise ValueError('每个配置须有三个种子')
        for metric in SUMMARY_METRICS:
            v=group[metric]
            rows.append(dict(dataset=dataset,level=level,variant=variant,metric=metric,n=3,
                             mean=v.mean(),std=v.std(ddof=1),min=v.min(),max=v.max()))
    return pd.DataFrame(rows)
