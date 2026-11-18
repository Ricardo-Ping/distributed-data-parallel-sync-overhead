"""E3仅生成图表和CSV，报告直接维护。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from E3.check_results import load_final_runs
from E3.config import DATASETS, EXPERIMENT, RESULTS_ROOT, SEEDS
from E3.metrics import seed_summary

NAMES={'mnist':'MNIST','fashionmnist':'FashionMNIST','cifar10':'CIFAR-10'}


def render(results: Path=RESULTS_ROOT) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    for font in ('Noto Sans CJK SC','Noto Sans CJK JP','Microsoft YaHei','SimHei'):
        try:
            font_manager.findfont(font,fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams.update({'font.family':[font],'axes.unicode_minus':False})
        break
    else:
        raise RuntimeError('需要中文字体，例如fonts-noto-cjk')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    frame=load_final_runs(results)
    summary=seed_summary(frame)
    frame.to_csv(results/'final.csv',index=False)
    summary.to_csv(results/'seed_summary.csv',index=False)
    paired=[]
    for dataset,group in frame.groupby('dataset'):
        base=group[group.micro_batch.eq(128)].set_index('seed')
        for row in group.to_dict('records'):
            paired.append(dict(dataset=dataset,micro_batch=row['micro_batch'],accumulation=row['accumulation'],seed=row['seed'],
                               throughput_relative=row['train_throughput']/base.loc[row['seed'],'train_throughput'],
                               update_time_relative=row['avg_update_ms']/base.loc[row['seed'],'avg_update_ms']))
    pd.DataFrame(paired).to_csv(results/'paired_relative.csv',index=False)
    folder=results/'figs'
    folder.mkdir(exist_ok=True)
    colors=('#2563A6','#C76A13','#268467')
    markers=('o','s','^')
    marker_sizes=(8,6,4)
    handles=[Line2D([],[],color=color,marker=marker,markersize=size,markerfacecolor='none',label=f'种子 {seed}')
             for seed,color,marker,size in zip(SEEDS,colors,markers,marker_sizes)]
    handles.append(Line2D([],[],color='#20242A',marker='D',markersize=10,markerfacecolor='none',linewidth=2.1,label='三种子均值'))
    xkey='micro_batch'
    xlabel='每卡 Micro-Batch × 累积次数（全局 batch=256）'

    def plot(axis,dataset,metric,scale=1):
        group=frame[frame.dataset.eq(dataset)]
        for seed,color,marker,size in zip(SEEDS,colors,markers,marker_sizes):
            values=group[group.seed.eq(seed)].sort_values(xkey)
            axis.plot(values[xkey],values[metric]*scale,color=color,marker=marker,linewidth=.9,alpha=.8,
                      markersize=size,markerfacecolor='none')
        means=group.groupby(xkey)[metric].mean().sort_index()
        axis.plot(means.index,means*scale,color='#20242A',marker='D',linewidth=2.1,markersize=10,markerfacecolor='none')
        ticks=sorted(group[xkey].unique())
        axis.set_xscale('log',base=2)
        axis.set_xticks(ticks,[f'{value}×{128//value}' for value in ticks])
        axis.set_xticks([],minor=True)
        axis.set_xlabel(xlabel)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.18)
        axis.set_title(NAMES[dataset])

    figures=[('F1_throughput.png','图1 固定全局批量的不同实现与训练吞吐',(('train_throughput','训练吞吐（样本/秒）',1),)),
             ('F2_update_compute.png','图2 更新时间与计算段效率',(('avg_update_ms','平均更新时间（毫秒/更新）',1),
                                                                        ('local_compute_throughput','每卡计算段吞吐（样本/计算秒）',1))),
             ('F3_sync_cost.png','图3 梯度同步的成本与摊销',(('avg_sync_ms','平均同步耗时（毫秒/更新）',1),
                                                                  ('sync_share','同步占更新时间（%）',100),
                                                                  ('sync_ms_per_sample','同步成本（微秒/全局样本）',1000))),
             ('F4_memory_accuracy.png','图4 训练显存与末轮测试结果',(('peak_allocated_mb','每卡训练峰值显存（MB）',1),
                                                                            ('test_acc','第20轮模型测试准确率（%）',100)))]
    for filename,title,metrics in figures:
        figure,axes=plt.subplots(len(metrics),3,figsize=(15,3.5*len(metrics)+1),squeeze=False)
        for row,(metric,label,scale) in enumerate(metrics):
            for column,dataset in enumerate(DATASETS):
                plot(axes[row,column],dataset,metric,scale)
                axes[row,column].set_ylabel(label)
        figure.suptitle(f'{EXPERIMENT} {title}',fontsize=15)
        figure.legend(handles=handles,loc='lower center',ncol=4,frameon=False)
        figure.tight_layout(rect=(0,.10 if len(metrics)==1 else .06,1,.94))
        figure.canvas.draw()
        figure.savefig(folder/filename,dpi=180,bbox_inches='tight')
        plt.close(figure)
        print(folder/filename)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=RESULTS_ROOT)
    render(parser.parse_args().results_dir)
