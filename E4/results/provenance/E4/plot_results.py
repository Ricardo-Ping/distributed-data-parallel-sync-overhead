"""E4仅生成CSV和统计图；实验报告直接维护。"""
import argparse
from pathlib import Path
import pandas as pd
from E4.config import DATASETS,SEEDS,RESULTS_ROOT
from E4.check_results import load_final_runs
from E4.metrics import seed_summary

NAMES={'mnist':'MNIST','fashionmnist':'FashionMNIST','cifar10':'CIFAR-10'}

def render(results=RESULTS_ROOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    for font in ('Noto Sans CJK SC','Noto Sans CJK JP','Microsoft YaHei','SimHei'):
        try: font_manager.findfont(font,fallback_to_default=False)
        except ValueError: continue
        plt.rcParams.update({'font.family':[font],'axes.unicode_minus':False});break
    else: raise RuntimeError('需要中文字体，例如Noto Sans CJK')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    frame=load_final_runs(results);frame.to_csv(results/'final.csv',index=False)
    seed_summary(frame).to_csv(results/'seed_summary.csv',index=False)
    paired=[]
    for (ds,k,seed),g in frame.groupby(['dataset','level','seed']):
        a=g[g.variant.eq('every_micro')].iloc[0];b=g[g.variant.eq('window_end')].iloc[0]
        paired.append(dict(dataset=ds,level=k,seed=seed,throughput_speedup=b.train_throughput/a.train_throughput,
                           sync_time_ratio=b.avg_sync_ms/a.avg_sync_ms,
                           allreduce_count_ratio=b.total_gradient_allreduces/a.total_gradient_allreduces))
    comparisons=pd.DataFrame(paired);comparisons.to_csv(results/'paired_comparison.csv',index=False)
    folder=results/'figs';folder.mkdir(exist_ok=True)
    variants=(('every_micro','#2766A6','每个 micro 同步'),('window_end','#C97424','仅窗口末次同步'))
    markers=('o','s','^');sizes=(7,5,3)
    handles=[Line2D([],[],color=color,linewidth=2,label=name) for _,color,name in variants]
    handles +=[Line2D([],[],color='#666666',marker=m,markerfacecolor='none',linestyle='none',label=f'种子 {s}') for s,m in zip(SEEDS,markers)]
    handles +=[Line2D([],[],color='#333333',marker='D',markerfacecolor='none',linewidth=2,label='粗线/菱形：三种子均值')]
    def axes_format(ax,ds,metric,values):
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.set_xticks([],minor=True)
        batch=int(frame.micro_batch.iloc[0]);ax.set_xlabel(f'累积次数 K（每卡 Micro-Batch={batch}）')
        low,high=min(values),max(values)
        if metric=='test_acc':
            margin=max(.3,(high-low)*.18);ax.set_ylim(max(0,low-margin),min(100,high+margin))
        else: ax.set_ylim(0,max(high*1.15,1e-9))
        ax.set_title(f'({"abc"[DATASETS.index(ds)]}) {NAMES[ds]}');ax.grid(alpha=.17)
    def draw(ax,ds,metric,scale,paired_plot=False):
        all_y=[]
        groups=[('paired','#333333','')] if paired_plot else variants
        for variant,color,_ in groups:
            g=comparisons[comparisons.dataset.eq(ds)] if paired_plot else frame[frame.dataset.eq(ds)&frame.variant.eq(variant)]
            for seed,marker,size in zip(SEEDS,markers,sizes):
                v=g[g.seed.eq(seed)].sort_values('level'); y=v[metric]*scale;all_y.extend(y)
                line,=ax.plot(v.level,y,color=color,marker=marker,markerfacecolor='none',markersize=size,linewidth=.8,alpha=.5)
                line.set_gid(f'seed:{variant}:{seed}')
            mean=g.groupby('level')[metric].mean().sort_index()
            line,=ax.plot(mean.index,mean*scale,color=color,marker='D',markerfacecolor='none',markersize=9,linewidth=2.2)
            line.set_gid(f'mean:{variant}')
        axes_format(ax,ds,metric,all_y)
        if paired_plot: ax.axhline(1,color='#888888',linestyle='--',linewidth=.8)
    figures=[('F1_calls_payload.png','同步次数与累计逻辑载荷',
              [('total_gradient_allreduces','20轮梯度归约次数',1),('total_logical_payload_bytes','每 rank 逻辑载荷（GB）',1e-9)],False),
             ('F2_sync_cost.png','同步成本与摊销',
              [('avg_sync_ms','同步耗时（ms/参数更新）',1),('sync_share','同步占窗口时间（%）',100),('sync_us_per_sample','同步成本（微秒/全局样本）',1)],False),
             ('F3_throughput_update.png','训练吞吐与完整累积窗口成本',
              [('train_throughput','训练吞吐（样本/秒）',1),('avg_update_ms','完整参数更新时间（ms）',1)],False),
             ('F4_quality_memory.png','末轮模型质量与训练显存',
              [('test_acc','末轮测试准确率（%）',100),('peak_allocated_mb','每卡训练峰值显存（MB）',1)],False),
             ('F5_paired_optimization.png','同K配对的通信优化收益',
              [('throughput_speedup','吞吐比：末次同步/每 micro 同步',1),('sync_time_ratio','同步耗时比：末次/每 micro',1)],True)]
    for filename,title,metrics,is_pair in figures:
        fig,axes=plt.subplots(len(metrics),3,figsize=(15,3.6*len(metrics)+1),squeeze=False)
        for row,(metric,label,scale) in enumerate(metrics):
            for col,ds in enumerate(DATASETS): draw(axes[row,col],ds,metric,scale,is_pair);axes[row,col].set_ylabel(label)
        fig.suptitle('E4 '+title,fontsize=15)
        legend=handles[2:] if is_pair else handles
        fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,.10 if len(metrics)==2 else .07,1,.95));fig.savefig(folder/filename,dpi=180,bbox_inches='tight');plt.close(fig)
        print(folder/filename)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',type=Path,default=RESULTS_ROOT);render(p.parse_args().results_dir)
