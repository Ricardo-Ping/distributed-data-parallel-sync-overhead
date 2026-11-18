"""E5仅生成CSV与统计图；未达目标保留删失状态。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from E5.config import DATASETS,SEEDS,RESULTS_ROOT,EXPERIMENT_ROOT,load_config,run_folder
from E5.check_results import load_final_runs
from E5.metrics import seed_summary

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
    baseline=frame[frame.level.eq(1)]
    matched=frame.merge(baseline,on=['dataset','seed'],suffixes=('','_h1'),validate='many_to_one')
    paired=matched[['dataset','level','seed']].copy()
    paired['throughput_speedup']=matched.train_throughput/matched.train_throughput_h1
    paired['sync_per_update_ratio']=matched.avg_sync_per_update_ms/matched.avg_sync_per_update_ms_h1
    paired['training_stage_time_ratio']=matched.training_stage_wall_s/matched.training_stage_wall_s_h1
    paired['test_accuracy_difference_pp']=(matched.test_acc-matched.test_acc_h1)*100
    paired['both_targets_reached']=matched.target_reached&matched.target_reached_h1
    paired['time_to_target_speedup']=(matched.time_to_target_s_h1/matched.time_to_target_s).where(paired.both_targets_reached)
    paired.to_csv(results/'paired_comparison.csv',index=False)
    folder=results/'figs';folder.mkdir(exist_ok=True)
    colors=('#2766A6','#C97424','#268467');markers=('o','s','^');sizes=(8,6,4)
    handles=[Line2D([],[],color=c,marker=m,markerfacecolor='none',label=f'种子 {s}') for s,c,m in zip(SEEDS,colors,markers)]
    handles.append(Line2D([],[],color='#222831',marker='D',markerfacecolor='none',linewidth=2.2,label='三种子均值'))
    def format_axis(ax,ds):
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.set_xticks([],minor=True)
        ax.set_xlabel('Local SGD 同步间隔 H');ax.set_title(f'({"abc"[DATASETS.index(ds)]}) {NAMES[ds]}');ax.grid(alpha=.17)
    def plot(ax,ds,metric,scale=1,source=None):
        source=frame if source is None else source;g=source[source.dataset.eq(ds)]
        for seed,color,marker,size in zip(SEEDS,colors,markers,sizes):
            v=g[g.seed.eq(seed)].sort_values('level')
            line,=ax.plot(v.level,v[metric]*scale,color=color,marker=marker,markerfacecolor='none',markersize=size,linewidth=.8,alpha=.7)
            line.set_gid(f'seed:{seed}')
        mean=g.groupby('level')[metric].mean().sort_index()
        line,=ax.plot(mean.index,mean*scale,color='#222831',marker='D',markerfacecolor='none',markersize=10,linewidth=2.2);line.set_gid('mean')
        format_axis(ax,ds);lo,hi=g[metric].min()*scale,g[metric].max()*scale
        if metric=='test_acc':
            margin=max(.3,(hi-lo)*.18);ax.set_ylim(max(0,lo-margin),min(100,hi+margin))
        else:
            upper=max(hi*1.15,1e-9)
            ax.set_ylim(0,min(100,upper) if metric in ('sync_share','gpu_gap_fraction') else upper)
    figures=[('F1_calls_payload.png','参数平均次数与逻辑载荷',
              [('total_model_allreduces','20轮参数归约次数',1),('total_logical_payload_bytes','每 rank 逻辑载荷（GB）',1e-9)]),
             ('F2_throughput_update.png','训练吞吐与更新摊销成本',
              [('train_throughput','训练吞吐（样本/秒）',1),('avg_update_ms','每次本地更新摊销时间（ms）',1)]),
             ('F3_sync_wait.png','同步路径与两卡到达差',
              [('avg_sync_ms','每次参数平均耗时（ms）',1),('sync_share','同步占轮次时间（%）',100),('arrival_gap_ms','两 rank 同步点到达差（ms）',1)]),
             ('F4_drift_quality.png','同步前模型漂移与末轮质量',
              [('drift_mean_relative_l2','同步前参数相对L2距离',1),('test_acc','末轮测试准确率（%）',100)])]
    for filename,title,metrics in figures:
        fig,axes=plt.subplots(len(metrics),3,figsize=(15,3.6*len(metrics)+1),squeeze=False)
        for row,(metric,label,scale) in enumerate(metrics):
            for col,ds in enumerate(DATASETS): plot(axes[row,col],ds,metric,scale);axes[row,col].set_ylabel(label)
        fig.suptitle('E5 '+title,fontsize=15);fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False)
        fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(folder/filename,dpi=180,bbox_inches='tight');plt.close(fig);print(folder/filename)
    # 收敛曲线按epoch对齐；不能把不同运行的墙钟坐标直接当作逐点均值。
    hcolors=('#2364A4','#C97424','#268467','#8F5BA6','#B64956')
    fig,axes=plt.subplots(2,3,figsize=(15,8.2),squeeze=False)
    for row,(metric,label,scale) in enumerate([('train_loss','训练交叉熵损失',1),('val_acc','完整验证集准确率（%）',100)]):
        for ax,ds in zip(axes[row],DATASETS):
            config=load_config(EXPERIMENT_ROOT/'configs'/f'{ds}.yaml')
            for h,color in zip(config.levels,hcolors):
                histories=[]
                for seed,marker in zip(SEEDS,markers):
                    v=pd.read_csv(run_folder(results,config,h,'local_sgd',seed)/'epochs.csv');histories.append(v[metric].to_numpy()*scale)
                    ax.plot(v.epoch,v[metric]*scale,color=color,alpha=.22,linewidth=.7,marker=marker,markersize=2,markevery=4)
                ax.plot(range(1,21),np.mean(histories,axis=0),color=color,linewidth=2,label=f'H={h}')
            ax.set_xticks([1,5,10,15,20]);ax.set_xlabel('训练 epoch');ax.set_ylabel(label);ax.set_title(f'({"abc"[DATASETS.index(ds)]}) {NAMES[ds]}');ax.grid(alpha=.17)
    fig.suptitle('E5 训练损失与验证收敛：细线为三个种子，粗线为逐轮均值',fontsize=14)
    convergence_handles=[Line2D([],[],color=c,linewidth=2,label=f'H={h}') for h,c in zip((1,2,4,8,16),hcolors)]
    convergence_handles +=[Line2D([],[],color='#666666',marker=m,linestyle='none',label=f'种子 {s}') for s,m in zip(SEEDS,markers)]
    fig.legend(handles=convergence_handles,loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.10,1,.94));fig.savefig(folder/'F5_validation.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # 未达目标显示20轮观测时长，用叉标记；均值仅在3/3均达到时显示。
    fig,axes=plt.subplots(1,3,figsize=(15,5));targets=[]
    for ax,ds in zip(axes,DATASETS):
        g=frame[frame.dataset.eq(ds)]
        for seed,color,marker in zip(SEEDS,colors,markers):
            v=g[g.seed.eq(seed)].sort_values('level');ax.plot(v.level,v.time_to_target_s,color=color,marker=marker,markerfacecolor='none',linewidth=.9)
            missed=v[~v.target_reached];ax.scatter(missed.level,missed.training_stage_wall_s,color=color,marker='x',s=45)
        means=[]
        for h,v in g.groupby('level'):
            reached=int(v.target_reached.sum());mean=v.time_to_target_s.mean() if reached==3 else np.nan
            means.append((h,mean));targets.append(dict(dataset=ds,level=h,reached=reached,total=3,
                           mean_time_s=mean,std_time_s=v.time_to_target_s.std() if reached==3 else np.nan))
        ax.plot([a for a,b in means],[b for a,b in means],color='#222831',marker='D',markerfacecolor='none',linewidth=2)
        observed=np.r_[g.time_to_target_s.dropna().to_numpy(),g.loc[~g.target_reached,'training_stage_wall_s'].to_numpy()]
        format_axis(ax,ds);upper=max(float(observed.max())*1.22,1e-9);ax.set_ylim(0,upper)
        for h,v in g.groupby('level'): ax.text(h,upper*.94,f'{int(v.target_reached.sum())}/3',ha='center',fontsize=9)
        ax.set_ylabel('达到验证目标的耗时（秒）')
        ax.set_title(f'({"abc"[DATASETS.index(ds)]}) {NAMES[ds]}，目标{g.target_accuracy.iloc[0]:.0%}')
    pd.DataFrame(targets).to_csv(results/'time_to_target_summary.csv',index=False)
    legend=handles+[Line2D([],[],color='#555555',marker='x',linestyle='none',label='未达到：20轮观测时长')]
    fig.suptitle('E5 Time-to-Target（含验证、诊断与日志；上方数字为达标种子数）',fontsize=13)
    fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.16,1,.92))
    fig.savefig(folder/'F6_time_to_target.png',dpi=180,bbox_inches='tight');plt.close(fig)
    profile_path=results/'profile_seed_summary.csv'
    if profile_path.exists():
        prof=pd.read_csv(profile_path);fig,axes=plt.subplots(2,3,figsize=(15,8.2),squeeze=False)
        for row,(metric,label,scale) in enumerate([('gpu_gap_fraction','无CUDA活动比例（%）',100),('nccl_kernel_ms','64步内NCCL kernel活动时长（ms）',1)]):
            for col,ds in enumerate(DATASETS): plot(axes[row,col],ds,metric,scale,prof);axes[row,col].set_ylabel(label)
        fig.suptitle('E5 独立Profiler诊断：两GPU先平均，再汇总三个种子',fontsize=14)
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False);fig.tight_layout(rect=(0,.08,1,.94))
        fig.savefig(folder/'F7_gpu_activity.png',dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',type=Path,default=RESULTS_ROOT);render(p.parse_args().results_dir)
