"""E5检查完整实验网格、原始计时与末轮GPU复测。"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Data.datasets import DATA_ROOT
from E5.config import DATASETS,SEEDS,EXPERIMENT,EXPERIMENT_ROOT,RESULTS_ROOT,load_config,run_folder,run_id
from E5.model import load_checkpoint
from E5.evaluate import evaluate_checkpoint
from E5.metrics import summarize_rounds

def load_final_runs(results=RESULTS_ROOT):
    rows=[]
    for dataset in DATASETS:
        c=load_config(EXPERIMENT_ROOT/'configs'/f'{dataset}.yaml')
        for level in c.levels:
            for variant in c.variants:
                for seed in SEEDS:
                    row=json.loads((run_folder(results,c,level,variant,seed)/'final.json').read_text())
                    keys=('experiment','run_id','dataset','level','variant','seed','epochs_ran','test_epoch','selection','ws','global_batch')
                    expected=(EXPERIMENT,run_id(c,level,variant,seed),dataset,level,variant,seed,20,20,'last',2,c.global_batch_for(level))
                    if tuple(row[k] for k in keys)!=expected: raise ValueError('运行身份不符')
                    rows.append(row)
    return pd.DataFrame(rows)

def check(results=RESULTS_ROOT,data_root=DATA_ROOT,records_only=False):
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    finals=load_final_runs(results);checked=[];splits={};orders={}
    for final in finals.to_dict('records'):
        ds,level,variant,seed=final['dataset'],final['level'],final['variant'],final['seed']
        c=load_config(EXPERIMENT_ROOT/'configs'/f'{ds}.yaml');folder=run_folder(results,c,level,variant,seed)
        if load_config(folder/'config.json')!=c: raise ValueError('实际配置不同')
        n,nv,original=(45000,5000,50000) if ds=='cifar10' else (54000,6000,60000)
        ids=np.load(folder/'split_indices.npz');a,b=ids['train'],ids['validation']
        if (len(a),len(b))!=(n,nv) or not np.array_equal(np.sort(np.r_[a,b]),np.arange(original)): raise ValueError('数据划分错误')
        hashes=tuple(hashlib.sha256(v.tobytes()).hexdigest() for v in (a,b))
        meta=json.loads((folder/'dataset.json').read_text())
        if hashes!=(meta['train_indices_sha256'],meta['val_indices_sha256']): raise ValueError('划分哈希不符')
        if ds in splits and splits[ds]!=hashes: raise ValueError('配置间划分不同')
        splits[ds]=hashes;frames=[]
        steps=math.ceil(n/c.global_batch_for(level)); count=math.ceil(steps/level);effective=c.micro_batch*level
        expected_sizes=np.tile([min(effective,n//2-j*effective) for j in range(count)],20)
        for rank in (0,1):
            runtime=json.loads((folder/f'runtime_rank{rank}.json').read_text())
            if (runtime['device'],runtime['backend'],runtime['rank'],runtime['world_size'],runtime['dtype'],runtime['allow_tf32'])!=(f'cuda:{rank}','nccl',rank,2,'float32',False): raise ValueError('设备协议不符')
            f=pd.read_csv(folder/f'rounds_rank{rank}.csv');index=np.arange(count*20)
            if not np.array_equal(f.iteration,index) or not np.array_equal(f.epoch,index//count+1) or not np.array_equal(f['round'],index%count) or not np.array_equal(f.local_samples,expected_sizes): raise ValueError('样本、尾批或更新数错误')
            order=json.loads((folder/f'sampling_rank{rank}.json').read_text());key=(ds,seed,rank)
            if set(order)!=set(map(str,range(1,21))) or key in orders and orders[key]!=order: raise ValueError('采样顺序变化')
            orders[key]=order
            expected_steps=np.tile([min(level,steps-j*level) for j in range(count)],20)
            expected_first=np.tile(np.arange(count)*level,20)
            expected_global=(f.epoch.to_numpy()-1)*steps+expected_first
            diagnostic_end=(n//(2*c.micro_batch)//level)*level
            expected_drift=expected_first+expected_steps==diagnostic_end
            if not np.array_equal(f.steps_count,expected_steps) or not np.array_equal(f.first_step,expected_first) or not np.array_equal(f.global_first_step,expected_global): raise ValueError('本地SGD步数错误')
            if not np.array_equal(f.drift_measured,expected_drift) or int(f.drift_measured.sum())!=20: raise ValueError('漂移诊断窗口错误')
            if not np.isfinite(f.loc[f.drift_measured,'drift_relative_l2']).all(): raise ValueError('漂移值无效')
            frames.append(f)
            local=pd.read_csv(folder/f'steps_rank{rank}.csv')
            local_index=np.arange(steps*20)
            sizes=np.tile([min(c.micro_batch,n//2-j*c.micro_batch) for j in range(steps)],20)
            expected_epoch=local_index//steps+1; expected_step=local_index%steps
            expected_round=(expected_epoch-1)*count+expected_step//level
            sync=(expected_step+1)%level==0
            sync|=expected_step==steps-1
            if not np.array_equal(local.iteration,local_index) or not np.array_equal(local.epoch,expected_epoch) or not np.array_equal(local.step,expected_step):
                raise ValueError('每次本地训练步记录不完整')
            if not np.array_equal(local.local_samples,sizes) or not np.array_equal(local.round_iteration,expected_round):
                raise ValueError('本地训练步与同步轮次不匹配')
            if not np.array_equal(local.ar_count,sync.astype(int)) or not np.array_equal(local.sync_happened,sync):
                raise ValueError('本地步实际同步位置错误')
            if not np.array_equal(local.ar_bytes,local.ar_count*final['gradient_bytes']): raise ValueError('本地步同步载荷错误')
            if not np.array_equal(local.sync_type,np.where(sync,'parameter_allreduce','none')): raise ValueError('E5应记录参数平均而非梯度平均')
            phases=['t_data_ms','t_compute_ms','t_sync_ms','t_opt_ms','t_other_ms']
            if not np.isfinite(local[[*phases,'t_update_ms','local_throughput','loss']]).all().all() or (local[phases]<0).any().any():
                raise ValueError('本地步计时无效')
            if not np.allclose(local[phases].sum(axis=1),local.t_update_ms,rtol=1e-9,atol=1e-7): raise ValueError('本地步分段不加和')
            if not np.allclose(local.local_throughput,local.local_samples*1000/local.t_update_ms,rtol=1e-9): raise ValueError('本地步吞吐计算错误')
            sums=local.groupby('round_iteration')[[*phases,'t_update_ms','local_samples','ar_count','ar_bytes']].sum().sort_index()
            for column in (*phases,'local_samples','ar_count','ar_bytes'):
                if not np.allclose(sums[column],f[column],rtol=1e-9,atol=1e-7): raise ValueError(f'本地步与H轮次的{column}不一致')
            if not np.allclose(sums.t_update_ms,f.t_round_ms,rtol=1e-9,atol=1e-7): raise ValueError('本地步累计耗时与实际H轮次不一致')
        if final.get('timing_schema_version')!=2: raise ValueError('缺少实际本地训练步计时版本')
        model,saved=load_checkpoint(folder/'checkpoint.pt');size=model.size_info()
        if (saved['run_id'],saved['seed'],saved['level'],saved['variant'])!=(final['run_id'],seed,level,variant): raise ValueError('模型身份错误')
        if any(final[k]!=v for k,v in size.items()): raise ValueError('模型规模变化')
        for key in ('test_acc','test_loss','val_acc','val_loss'):
            if final[key]!=saved['metrics'][key]: raise ValueError('检查点指标不符')
        del model,saved;gc.collect()
        recomputed=summarize_rounds(pd.concat(frames,ignore_index=True),c,level,size['gradient_bytes'])
        for k,v in recomputed.items():
            if v is None and final[k] is None: continue
            if not math.isclose(final[k],v,rel_tol=1e-9,abs_tol=1e-9): raise ValueError(f'{final["run_id"]}: {k}复算不符')
        if final['total_train_samples']!=n*20 or final['total_updates']!=steps*20: raise ValueError('训练预算不符')
        if final['total_model_allreduces']!=count*20: raise ValueError('模型归约次数错误')
        history=pd.read_csv(folder/'epochs.csv')
        if list(history.epoch)!=list(range(1,21)) or any('test' in k for k in history): raise ValueError('验证测试未隔离')
        for k in ('train_loss','val_acc','val_loss'):
            if not math.isclose(final[k],history.iloc[-1][k],rel_tol=1e-9,abs_tol=1e-10): raise ValueError('最终指标非末轮')
        reached=history[history.val_acc.ge(c.target_accuracy)]
        if bool(len(reached))!=final['target_reached']: raise ValueError('目标达成标记错误')
        if len(reached):
            first=reached.iloc[0]
            if final['target_epoch']!=int(first.epoch) or not math.isclose(final['time_to_target_s'],first.elapsed_wall_s,rel_tol=1e-10): raise ValueError('Time-to-Target计算错误')
        elif final['target_epoch'] is not None or final['time_to_target_s'] is not None: raise ValueError('未达目标不应给出时间')
        if not history.elapsed_wall_s.is_monotonic_increasing: raise ValueError('墙钟记录非递增')
        for rank in (0,1):
            observed=frames[rank].loc[frames[rank].drift_measured,'drift_relative_l2'].to_numpy()
            if rank==0 and not np.allclose(history.drift_pre_sync_relative_l2,observed,rtol=1e-9,atol=1e-12): raise ValueError('逐轮漂移记录不符')
        record=dict(run_id=final['run_id'],records_ok=True)
        if not records_only:
            measured=evaluate_checkpoint(folder/'checkpoint.pt',device=f'cuda:{len(checked)%2}',data_root=data_root)
            if measured['accuracy']!=final['test_acc'] or not math.isclose(measured['loss'],final['test_loss'],rel_tol=1e-5,abs_tol=1e-6): raise ValueError('GPU复测不符')
            record['independent_test']=measured
        checked.append(record);print(f'[checked] {final["run_id"]}',flush=True)
    record=dict(experiment=EXPERIMENT,runs=len(checked),records_only=records_only,checked=checked,sample_order_consistent=True,tail_samples_preserved=True)
    (results/('records_validation.json' if records_only else 'validation.json')).write_text(json.dumps(record,indent=2)+'\n')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',type=Path,default=RESULTS_ROOT)
    p.add_argument('--data-root',type=Path,default=DATA_ROOT);p.add_argument('--records-only',action='store_true');args=p.parse_args()
    check(args.results_dir,args.data_root,args.records_only)
