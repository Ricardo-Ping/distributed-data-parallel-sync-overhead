"""E4独立启动器：五档K×两种同步方式×三数据集×三种子。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import torch
import torch.multiprocessing as mp
from Data.datasets import DATA_ROOT
from E4.config import DATASETS,EXPERIMENT,EXPERIMENT_ROOT,PROJECT_ROOT,RESULTS_ROOT,SEEDS,load_config,run_folder
from E4.train import train_worker

def schedule(config):
    jobs=[]
    for index,seed in enumerate(config.seeds):
        levels=config.levels[index:]+config.levels[:index]
        for offset,level in enumerate(levels):
            variants=config.variants if (index+offset)%2==0 else config.variants[::-1]
            jobs.extend((level,variant,seed) for variant in variants)
    return jobs

def completed(folder,config,level,variant,seed):
    if not folder.exists(): return False
    names=('final.json','checkpoint.pt','config.json','epochs.csv','dataset.json','split_indices.npz',
           'runtime_rank0.json','runtime_rank1.json','sampling_rank0.json','sampling_rank1.json')
    names+=tuple(f'steps_rank{r}.csv' for r in (0,1))
    if not all((folder/name).is_file() for name in names): raise FileExistsError(f'{folder}不完整，请先归档')
    final=json.loads((folder/'final.json').read_text())
    expected=(EXPERIMENT,config.dataset,level,variant,seed,20,20,'last',2)
    actual=tuple(final.get(k) for k in ('experiment','dataset','level','variant','seed','epochs_ran','test_epoch','selection','ws'))
    if actual!=expected or load_config(folder/'config.json')!=config: raise ValueError('已有结果与计划不同')
    return True

def record_environment(output,configs,data_root):
    files=sorted([*EXPERIMENT_ROOT.glob('*.py'),*EXPERIMENT_ROOT.glob('configs/*.yaml'),PROJECT_ROOT/'Data/datasets.py'])
    hashes={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    provenance=output/'provenance'
    if (provenance/'source_sha256.json').exists():
        old=json.loads((provenance/'source_sha256.json').read_text())
        # 分析脚本可在训练后改图；续跑严格核对所有训练与配置来源。
        core={k:v for k,v in hashes.items() if not k.endswith(('plot_results.py','check_results.py','profile_activity.py'))}
        old_core={k:v for k,v in old.items() if not k.endswith(('plot_results.py','check_results.py','profile_activity.py'))}
        if core!=old_core: raise ValueError('训练代码快照变化，请使用新输出目录')
        return
    provenance.mkdir(parents=True,exist_ok=True)
    for p in files:
        target=provenance/p.relative_to(PROJECT_ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    (provenance/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    def capture(args): return subprocess.run(args,capture_output=True,text=True,check=True).stdout
    environment=dict(recorded_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),experiment=EXPERIMENT,
        platform=platform.platform(),python=sys.version,pytorch=str(torch.__version__),cuda=torch.version.cuda,
        nccl=list(torch.cuda.nccl.version()),gpu_details=capture(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv']),
        topology=capture(['nvidia-smi','topo','-m']),host_context=capture(['lscpu']),
        p2p_0_to_1=torch.cuda.can_device_access_peer(0,1),source_sha256=hashes,data_root=str(data_root),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','all'),configs=[asdict(c) for c in configs])
    (output/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')

def launch(config,level,variant,seed,folder,data_root):
    for attempt in range(3):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        try:
            mp.spawn(train_worker,args=(port,config,level,variant,seed,folder.resolve(),data_root.resolve()),nprocs=2,join=True)
            return
        except mp.ProcessRaisedException as error:
            if attempt==2 or not {p.name for p in folder.iterdir()}<={'config.json'} or 'EADDRINUSE' not in str(error): raise
            print(f'[retry] NCCL端口{port}被占用，重新启动初始化',flush=True)

def run(configs,output=RESULTS_ROOT,data_root=DATA_ROOT,*,seed=None,level=None,variant=None,list_only=False,resume=False):
    jobs=[(c,l,v,s) for c in configs for l,v,s in schedule(c)
          if (seed is None or s==seed) and (level is None or l==level) and (variant is None or v==variant)]
    for c,l,v,s in jobs:
        print(f'{EXPERIMENT}/{c.dataset}: level={l}, mode={v}, seed={s}, GPUs=2, batch={c.micro_batch}, global={c.global_batch_for(l)}, epochs=20',flush=True)
    if list_only: return
    if resume: jobs=[(c,l,v,s) for c,l,v,s in jobs if not completed(run_folder(output,c,l,v,s),c,l,v,s)]
    if not jobs:
        if resume: print('[resume] 已全部完成',flush=True);return
        raise ValueError('没有匹配的运行')
    if not torch.cuda.is_available() or torch.cuda.device_count()<2 or not torch.distributed.is_nccl_available():
        raise RuntimeError('需要双CUDA GPU与NCCL')
    for c,l,v,s in jobs:
        folder=run_folder(output,c,l,v,s)
        if folder.exists(): raise FileExistsError(f'{folder}已存在，拒绝覆盖')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);record_environment(output,configs,data_root)
    for c,l,v,s in jobs:
        folder=run_folder(output,c,l,v,s);folder.mkdir(parents=True)
        (folder/'config.json').write_text(json.dumps(asdict(c),indent=2)+'\n')
        launch(c,l,v,s,folder,data_root)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=(*DATASETS,'all'),default='all');p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--level',type=int);p.add_argument('--variant');p.add_argument('--output-dir',type=Path,default=RESULTS_ROOT)
    p.add_argument('--data-root',type=Path,default=DATA_ROOT);p.add_argument('--list',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();datasets=DATASETS if args.dataset=='all' else (args.dataset,)
    configs=[load_config(EXPERIMENT_ROOT/'configs'/f'{d}.yaml') for d in datasets]
    run(configs,args.output_dir.resolve(),args.data_root,seed=args.seed,level=args.level,variant=args.variant,list_only=args.list,resume=args.resume)

if __name__=='__main__': main()
