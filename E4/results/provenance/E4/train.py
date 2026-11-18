"""E4：相同累积窗口内，只改变梯度同步频率；每窗口只更新一次参数。"""
from __future__ import annotations
import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from Data.datasets import DATA_ROOT
from E4.config import EXPERIMENT,run_id
from E4.data import load_split,train_loader
from E4.distributed import initialize,cleanup,SyncStatistics,register_hook
from E4.evaluate import evaluate_model
from E4.metrics import summarize_steps,write_rows
from E4.model import MLP,save_checkpoint

class Trainer:
    def __init__(self,config,level,variant,seed,rank,folder,data_root=DATA_ROOT,*,split=None):
        self.config,self.level,self.variant,self.seed,self.rank=config,level,variant,seed,rank
        if variant not in config.variants: raise ValueError('未知同步方式')
        self.folder=Path(folder); self.folder.mkdir(parents=True,exist_ok=True)
        self.device=torch.device('cuda',rank); torch.cuda.set_device(self.device); torch.manual_seed(seed)
        self.split=load_split(config,data_root) if split is None else split
        self.loader=train_loader(self.split,config,rank,2,seed,level)
        self.model=MLP(self.split.raw.in_dim).to(self.device)
        self.network=DistributedDataParallel(self.model,device_ids=[rank],output_device=rank,bucket_cap_mb=config.bucket_cap_mb)
        self.synchronization=SyncStatistics(); register_hook(self.network,self.synchronization,2)
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=config.learning_rate,momentum=config.momentum)
        self.identifier=run_id(config,level,variant,seed)
        self.micro_iteration=0; self.peak_allocated=0; self.sampling_hashes={}
        runtime=dict(experiment=EXPERIMENT,run_id=self.identifier,device=str(self.device),backend=dist.get_backend(),
                     rank=rank,world_size=2,host_threads=torch.get_num_threads(),dtype='float32',allow_tf32=False,
                     level=level,variant=variant,micro_batch=config.micro_batch,global_batch=config.global_batch_for(level),
                     timing_mode='cuda_synchronized_accumulation_window',bucket_cap_mb=config.bucket_cap_mb)
        (self.folder/f'runtime_rank{rank}.json').write_text(json.dumps(runtime,indent=2)+'\n')
        if rank==0: self.split.save(self.folder,config)

    def backward_window(self,features,labels):
        data_ms,fb_ms,sync_micros=0.0,0.0,0
        size=len(labels); losses=[]; chunks=range(0,size,self.config.micro_batch)
        for offset in chunks:
            end=min(offset+self.config.micro_batch,size)
            start=time.perf_counter()
            x,y=features[offset:end].to(self.device),labels[offset:end].to(self.device)
            torch.cuda.synchronize(self.device)
            data_ms+=(time.perf_counter()-start)*1000
            last=end==size; before=self.synchronization.count; start=time.perf_counter()
            context=self.network.no_sync() if self.variant=='window_end' and not last else nullcontext()
            with context:
                loss=torch.nn.functional.cross_entropy(self.network(x),y)
                (loss*((end-offset)/size)).backward()
            torch.cuda.synchronize(self.device)
            fb_ms+=(time.perf_counter()-start)*1000
            synced=self.synchronization.count>before
            if synced!=(self.variant=='every_micro' or last): raise RuntimeError('梯度同步位置不符')
            sync_micros+=int(synced); losses.append((loss.detach(),end-offset))
        return data_ms,fb_ms,len(chunks),sync_micros,losses

    def train_epoch(self,epoch):
        self.model.train(); self.loader.sampler.set_epoch(epoch-1)
        ids=np.asarray(list(self.loader.sampler),dtype=np.int64)
        self.sampling_hashes[str(epoch)]=hashlib.sha256(ids.tobytes()).hexdigest()
        self.optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(self.device)
        rows=[]; loss_sum=0.; samples=0
        torch.cuda.synchronize(self.device); previous=time.perf_counter()
        for step,(features,labels) in enumerate(self.loader):
            window_start=previous; fetch_ms=(time.perf_counter()-window_start)*1000
            self.synchronization.reset()
            data_ms,fb_ms,micro_count,sync_micros,losses=self.backward_window(features,labels)
            start=time.perf_counter(); self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.device); end=time.perf_counter()
            opt_ms=(end-start)*1000; total_ms=(end-window_start)*1000
            sync=self.synchronization; compute_ms=fb_ms-sync.milliseconds
            data_ms+=fetch_ms; other_ms=total_ms-data_ms-fb_ms-opt_ms
            expected=micro_count if self.variant=='every_micro' else 1
            if min(compute_ms,other_ms)<-1e-7 or sync.count!=expected or sync_micros!=expected:
                raise RuntimeError('计时或同步次数无效')
            if sync.nbytes!=self.model.size_info()['gradient_bytes']*expected: raise RuntimeError('梯度载荷错误')
            size=len(labels); full=size==self.config.effective_local_batch(self.level)
            value=sum(v.item()*n for v,n in losses)/size
            rows.append(dict(epoch=epoch,step=step,iteration=(epoch-1)*len(self.loader)+step,
                             micro_start=self.micro_iteration,rank=self.rank,full_batch=full,
                             measured=full and self.micro_iteration>=self.config.warmup_micro_steps,
                             local_samples=size,micro_count=micro_count,sync_micro_count=sync_micros,
                             t_data_ms=data_ms,t_compute_ms=max(0.,compute_ms),t_sync_ms=sync.milliseconds,
                             t_opt_ms=opt_ms,t_other_ms=max(0.,other_ms),t_update_ms=total_ms,
                             ar_count=sync.count,ar_bytes=sync.nbytes,loss=value))
            self.micro_iteration+=micro_count; loss_sum+=value*size; samples+=size
            previous=time.perf_counter()
        self.peak_allocated=max(self.peak_allocated,torch.cuda.max_memory_allocated(self.device))
        write_rows(self.folder/f'steps_rank{self.rank}.csv',rows,append=epoch>1)
        (self.folder/f'sampling_rank{self.rank}.json').write_text(json.dumps(self.sampling_hashes,indent=2)+'\n')
        totals=torch.tensor([loss_sum,samples],device=self.device,dtype=torch.float64); dist.all_reduce(totals)
        return (totals[0]/totals[1]).item()

    def train(self):
        val_loader=DataLoader(self.split.subset('validation'),batch_size=self.config.eval_batch,shuffle=False)
        history=[]; started=time.perf_counter()
        for epoch in range(1,21):
            train_loss=self.train_epoch(epoch)
            if self.rank==0:
                val=evaluate_model(self.model,val_loader)
                if not math.isfinite(train_loss+val['loss']): raise RuntimeError('训练或验证非有限')
                history.append(dict(epoch=epoch,train_loss=train_loss,val_loss=val['loss'],val_acc=val['accuracy']))
                pd.DataFrame(history).to_csv(self.folder/'epochs.csv',index=False)
                print(f"[epoch] {self.identifier} {epoch}/20 val={val['loss']:.5f} acc={val['accuracy']:.4%}",flush=True)
            dist.barrier()
        (self.folder/f'memory_rank{self.rank}.json').write_text(json.dumps(dict(allocated_bytes=self.peak_allocated))+'\n')
        dist.barrier(); final=None
        if self.rank==0:
            test=evaluate_model(self.model,DataLoader(self.split.subset('test'),batch_size=self.config.eval_batch,shuffle=False))
            frame=pd.concat([pd.read_csv(self.folder/f'steps_rank{r}.csv') for r in (0,1)],ignore_index=True)
            size=self.model.size_info(); performance=summarize_steps(frame,self.config,self.level,self.variant,size['gradient_bytes'])
            memory=[json.loads((self.folder/f'memory_rank{r}.json').read_text()) for r in (0,1)]
            final=dict(experiment=EXPERIMENT,run_id=self.identifier,dataset=self.config.dataset,model='M',level=self.level,
                       variant=self.variant,accumulation=self.level,micro_batch=self.config.micro_batch,seed=self.seed,ws=2,
                       global_batch=self.config.global_batch_for(self.level),epochs_ran=20,test_epoch=20,selection='last',
                       train_samples=len(self.split.train_indices),val_samples=len(self.split.validation_indices),test_samples=test['samples'],
                       train_loss=train_loss,val_loss=val['loss'],val_acc=val['accuracy'],test_loss=test['loss'],test_acc=test['accuracy'],
                       peak_allocated_mb=max(m['allocated_bytes'] for m in memory)/1e6,wall_time_s=time.perf_counter()-started,
                       **size,**performance)
            save_checkpoint(self.folder/'checkpoint.pt',self.model,self.config,self.level,self.variant,self.seed,self.identifier,
                            {key:final[key] for key in ('val_loss','val_acc','test_loss','test_acc')})
            (self.folder/'final.json').write_text(json.dumps(final,indent=2)+'\n')
            print(f"[final] {self.identifier} Q={final['train_throughput']:.1f} sync={final['sync_share']:.2%}",flush=True)
        dist.barrier(); return final

def train_worker(rank,port,config,level,variant,seed,folder,data_root):
    initialize(rank,2,port,config.host_threads)
    try: Trainer(config,level,variant,seed,rank,folder,data_root).train()
    finally: cleanup()
