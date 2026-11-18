"""真实数据的完整窗口、尾批梯度及SGD数值诊断，不计入36次正式训练。"""
from __future__ import annotations

import argparse
import copy
import json
import socket
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from Data.datasets import DATA_ROOT
from E3.config import DATASETS, EXPERIMENT_ROOT, RESULTS_ROOT, load_config
from E3.data import load_split
from E3.distributed import cleanup, initialize
from E3.train import Trainer


def worker(rank: int, port: int, output: Path, data_root: Path, scratch: Path) -> None:
    initialize(rank,2,port,4)
    rows=[]
    try:
        for dataset in DATASETS:
            config=load_config(EXPERIMENT_ROOT/'configs'/f'{dataset}.yaml')
            split=load_split(config,data_root)
            for batch in config.batch_sizes:
                trainer=Trainer(config,batch,1234,rank,scratch/dataset/f'b{batch}',split=split)
                reference=copy.deepcopy(trainer.model)
                optimizer=torch.optim.SGD(reference.parameters(),lr=config.learning_rate,momentum=config.momentum)
                features,labels=trainer.loader.dataset.tensors
                order=torch.randperm(len(labels),generator=torch.Generator().manual_seed(1234))
                global_batch=config.global_batch_for(batch)
                tail_start=(len(labels)//global_batch)*global_batch
                for name,ids in [('full',order[:global_batch]),('tail',order[tail_start:])]:
                    trainer.optimizer.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    trainer.synchronization.reset()
                    local=ids[rank::2]
                    trainer.backward_window(features[local],labels[local])
                    F.cross_entropy(reference(features[ids].to(trainer.device)),labels[ids].to(trainer.device)).backward()
                    difference,norm,maximum=0.0,0.0,0.0
                    for actual,expected in zip(trainer.model.parameters(),reference.parameters()):
                        torch.testing.assert_close(actual.grad,expected.grad,rtol=5e-4,atol=3e-6)
                        delta=actual.grad.double()-expected.grad.double()
                        difference+=delta.square().sum().item()
                        norm+=expected.grad.double().square().sum().item()
                        maximum=max(maximum,delta.abs().max().item())
                    trainer.optimizer.step()
                    optimizer.step()
                    parameter_maximum=0.0
                    for actual,expected in zip(trainer.model.parameters(),reference.parameters()):
                        torch.testing.assert_close(actual,expected,rtol=5e-4,atol=3e-6)
                        parameter_maximum=max(parameter_maximum,(actual-expected).abs().max().item())
                    assert trainer.synchronization.count==1
                    rows.append(dict(dataset=dataset,micro_batch=batch,accumulation=config.accumulation_for(batch),
                                     seed=1234,rank=rank,device=str(trainer.device),backend=dist.get_backend(),
                                     window=name,global_samples=len(ids),
                                     max_absolute_gradient_difference=maximum,relative_gradient_l2=(difference/norm)**.5,
                                     max_absolute_parameter_difference=parameter_maximum,gradient_allreduces=1))
                dist.barrier()
                del trainer,reference,optimizer,features,labels
            del split
        (output/f'numerical_equivalence_rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
    finally:
        cleanup()


def verify(output: Path=RESULTS_ROOT,data_root: Path=DATA_ROOT) -> pd.DataFrame:
    if not torch.cuda.is_available() or torch.cuda.device_count()<2 or not dist.is_nccl_available():
        raise RuntimeError('需要双CUDA GPU和NCCL')
    output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='e3_numerics_') as folder:
        mp.spawn(worker,args=(port,output.resolve(),data_root.resolve(),Path(folder)),nprocs=2,join=True)
    rows=[]
    for rank in (0,1):
        rows.extend(json.loads((output/f'numerical_equivalence_rank{rank}.json').read_text()))
    frame=pd.DataFrame(rows)
    assert len(frame)==48
    frame.to_csv(output/'numerical_equivalence.csv',index=False)
    return frame


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=RESULTS_ROOT)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    args=parser.parse_args()
    measured=verify(args.results_dir,args.data_root)
    print(measured.groupby('dataset')[['max_absolute_gradient_difference','relative_gradient_l2',
                                        'max_absolute_parameter_difference']].max().to_string())
