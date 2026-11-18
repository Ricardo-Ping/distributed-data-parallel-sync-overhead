"""E2：固定双GPU，五档每卡batch，三个数据集、三个种子，共45次运行。"""
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
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.multiprocessing as mp

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Data.datasets import DATA_ROOT
from E2.config import DATASETS, EXPERIMENT, EXPERIMENT_ROOT, PROJECT_ROOT, RESULTS_ROOT, SEEDS, Config, load_config, run_folder
from E2.train import train_worker


def schedule(config: Config) -> list[tuple[int, int]]:
    return [(batch, seed) for index, seed in enumerate(config.seeds)
            for batch in config.batch_sizes[index:]+config.batch_sizes[:index]]


def check_output(config: Config, jobs: list[tuple[int, int]], output: Path) -> None:
    for batch, seed in jobs:
        folder = run_folder(output, config, batch, seed)
        if folder.exists():
            raise FileExistsError(f'{folder} 已存在，请先归档或选择新的输出目录')


def record_environment(output: Path, configs: list[Config], data_root: Path) -> None:
    paths = sorted([*EXPERIMENT_ROOT.glob('*.py'), *EXPERIMENT_ROOT.glob('configs/*.yaml'), PROJECT_ROOT/'Data/datasets.py'])
    hashes = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    provenance = output/'provenance'
    if provenance.exists():
        if json.loads((provenance/'source_sha256.json').read_text()) != hashes:
            raise ValueError('代码快照已变化，请使用新的输出目录')
        return
    provenance.mkdir()
    for path in paths:
        target = provenance/path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (provenance/'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    def capture(args):
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout
    environment = dict(recorded_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), experiment=EXPERIMENT,
                       platform=platform.platform(), python=sys.version, pytorch=str(torch.__version__),
                       cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()),
                       gpu_details=capture(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,power.limit','--format=csv']),
                       topology=capture(['nvidia-smi','topo','-m']), host_context=capture(['lscpu']),
                       p2p_0_to_1=torch.cuda.can_device_access_peer(0,1), source_sha256=hashes,
                       data_root=str(data_root.resolve()), configs=[asdict(config) for config in configs],
                       cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','all'))
    (output/'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2)+'\n')


def run(configs: list[Config], output: Path = RESULTS_ROOT, data_root: Path = DATA_ROOT,
        seed: int | None = None, batch: int | None = None, *, list_only: bool = False) -> None:
    planned = [(config, [(b,s) for b,s in schedule(config) if (batch is None or b==batch) and (seed is None or s==seed)])
               for config in configs]
    for config, jobs in planned:
        for b,s in jobs:
            print(f'{EXPERIMENT}/{config.dataset}: seed={s}, GPUs=2, micro_batch={b}, k={config.accumulation_for(b)}, global_batch={config.global_batch_for(b)}, epochs=20', flush=True)
    if list_only:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count()<2 or not torch.distributed.is_nccl_available():
        raise RuntimeError('需要两张可见CUDA GPU与NCCL')
    if not any(jobs for _,jobs in planned):
        raise ValueError('没有匹配的配置')
    for config,jobs in planned:
        check_output(config,jobs,output)
    output.mkdir(parents=True,exist_ok=True)
    record_environment(output,configs,data_root)
    for config,jobs in planned:
        for b,s in jobs:
            folder=run_folder(output,config,b,s)
            folder.mkdir(parents=True)
            (folder/'config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',0))
                port=listener.getsockname()[1]
            mp.spawn(train_worker,args=(port,config,b,s,folder.resolve(),data_root.resolve()),nprocs=2,join=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=(*DATASETS,'all'),default='all')
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--batch',type=int)
    parser.add_argument('--output-dir',type=Path,default=RESULTS_ROOT)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args()
    configs=[load_config(EXPERIMENT_ROOT/'configs'/f'{name}.yaml') for name in (DATASETS if args.dataset=='all' else (args.dataset,))]
    run(configs,args.output_dir.resolve(),args.data_root,args.seed,args.batch,list_only=args.list)


if __name__=='__main__':
    main()
