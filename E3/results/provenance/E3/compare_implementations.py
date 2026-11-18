"""记录相同全局batch的末轮参数与测试结果差异，不要求长程训练逐位相同。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from E3.config import DATASETS, EXPERIMENT_ROOT, RESULTS_ROOT, SEEDS, load_config, run_folder
from E3.model import load_checkpoint


def compare(results: Path=RESULTS_ROOT) -> pd.DataFrame:
    rows=[]
    for dataset in DATASETS:
        config=load_config(EXPERIMENT_ROOT/'configs'/f'{dataset}.yaml')
        for seed in SEEDS:
            base,base_saved=load_checkpoint(run_folder(results,config,128,seed)/'checkpoint.pt')
            reference=base.state_dict()
            for batch in config.batch_sizes:
                actual,saved=load_checkpoint(run_folder(results,config,batch,seed)/'checkpoint.pt')
                squared,norm,maximum=0.0,0.0,0.0
                for key,value in actual.state_dict().items():
                    difference=(value-reference[key]).double()
                    squared+=difference.square().sum().item()
                    norm+=reference[key].double().square().sum().item()
                    maximum=max(maximum,difference.abs().max().item())
                rows.append(dict(dataset=dataset,seed=seed,micro_batch=batch,accumulation=config.accumulation_for(batch),
                                 max_absolute_parameter_difference=maximum,relative_parameter_l2=(squared/norm)**.5,
                                 test_accuracy_difference_pp=100*(saved['metrics']['test_acc']-base_saved['metrics']['test_acc']),
                                 test_loss_difference=saved['metrics']['test_loss']-base_saved['metrics']['test_loss']))
                del actual,saved
            del base,base_saved
    frame=pd.DataFrame(rows)
    frame.to_csv(results/'implementation_differences.csv',index=False)
    return frame


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=RESULTS_ROOT)
    print(compare(parser.parse_args().results_dir).to_string(index=False))
