"""E5：Local SGD，本地momentum保持独立，周期性平均模型参数。"""
from dataclasses import dataclass
from pathlib import Path
import yaml

EXPERIMENT='E5'
EXPERIMENT_ROOT=Path(__file__).resolve().parent
PROJECT_ROOT=EXPERIMENT_ROOT.parent
RESULTS_ROOT=EXPERIMENT_ROOT/'results'
DATASETS=('mnist','fashionmnist','cifar10')
SEEDS=(1234,1235,1236)
LEVELS=(1,2,4,8,16)
VARIANTS=('local_sgd',)

@dataclass(frozen=True)
class Config:
    dataset: str
    learning_rate: float
    target_accuracy: float
    micro_batch: int=128
    epochs: int=20
    world_size: int=2
    levels: tuple[int,...]=LEVELS
    variants: tuple[str,...]=VARIANTS
    seeds: tuple[int,...]=SEEDS
    momentum: float=.9
    validation_fraction: float=.1
    split_seed: int=1234
    eval_batch: int=1024
    warmup_updates: int=16
    host_threads: int=4
    device: str='cuda'

    def __post_init__(self):
        if self.dataset not in DATASETS or self.learning_rate<=0 or self.micro_batch<1 or not 0<self.target_accuracy<=1:
            raise ValueError('数据集、学习率、batch或目标准确率无效')
        if (self.epochs,self.world_size,self.device)!=(20,2,'cuda'):
            raise ValueError('E5固定20轮、双GPU与CUDA')
        if (self.levels,self.variants,self.seeds)!=(LEVELS,VARIANTS,SEEDS):
            raise ValueError('E5固定五档H、Local SGD及三个种子')
        if not 0<self.validation_fraction<1 or not 0<=self.momentum<1 or min(self.eval_batch,self.host_threads)<1 or self.warmup_updates<0:
            raise ValueError('训练配置无效')

    def effective_local_batch(self,level):
        if level not in self.levels: raise ValueError('H不在实验计划中')
        return self.micro_batch

    def global_batch_for(self,level): return self.world_size*self.effective_local_batch(level)

def load_config(path):
    values=yaml.safe_load(Path(path).read_text())
    for key in ('levels','variants','seeds'):
        if key in values: values[key]=tuple(values[key])
    return Config(**values)

def run_id(config,level,variant,seed): return f'e5_{config.dataset}_h{level}_seed{seed}'

def run_folder(output,config,level,variant,seed):
    return Path(output)/config.dataset/f'h{level}'/f'seed{seed}'

def resolve_checkpoint(path):
    for p in (Path(path),PROJECT_ROOT/path,EXPERIMENT_ROOT/path):
        if p.is_file(): return p.resolve()
    raise FileNotFoundError(path)
