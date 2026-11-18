"""与E0相同的MLP-M；保存第20轮模型。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from E5.config import Config, EXPERIMENT, resolve_checkpoint
from E5.distributed import cuda_device


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(nn.Linear(in_dim, 1024), nn.ReLU(), nn.Linear(1024, 512),
                                 nn.ReLU(), nn.Linear(512, 10))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def size_info(self) -> dict:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        return {"parameters": parameters, "gradient_bytes": parameters * 4, "in_dim": self.in_dim}


def save_checkpoint(path,model,config,level,variant,seed,identifier,metrics):
    torch.save({'format_version':1,'experiment':EXPERIMENT,'run_id':identifier,'epoch':20,'selection':'last',
                'dataset':config.dataset,'model':'M','in_dim':model.in_dim,'level':level,'variant':variant,
                'micro_batch':config.micro_batch,'global_batch':config.global_batch_for(level),
                'training_config':asdict(config),'seed':seed,'metrics':metrics,
                'state_dict':{k:v.detach().clone() for k,v in model.state_dict().items()}},path)


def load_checkpoint(path: str | Path, device: str = "cuda:0") -> tuple[MLP, dict]:
    saved = torch.load(resolve_checkpoint(path), map_location=cuda_device(device), weights_only=True)
    if (saved.get("experiment"), saved.get("epoch"), saved.get("selection"), saved.get("model")) != (EXPERIMENT, 20, "last", "M"):
        raise ValueError(f"需要{EXPERIMENT}第20轮MLP-M检查点")
    model = MLP(saved["in_dim"]).to(cuda_device(device))
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved
