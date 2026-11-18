"""与双卡相同的全局排列；按偶/奇位置可还原原来的两个 rank。"""
from __future__ import annotations

import hashlib

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler


def global_loader(split, case):
    dataset = split.subset("train")
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=case.seed)
    return DataLoader(dataset, batch_size=case.global_batch, sampler=sampler,
                      num_workers=0, drop_last=False)


def sampling_record(sampler):
    order = np.asarray(list(sampler), dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError("单卡样本排列必须无遗漏、无重复")
    sha = lambda values: hashlib.sha256(values.tobytes()).hexdigest()
    return {"global_sha256": sha(order), "reference_rank0_sha256": sha(order[0::2]),
            "reference_rank1_sha256": sha(order[1::2])}


def serial_loaders(split, case):
    dataset = split.subset("train")
    if len(dataset) % 2:
        raise ValueError("串行双副本基线不补齐或重复训练样本")
    return [DataLoader(dataset, batch_size=case.micro_batch,
                       sampler=DistributedSampler(dataset, num_replicas=2, rank=replica,
                                                  shuffle=True, seed=case.seed),
                       num_workers=0, drop_last=False) for replica in (0, 1)]


def serial_sampling_record(loaders):
    orders = [np.asarray(list(loader.sampler), dtype=np.int64) for loader in loaders]
    if len(orders[0]) != len(orders[1]):
        raise ValueError("两个逻辑副本的样本数不同")
    global_order = np.empty(len(orders[0]) * 2, dtype=np.int64)
    global_order[0::2], global_order[1::2] = orders
    if not np.array_equal(np.sort(global_order), np.arange(len(global_order))):
        raise ValueError("串行双副本的样本存在遗漏或重复")
    sha = lambda values: hashlib.sha256(values.tobytes()).hexdigest()
    return {"global_sha256": sha(global_order), "reference_rank0_sha256": sha(orders[0]),
            "reference_rank1_sha256": sha(orders[1])}
