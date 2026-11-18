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
