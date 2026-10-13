import copy
import random
from typing import Dict

import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict


class BatchStratifiedSampler(Sampler):
    """
    Batch分层抽样
    batch的样本按指定的比例分布，如(a=0.5, b=0.3, c=0.2),
    batch_size=10，int(0.5 * batch_size)的样本属于a, int(0.3 * batch_size)的样本属于b，(batch_size-len(a)-len(b))的样本属于c
    """

    def __init__(self, dataset, domain_ratios: Dict, batch_size, drop_last=True):
        super().__init__()
        assert len(dataset) > 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.domain_ratios = copy.deepcopy(domain_ratios)

        # 按domain分组样本索引
        self.domain_indices = defaultdict(list)
        for idx in range(len(dataset)):
            self.domain_indices[dataset[idx]["domain"]].append(idx)
        for key in list(self.domain_ratios.keys()):
            if len(self.domain_indices[key]) == 0:
                del self.domain_indices[key], self.domain_ratios[key]
                print(f"{key} is empty, delete in sampling.")

        domain_indices_count = {key: len(value) for key, value in self.domain_indices.items()}
        print(f"domain_indices count: {domain_indices_count}")

        # Renormalize pruned ratios (do not reuse the constructor arg).
        sum_values = sum(self.domain_ratios.values())
        assert sum_values > 0, "No domains left after removing empty domain_ratios entries."
        self.domain_ratios = {key: value / sum_values for key, value in self.domain_ratios.items()}
        # 计算每个domain在每个batch中的样本数
        self.domain_batch_num = {}
        accumulated = 0
        domain_list = list(self.domain_ratios.keys())

        for i, domain in enumerate(domain_list):
            if i == len(domain_list) - 1:
                self.domain_batch_num[domain] = batch_size - accumulated
            else:
                self.domain_batch_num[domain] = int(self.domain_ratios[domain] * batch_size)
                accumulated += self.domain_batch_num[domain]

        assert (
            sum(self.domain_batch_num.values()) == batch_size
        ), "Sum of domain samples per batch must equal batch size."

        self.domain_batch_capacities = {}
        for domain, batch_num in self.domain_batch_num.items():
            if drop_last:
                self.domain_batch_capacities[domain] = len(self.domain_indices[domain]) // batch_num
            else:
                self.domain_batch_capacities[domain] = (len(self.domain_indices[domain]) + batch_num - 1) // batch_num
        self.total_batches = max(self.domain_batch_capacities.values())

    def __iter__(self):
        shuffled = {}
        for domain, indices in self.domain_indices.items():
            indices_np = np.array(indices)
            np.random.shuffle(indices_np)
            shuffled[domain] = indices_np

        # 扩展每个domain的索引以满足max_batch容量
        extended_indices = {}
        for domain, indices in shuffled.items():
            num = self.domain_batch_num[domain]
            total_required = self.total_batches * num
            repeat_times = (total_required + len(indices) - 1) // len(indices)
            extended = np.tile(indices, repeat_times)[:total_required]
            extended_indices[domain] = extended

        # 生成Batch
        batches = []
        for i in range(self.total_batches):
            batch = []
            for domain in self.domain_ratios:
                num = self.domain_batch_num[domain]
                start = i * num
                end = start + num
                batch.extend(extended_indices[domain][start:end].tolist())
            np.random.shuffle(batch)
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.total_batches
