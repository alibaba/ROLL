# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import random
from typing import Dict

import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict


class BatchStratifiedSampler(Sampler):
    """
    Batch Stratified Sampling:
        Ensures that each batch contains samples from different domains in a specified ratio.
        Example: For domain ratios (a=0.5, b=0.3, c=0.2) and batch_size=10:
            - 5 samples from domain 'a'
            - 3 samples from domain 'b'
            - 2 samples from domain 'c'
    """

    def __init__(self, dataset, domain_ratios: Dict, batch_size, drop_last=True):
        super().__init__()
        assert len(dataset) > 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.domain_ratios = copy.deepcopy(domain_ratios)
        sum_values = sum(domain_ratios.values())

        # Group sample indices by domain
        self.domain_indices = defaultdict(list)
        for idx in range(len(dataset)):
            self.domain_indices[dataset[idx]["domain"]].append(idx)
        for key in list(self.domain_ratios.keys()):
            if len(self.domain_indices[key]) == 0:
                del self.domain_indices[key], self.domain_ratios[key]
                print(f"{key} is empty, delete in sampling.")

        domain_indices_count = {key: len(value) for key, value in self.domain_indices.items()}
        print(f"domain_indices count: {domain_indices_count}")

        self.domain_ratios = {key: value / sum_values for key, value in domain_ratios.items()}
        # Calculate number of samples per domain in each batch
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

        # Expand indices to meet max batch capacity
        extended_indices = {}
        for domain, indices in shuffled.items():
            num = self.domain_batch_num[domain]
            total_required = self.total_batches * num
            repeat_times = (total_required + len(indices) - 1) // len(indices)
            extended = np.tile(indices, repeat_times)[:total_required]
            extended_indices[domain] = extended

        # Generate batches
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
