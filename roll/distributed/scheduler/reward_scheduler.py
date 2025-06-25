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
from collections import defaultdict
from typing import Dict, Optional, List, Any

import ray
from tqdm import tqdm
import torch

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()


@ray.remote
class RewardScheduler:
    """        
    Reward scheduler is different from generation scheduler. 
    1. Reward scheduler must handle reward computation for samples in different domains, no need to implement request-level interfaces (unlike generation scheduler).  
    2. VLLM can support continuous batching and dynamic request addition, while reward computation cannot. 
    Instead, reward computation is performed by direct RPC calls via `reward_cluster.compute_rewards` that can improve scalability and concurrency.  

    Key responsibilities of the reward schedule:
    - Domain routing: Route samples to appropriate reward clusters based on domain.  
    - DP dispatch load balancing and handle cases with insufficient DP size. 
    """

    def __init__(self):
        self.reward_clusters: Optional[Dict[str, Cluster]] = None
        self.pipeline_config = None
        self.progress_bar: Optional[tqdm] = None

    def compute_rewards(self, data: DataProto, reward_clusters: Dict[str, Any], pipeline_config) -> DataProto:
        """
        Compute rewards while maintaining the original order of input data.
        """
        self.pipeline_config = pipeline_config
        self.reward_clusters = reward_clusters
        data.batch["prompt_id"] = torch.arange(data.batch.batch_size[0], device=data.batch.device)

        # Group data by domain
        grouped_data: Dict[str, DataProto] = data.group_by("domain")

        domain_rewards_refs: Dict[str, List[ray.ObjectRef]] = defaultdict(list)
        for domain, reward_cluster in reward_clusters.items():
            if domain not in grouped_data.keys():
                continue
            domain_rewards_refs[domain].extend(
                reward_cluster.compute_rewards(data=grouped_data[domain], blocking=False)
            )

        rewards_list: List[DataProto] = []
        for domain, domain_rewards_ref in domain_rewards_refs.items():
            # Ensure output schema consistency across reward clusters
            # Reward worker's `compute_rewards` interface returns results in order
            if domain not in grouped_data.keys():
                continue
            domain_rewards: DataProto = DataProto.materialize_concat(data_refs=domain_rewards_ref)
            domain_rewards.batch["prompt_id"] = grouped_data[domain].batch["prompt_id"]
            rewards_list.append(domain_rewards)

        rewards = DataProto.concat(rewards_list)

        # reorder
        _, sorted_indices = torch.sort(rewards.batch["prompt_id"])
        rewards.reorder(indices=sorted_indices)
        rewards.pop("prompt_id")

        return rewards
