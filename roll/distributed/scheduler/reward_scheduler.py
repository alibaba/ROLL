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
    Reward service is different from generation, request interface:
        Reward scheduler needs to solve the reward calculation problem for samples from different domains, no need to implement request-level interface;
        And reward calculation is different from vllm, vllm can continue batch, so it can dynamically add requests, reward cannot,
            directly use rpc to call reward_cluster.compute_rewards (using rpc method, can increase the number of rewards, increase concurrent processing capacity)

    Problems that reward scheduler needs to solve:
        Route rewards by domain
        dp dispatch load balancing/insufficient dp_size limitations
    """

    def __init__(self):
        self.reward_clusters: Optional[Dict[str, Cluster]] = None
        self.pipeline_config = None
        self.progress_bar: Optional[tqdm] = None

    def compute_rewards(self, data: DataProto, reward_clusters: Dict[str, Any], pipeline_config) -> DataProto:
        """
        Return rewards in order
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
            # All rewards require consistent output schema
            # Reward worker compute_rewards interface returns results in order
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
