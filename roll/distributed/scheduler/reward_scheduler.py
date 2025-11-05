from collections import defaultdict
from typing import Dict, Optional, List, Any

import ray
from tqdm import tqdm
import torch

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()


class RewardScheduler:
    """
    reward 服务化和generate不同, request接口：
        reward scheduler需要解决的是不同域的sample的reward计算问题, 不需要实现request粒度的接口；
        并且reward计算和vllm不同，vllm可以continue batch，所以可以动态add request, reward不行，
            直接rpc调用reward_cluster.compute_rewards即可(使用rpc方式调用，可以增加reward的数量，增大并发处理能力)

    reward scheduler需要解决的问题:
        按domain路由reward
        dp dispatch 均分/不足dp_size 的限制
    """

    def __init__(self):
        self.reward_clusters: Optional[Dict[str, Cluster]] = None
        self.pipeline_config = None
        self.progress_bar: Optional[tqdm] = None

    def compute_rewards(self, data: DataProto, reward_clusters: Dict[str, Any], pipeline_config) -> DataProto:
        """
        保序返回rewards
        """
        self.pipeline_config = pipeline_config
        self.reward_clusters = reward_clusters
        data.batch["prompt_id"] = torch.arange(data.batch.batch_size[0], device=data.batch.device)

        # 按domain group by data
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
            # 各reward的输出schema要求一致
            # reward worker compute_rewards 接口返回结果保序
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
