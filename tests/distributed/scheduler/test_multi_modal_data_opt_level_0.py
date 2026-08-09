"""Bug: get_batch_opt_level_0() drops multi_modal_data before it reaches generation.

DynamicSamplingScheduler.get_batch_opt_level_0() built gen_batch with
    request_data.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
which never carries non_tensor_batch["multi_modal_data"] into gen_batch. For VLM
requests, the generation backend (router.py / sglang_strategy.py, which read
multi_modal_data from batch.non_tensor_batch to build image payloads) then receives
a text-only prompt that still contains image placeholder tokens, and the model's
M-RoPE position code indexes an empty image_grid_thw, producing an IndexError.

Fix: forward multi_modal_data into gen_batch via pop()'s non_tensor_batch_keys, same
as PR #446 (ae4c065), which a later release-publish commit (7f9d4d3) accidentally
dropped again.
"""

import asyncio

import numpy as np
import torch

from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto


class _RecordingActorCluster:
    dp_size = 1

    def __init__(self):
        self.received_keys = None

    def generate(self, gen_batch):
        # Snapshot immediately: gen_batch may be aliased and mutated by a later union().
        self.received_keys = set(gen_batch.non_tensor_batch.keys())
        return gen_batch


class _NoOpRewardScheduler:
    async def compute_rewards(self, data, reward_clusters, pipeline_config):
        return DataProto(meta_info={"metrics": {}})


def _next_vlm_item():
    return {
        "prompt": torch.ones((1, 1)),
        "domain": np.array(["default"], dtype=object),
        "multi_modal_data": np.array([{"image": ["fake"]}], dtype=object),
    }


def _collect_fn(data_item_list):
    assert len(data_item_list) == 1
    return data_item_list[0]


def test_get_batch_opt_level_0_forwards_multi_modal_data():
    async def run():
        actor_cluster = _RecordingActorCluster()

        class _Scheduler:
            is_val = True
            get_next_dataset_item = staticmethod(_next_vlm_item)
            collect_fn = staticmethod(_collect_fn)

        scheduler = _Scheduler()
        scheduler.actor_cluster = actor_cluster
        scheduler.reward_scheduler = _NoOpRewardScheduler()
        scheduler.reward_clusters = {}
        scheduler.pipeline_config = object()

        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 1}})
        await DynamicSamplingScheduler.get_batch_opt_level_0(scheduler, data, batch_size=1)

        assert actor_cluster.received_keys is not None
        assert "multi_modal_data" in actor_cluster.received_keys, (
            "multi_modal_data must reach actor_cluster.generate() for VLM requests; "
            f"got non_tensor_batch keys {actor_cluster.received_keys}"
        )

    asyncio.run(run())
