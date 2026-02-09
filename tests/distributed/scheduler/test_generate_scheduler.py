import asyncio
import ray
import math
import random
from typing import List, Optional
from dataclasses import dataclass
import torch
import numpy as np
import pytest

from roll.distributed.scheduler.generate_scheduler import (
    DynamicSamplingScheduler,
    RolloutContext,
    LoadBalancer,
    ExperienceItem,
)
import roll.distributed.scheduler.user_defined_rollout_loop as udrl
from roll.distributed.scheduler.user_defined_rollout_loop import UserDefinedRolloutLoop as UserDefinedRolloutLoopBase
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.executor.worker import RankInfo
from roll.configs import ModelArguments
from roll.configs.worker_config import WorkerConfig
from roll.pipeline.rlvr.rlvr_config import RewardConfig, RewardFilterConfig
from roll.utils.logging import get_logger


logger = get_logger()


async def test_load_balancer():
    load_balancer = LoadBalancer(mp_rank_zero={0:0, 1:0, 2:0, 3:0}, max_running_requests=2)

    leases = []
    for i in range(8):
        lease = await load_balancer.acquire(1)
        assert lease._dp_rank == i % 4
        leases.append(lease)
    assert load_balancer.full()
    for i in range(8):
        leases[i].clear()
    assert load_balancer.empty()

    async def process_new_prompt():
        lease = await load_balancer.acquire(2)
        await asyncio.sleep(2)
        for i in range(2):
            assert lease.lease == 2 - i
            async with lease.lock(1) as dp_rank:
                assert dp_rank == lease._dp_rank
            assert lease.lease == 1 - i
        return lease._dp_rank

    tasks = [asyncio.create_task(process_new_prompt()) for _ in range(4)]
    await asyncio.sleep(1)
    assert load_balancer.full()
    await asyncio.sleep(2)
    assert load_balancer.empty()
    await load_balancer.wait_complete()
    assert load_balancer.empty()
    ret = await asyncio.gather(*tasks)
    assert len(ret) == 4 and sum(ret) == 6
    assert set(ret) == set([0, 1, 2, 3])

    tasks = [asyncio.create_task(process_new_prompt()) for _ in range(8)]
    await asyncio.sleep(1)
    assert load_balancer.full()
    await asyncio.sleep(2)
    assert load_balancer.full()
    await load_balancer.wait_complete()
    assert load_balancer.empty()
    ret = await asyncio.gather(*tasks)
    assert len(ret) == 8 and sum(ret) == 12
    assert set(ret) == set([0, 1, 2, 3])

    async def suspended():
        while load_balancer._suspend:
            load_balancer.suspend_event.clear()
            await load_balancer.suspend_event.wait()

    load_balancer.suspend() 
    tasks = [asyncio.create_task(process_new_prompt()) for _ in range(8)]
    await asyncio.sleep(1)
    assert load_balancer.empty()
    wait_task = asyncio.create_task(suspended())
    await asyncio.sleep(1)
    assert not wait_task.done()
    load_balancer.resume()
    await wait_task
    await asyncio.sleep(1)
    assert load_balancer.full()
    await load_balancer.wait_complete()
    assert load_balancer.empty()
    ret = await asyncio.gather(*tasks)
    assert len(ret) == 8 and sum(ret) == 12
    assert set(ret) == set([0, 1, 2, 3])


@ray.remote
class MockWorker:
    async def generate_request(self, data: DataProto):
        if "turn" not in data.meta_info:
            data.meta_info["turn"] = 1
        else:
            data.meta_info["turn"] += 1

        if data.meta_info["turn"] < 3:
            data.meta_info["finihsh_reasons"] = ["abort"]
        else:
            data.meta_info["finihsh_reasons"] = ["stop"]

        return data

    async def compute_rewards(self, data: DataProto):
        return data

    async def abort_requests(self, ids):
        return

class MockCluster:
    def __init__(self, workers: List[MockWorker]):
        self.workers = workers
        self.worker_rank_info = [RankInfo() for _ in range(4)]
        self.worker_config = WorkerConfig(model_args=ModelArguments(model_type="diffusion_module"))

    def get_rank_info(self, rank):
        return self.worker_rank_info[rank]
    
class MockCollectFn:
    def __init__(self, tokenizer):
        pass

    def __call__(self, data):
        assert isinstance(data, list)
        assert len(data) == 1
        assert isinstance(data[0], dict)
        domain = [data[0]["domain"]]
        data[0]["domain"] = np.empty(len(domain), dtype=object)
        data[0]["domain"][:] = domain
        return data[0]

@dataclass
class MockPipelineConfig:
    is_val: bool = False

    async_generation_ratio: float = 0
    max_running_requests: int = 128
    is_num_return_sequences_expand: bool = True # this unit test only support is_num_return_sequences_expand
    is_use_additional_prompts: bool = False
    max_additional_running_prompts: int = 0
    user_defined_rollout_loop_cls: str = "roll.distributed.scheduler.user_defined_rollout_loop.UserDefinedRolloutLoop"

    seed: int = 0
    sequence_length: int = 0
    val_sequence_length: int = 0
    prompt_length: int = 0

    rewards = {"default": RewardConfig(query_filter_config=RewardFilterConfig(type="no_filter"))}

def postprocess_paused_data(pre_data, data: DataProto, sequence_length, prompt_length) -> DataProto:
    return data
udrl.postprocess_paused_data = postprocess_paused_data

def postprocess_output_data(request, data: DataProto, sequence_length) -> DataProto:
    return data
udrl.postprocess_output_data = postprocess_output_data

class UserDefinedRolloutLoopWithFilter(UserDefinedRolloutLoopBase):
    def __init__(self):
        super().__init__()
        self.used_prompt = 0

    async def process_new_prompt(self, context: RolloutContext) -> Optional[DataProto|List[DataProto]]:
        ret = await super().process_new_prompt(context)
        self.used_prompt += 1
        if self.used_prompt < 16:
            return None
        else:
            return ret

class UserDefinedRolloutLoopWithDynamicSamplen(UserDefinedRolloutLoopBase):
    async def process_new_prompt(self, context: RolloutContext) -> Optional[DataProto|List[DataProto]]:
        ret = await super().process_new_prompt(context)
        assert isinstance(ret, list)
        # dynamic num_return_sequences
        if random.choice([True, False]):
            return ret * 2
        else:
            return ret[0]

class MockDynamicSamplingScheduler(DynamicSamplingScheduler):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.mock_pipeline_config = pipeline_config

    async def set_scheduler(self):
        actor_cluster = MockCluster([MockWorker.remote() for _ in range(4)])
        reward_clusters = {"default": MockCluster([MockWorker.remote() for _ in range(4)])}
        await super().set_scheduler(
            actor_cluster,
            reward_clusters,
            dataset=range(0,1024),
            collect_fn_cls=MockCollectFn,
            collect_fn_kwargs={},
            is_val=self.mock_pipeline_config.is_val,
        )

    def get_next_dataset_item(self):
        return {
            "prompt": torch.ones((1, 1)),
            "response_level_rewards": torch.ones((1, 1)),
            "domain": "default",
        }

    def collect_items_as_batch(self, finished_items: List[ExperienceItem]):
        batch = DataProto(meta_info={
            "finished_items": finished_items,
            "metrics": {},
        })
        return batch

async def test_val():
    logger.info("TEST test_val")
    async_generation_ratio = 2
    pipeline_config = MockPipelineConfig(
        is_val=True,
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=False,
        max_additional_running_prompts=0,
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        await scheduler.pause_sampling()
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(data=data, global_step=i, batch_size=4)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step == max(0, i)
            assert item.prompt_id in list(range(i * 4, (i + 1) * 4)), f"{[item.prompt_id for item in ret]}"
        logger.info(f"test_val step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
    await scheduler.shutdown()

async def test_sync():
    logger.info("TEST test_sync")
    async_generation_ratio = 0
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=False,
        max_additional_running_prompts=0,
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        await scheduler.pause_sampling()
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(data=data, global_step=i, batch_size=4)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step == max(0, i)
            assert item.prompt_id in list(range(i * 4, (i + 1) * 4)), f"{[item.prompt_id for item in ret]}"
        logger.info(f"test_sync step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
    await scheduler.shutdown()

async def test_sync_pause():
    logger.info("TEST test_sync_pause")
    async_generation_ratio = 0
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=False,
        max_additional_running_prompts=0,
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(data=data, global_step=i, batch_size=4)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step == max(0, i)
            assert item.prompt_id in list(range(i * 4, (i + 1) * 4)), f"{[item.prompt_id for item in ret]}"
        logger.info(f"test_sync_pause step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
    await scheduler.shutdown()

async def test_sync_filter():
    logger.info("TEST test_sync_filter")
    async_generation_ratio = 0
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=True,
        max_additional_running_prompts=2,
        user_defined_rollout_loop_cls="tests.distributed.scheduler.test_generate_scheduler.UserDefinedRolloutLoopWithFilter",
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        await scheduler.pause_sampling()
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(global_step=i, batch_size=4, data=data)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step == max(0, i)
        logger.info(f"test_sync_filter step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
    await scheduler.shutdown()

async def test_sync_additional_prompts():
    logger.info("TEST test_sync_additional_prompts")
    async_generation_ratio = 0
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=True,
        max_additional_running_prompts=2,
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        await scheduler.pause_sampling()
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(data=data, global_step=i, batch_size=4)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step == max(0, i)
        logger.info(f"test_sync_additional_prompts step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
    await scheduler.shutdown()

async def test_sync_dynamic_num_return_sequences():
    logger.info("TEST test_sync_dynamic_num_return_sequences")
    async_generation_ratio = 0
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=True,
        max_additional_running_prompts=2,
        user_defined_rollout_loop_cls="tests.distributed.scheduler.test_generate_scheduler.UserDefinedRolloutLoopWithDynamicSamplen",
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        await scheduler.pause_sampling()
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(global_step=i, batch_size=4, data=data)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step == max(0, i)
        logger.info(f"test_sync_dynamic_num_return_sequences step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
    await scheduler.shutdown()

async def test_sync_dynamic_num_return_sequences_exception():
    logger.info("TEST test_sync_dynamic_num_return_sequences_exception")
    async_generation_ratio = 0
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=False,
        max_additional_running_prompts=0,
        user_defined_rollout_loop_cls="tests.distributed.scheduler.test_generate_scheduler.UserDefinedRolloutLoopWithDynamicSamplen",
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    with pytest.raises(Exception):
        for i in range(10):
            logger.info(f"pipeline step {i}")
            await scheduler.pause_sampling()
            data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
            ret = await scheduler.get_batch(global_step=i, batch_size=4, data=data)
            # logger.info(f"step {i}: {ret}")
            ret = ret.meta_info["finished_items"]
            assert len(ret) == 8, f"{len(ret)=}"
            for item in ret:
                assert item.sampling_start_step == max(0, i)
            logger.info(f"test_sync_dynamic_num_return_sequences_exception step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
        await scheduler.shutdown()

async def test_1_off():
    logger.info("TEST test_1_off")
    async_generation_ratio = 1
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=False,
        max_additional_running_prompts=0,
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        await scheduler.pause_sampling()
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(data=data, global_step=i, batch_size=4)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step >= max(0, i - math.ceil(async_generation_ratio))
            assert item.sampling_start_step <= i
            assert item.prompt_id >= max(0, i - async_generation_ratio) * 4
            assert item.prompt_id < (i + 1 + async_generation_ratio) * 4
        logger.info(f"test_1_off step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
        await asyncio.sleep(2)
    await scheduler.shutdown()

async def test_3_off():
    logger.info("TEST test_3_off")
    async_generation_ratio = 3.0
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=False,
        max_additional_running_prompts=0,
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        await scheduler.pause_sampling()
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(data=data, global_step=i, batch_size=4)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step >= max(0, i - math.ceil(async_generation_ratio))
            assert item.sampling_start_step <= i
            assert item.prompt_id >= max(0, i - async_generation_ratio) * 4
            assert item.prompt_id < (i + 1 + async_generation_ratio) * 4
        logger.info(f"test_3_off step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
        await asyncio.sleep(2)
    await scheduler.shutdown()

async def test_2_5_off():
    logger.info("TEST test_2_5_off")
    async_generation_ratio = 2.5
    pipeline_config = MockPipelineConfig(
        async_generation_ratio=async_generation_ratio,
        max_running_requests=2,
        is_use_additional_prompts=False,
        max_additional_running_prompts=0,
    )
    scheduler = MockDynamicSamplingScheduler(pipeline_config)
    await scheduler.set_scheduler()
    for i in range(10):
        logger.info(f"pipeline step {i}")
        await scheduler.pause_sampling()
        data = DataProto(meta_info={"generation_config": {"num_return_sequences": 2}})
        ret = await scheduler.get_batch(data=data, global_step=i, batch_size=4)
        # logger.info(f"step {i}: {ret}")
        ret = ret.meta_info["finished_items"]
        assert len(ret) == 8, f"{len(ret)=}"
        for item in ret:
            assert item.sampling_start_step >= max(0, i - math.ceil(async_generation_ratio))
            assert item.sampling_start_step <= i
        logger.info(f"test_2_5_off step={i}, response step={[item.sampling_start_step for item in ret]}, prompt_id={[item.prompt_id for item in ret]}")
        await asyncio.sleep(2)
    await scheduler.shutdown()

async def test_dynamic_sampling_scheduler():
    await test_val()
    await test_sync()
    await test_sync_pause()
    await test_sync_filter()
    await test_sync_additional_prompts()
    await test_sync_dynamic_num_return_sequences()
    await test_sync_dynamic_num_return_sequences_exception()
    await test_1_off()
    await test_3_off()
    await test_2_5_off()


if __name__ == "__main__":
    ray.init()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_load_balancer())
    loop.run_until_complete(test_dynamic_sampling_scheduler())
