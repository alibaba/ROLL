import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from roll.distributed.scheduler.rollout_scheduler import (
    RolloutScheduler,
    GroupQueueManager,
)
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import GroupFilter


FULL_DATASET_ITER=4

class TestGroupFilter(GroupFilter):
    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        return episode_id % 3 == 0

@dataclass
class MockAgenticConfig:
    async_generation_ratio: int
    rollout_batch_size: int

class MockEnvManagerConfig:
    def __init__(
        self,
        world_size,
        env_groups,
        group_size,
        group_size_redundancy,
        rollout_batch_size,
        enable_filter,
        enable_redundancy,
    ):
        self.world_size = world_size
        self.env_groups = env_groups
        self.group_size = group_size
        self.group_size_redundancy = group_size_redundancy if enable_redundancy else 0
        self.final_group_size = group_size + self.group_size_redundancy
        if enable_filter:
            self.group_filter_cls = "tests.distributed.scheduler.test_rollout_scheduler.TestGroupFilter"
        else:
            self.group_filter_cls = "roll.pipeline.agentic.agentic_pipeline.GroupFilter"

        train_env_num = self.env_groups * self.group_size

        self.max_traj_per_env = (rollout_batch_size + train_env_num - 1) // train_env_num

        self.max_env_num_per_worker = self.env_groups * self.final_group_size
        self.env_num = self.world_size * self.max_env_num_per_worker
        self.env_configs = {0: {i: {"group_id": i} for i in range(env_groups)}}
        print(f"config: {self.env_num=} {self.world_size=} {self.max_env_num_per_worker=} {self.max_traj_per_env=}")

class MockEnvironmentWorker:
    def __init__(self, thread_id, gropu_id, output_queue):
        self.thread_id = thread_id
        self.group_id = gropu_id
        self.output_queue = output_queue
        self.current_step = None

    def run_rollout_loop(self, full_dataset):
        iter = 0
        while True:
            iter += 1
            episode_id = ray.get(self.output_queue.get_episode_id.remote(self.group_id))
            if episode_id is None:
                print("Env worker exit on episode_id is None")
                break
            elif full_dataset and episode_id == FULL_DATASET_ITER:
                print("Env worker exit on traverse all dataset")
                break
            else:
                start_step = self.current_step
            assert start_step is not None
            DataProto(meta_info={"global_step": start_step})
            ray.get(self.output_queue.put.remote(self.group_id, episode_id, start_step, (start_step, episode_id)))
        ray.get(self.output_queue.put.remote(self.group_id, episode_id, start_step, None))

class MockEnvManager(Worker):
    def __init__(self, env_manager_config, env_output_queue, full_dataset):
        assert env_manager_config.world_size == 1
        self.output_queue = env_output_queue
        self.full_dataset = full_dataset
        self.workers = [
            MockEnvironmentWorker(thread_id=i, gropu_id=i // env_manager_config.final_group_size, output_queue=env_output_queue)
            for i in range(env_manager_config.env_num)
        ]

    def stop(self, blocking=False):
        async def _stop():
            for worker in self.workers:
                pass
        return [_stop()]

    def update_step(self, step, blocking=False):
        async def _update_step():
            for worker in self.workers:
                worker.current_step = step
        return [_update_step()]

    def run_rollout_loop(self, seed, blocking=False):
        async def _run_rollout_loop():
            loop = asyncio.get_event_loop()
            pool = ThreadPoolExecutor(max_workers=len(self.workers))
            await asyncio.gather(*[loop.run_in_executor(pool, worker.run_rollout_loop, self.full_dataset) for worker in self.workers])
            pool.shutdown()
        return [_run_rollout_loop()]

@ray.remote
class MockRequestScheduler:
    def __init__(self, *args, **kwargs):
        pass

    async def suspend(self):
        pass

    async def resume(self):
        pass

    async def abort_request(self):
        pass

@ray.remote
class MockRolloutScheduler(RolloutScheduler):
    def __init__(self, config, env_manager_config, mode):
        self.config = config
        self.env_manager_config = env_manager_config
        self.mode = mode

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        self.env_output_queue = GroupQueueManager.options(
            max_concurrency = env_num + 1 # reserve extra one for get_batch
        ).remote(
            self.config,
            self.env_manager_config,
            mode
        )

        self.generate_scheduler = MockRequestScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                max_concurrency = env_num + 1 # reserve extra one for suspend/resume
            ).remote()

        self.es_manager = MockEnvManager(
            env_manager_config=self.env_manager_config,
            env_output_queue=self.env_output_queue,
            full_dataset=config.rollout_batch_size<=0,
        )

        self.rollout_task = None

    # FIXME use RolloutScheduler.get_batch
    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]

        # start env manager
        if self.rollout_task is None:
            seed = random.randint(0, 1000000) if self.mode == "train" else self.config.seed
            self.rollout_task = asyncio.create_task(self._run_rollout_loop(seed))

        await asyncio.gather(*self.es_manager.update_step(global_step, blocking=False))
        await self.env_output_queue.advance_step.remote(global_step)
        await self.generate_scheduler.resume.remote()

        get_task = asyncio.create_task(self._get_batch(batch_size, global_step))
        await asyncio.wait({get_task, self.rollout_task}, return_when=asyncio.FIRST_COMPLETED)
        if self.rollout_task.done() and self.rollout_task.exception() is not None:
            await self.rollout_task
            assert False
        data_batch = await get_task
        if batch_size <= 0:
            await self.rollout_task
            self.rollout_task = None
            await self.env_output_queue.clear.remote()
        return data_batch

async def async_test_GroupQueueManager(rollout_batch_size, async_generation_ratio, enable_filter=True, enable_redundancy=True):
    print(f">>>>>>>>>>>>>>>>>>>>>>>> TEST rollout_batch_size {rollout_batch_size} async_generation_ratio {async_generation_ratio}")
    config = MockAgenticConfig(rollout_batch_size=rollout_batch_size, async_generation_ratio=async_generation_ratio)

    env_manager_config = MockEnvManagerConfig(
        world_size=1,
        env_groups=2,
        group_size=8, # grpo
        group_size_redundancy=4,
        rollout_batch_size=rollout_batch_size,
        enable_filter=enable_filter,
        enable_redundancy=enable_redundancy,
    )

    scheduler = MockRolloutScheduler.remote(config, env_manager_config, "train")

    for i in range(10):
        current_step = i
        data = DataProto(meta_info={"global_step": current_step})
        await scheduler.suspend.remote()
        batch = await scheduler.get_batch.remote(data=data, batch_size=rollout_batch_size)

        print(f"batch on step({current_step}): {[rollout[0] for rollout in batch]}")
        expected = FULL_DATASET_ITER * env_manager_config.env_groups * env_manager_config.group_size if rollout_batch_size <= 0 else rollout_batch_size
        assert len(batch) == expected, f"{len(batch)=} expected={expected}"
        assert all(rollout[0] == batch[0][0] for rollout in batch), "Not all start_step are equal"
        assert (
            all(max(0, current_step - async_generation_ratio) == rollout[0] for rollout in batch)
        ), f"current_step({current_step}) - rollout_step({batch[0][0]}) exceed async_generation_ratio"

        await asyncio.sleep(1)
    await scheduler.shutdown.remote()

def test_GroupQueueManager():
    loop = asyncio.get_event_loop()

    # default_setting:
    #   env_num=16

    # batch_size = -1
    loop.run_until_complete(async_test_GroupQueueManager(-1, 0, enable_filter=False, enable_redundancy=False))

    # sync training
    loop.run_until_complete(async_test_GroupQueueManager(16, 0))
    loop.run_until_complete(async_test_GroupQueueManager(8, 0))
    loop.run_until_complete(async_test_GroupQueueManager(24, 0))
    loop.run_until_complete(async_test_GroupQueueManager(32, 0))
    loop.run_until_complete(async_test_GroupQueueManager(64, 0))

    # async training: 2
    loop.run_until_complete(async_test_GroupQueueManager(16, 2))
    loop.run_until_complete(async_test_GroupQueueManager(8, 2))
    # do not test batch_size 12, because 12 % group_size != 0
    loop.run_until_complete(async_test_GroupQueueManager(24, 2))
    loop.run_until_complete(async_test_GroupQueueManager(32, 2))
    loop.run_until_complete(async_test_GroupQueueManager(64, 2))

    # async training: 7
    loop.run_until_complete(async_test_GroupQueueManager(16, 7))
    loop.run_until_complete(async_test_GroupQueueManager(8, 7))

    # async training: 1
    loop.run_until_complete(async_test_GroupQueueManager(16, 1))
    loop.run_until_complete(async_test_GroupQueueManager(8, 1))
    loop.run_until_complete(async_test_GroupQueueManager(24, 1))
    loop.run_until_complete(async_test_GroupQueueManager(32, 1))
    loop.run_until_complete(async_test_GroupQueueManager(64, 1))

if __name__ == "__main__":
    test_GroupQueueManager()
