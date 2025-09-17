import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import ray

from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager

TEST_EXCEPTION = False

class AgenticConfig:
    pass

class EnvManagerConfig:
    pass

async def async_test_GroupQueueManager(rollout_batch_size, async_generation_ratio):
    print(f">>>>>>>>>>>>>>>>>>>>>>>> TEST rollout_batch_size {rollout_batch_size} async_generation_ratio {async_generation_ratio}")
    config = AgenticConfig()
    config.async_generation_ratio = async_generation_ratio

    env_manager_config = EnvManagerConfig()
    env_manager_config.world_size = 1
    env_manager_config.env_groups = 2
    env_manager_config.group_size = 8 # grpo
    train_env_num = env_manager_config.env_groups * env_manager_config.group_size
    env_manager_config.max_env_num_per_worker = train_env_num
    env_manager_config.env_configs = {0: {0: {"group_id": 0}, 1: {"group_id": 1}}}

    traj_per_env = (rollout_batch_size + train_env_num - 1) // train_env_num
    env_manager_config.max_traj_per_env = traj_per_env

    env_num = env_manager_config.world_size * env_manager_config.max_env_num_per_worker

    env_output_queue = GroupQueueManager.options(
        max_concurrency = env_num + 1
    ).remote(
        config,
        env_manager_config,
        "train"
    )

    current_step = 0
    stoped_threads = 0
    barrier = threading.Barrier(env_num + 1)

    def run_rollout_loop(thread_id, group_id, output_queue):
        nonlocal stoped_threads
        if TEST_EXCEPTION:
            raise Exception("test exception")

        episode_id = 0
        for i in range(10):
            if async_generation_ratio == 0:
                barrier.wait()
            rollout = current_step
            for j in range(env_manager_config.max_traj_per_env):
                ray.get(output_queue.put.remote(group_id, episode_id, 0, rollout))
                episode_id += 1
            if async_generation_ratio == 0:
                barrier.wait()
        stoped_threads += 1

    async def rollout():
        nonlocal current_step
        try:
            for i in range(10):
                current_step = i
                if async_generation_ratio == 0:
                    barrier.wait()
                batch = await env_output_queue.get_batch.remote(rollout_batch_size)
                print(f"batch on step({current_step}): {batch}")
                if rollout_batch_size >= env_num and rollout_batch_size % env_num == 0: 
                    assert all((current_step - rollout) <= async_generation_ratio for rollout in batch), f"current_step - rollout_step exceed async_generation_ratio"
                if async_generation_ratio == 0:
                    env_output_queue.prepare_clear.remote()
                    barrier.wait()
                    env_output_queue.clear.remote(rollout_batch_size)
                await asyncio.sleep(1)
            env_output_queue.prepare_clear.remote()
            # unblock all run_rollout_loop threads
            # cannot call env_output_queue.clear here (otherwise must wait all threads are finished)
        except Exception as e:
            sys.exit(f"ERROR rollout get exception: {e}")

    rollout_task = asyncio.create_task(rollout())

    with ThreadPoolExecutor(max_workers=16) as pool:
        loop = asyncio.get_event_loop()
        try:
            assert env_manager_config.world_size == 1
            if TEST_EXCEPTION:
                assert 2 < env_num
                await asyncio.gather(
                    *[loop.run_in_executor(pool, run_rollout_loop, i, i // env_manager_config.group_size, env_output_queue) for i in range(2)]
                )
            else:
                await asyncio.gather(
                    *[loop.run_in_executor(pool, run_rollout_loop, i, i // env_manager_config.group_size, env_output_queue) for i in range(env_num)]
                )
        except Exception as e:
            ref = env_output_queue.put_exception.remote(e)
            await asyncio.wrap_future(ref.future())

    await rollout_task

def test_GroupQueueManager():
    loop = asyncio.get_event_loop()

    # env_num is 16

    # test BoundedGroupQueue
    loop.run_until_complete(async_test_GroupQueueManager(16, 2))
    loop.run_until_complete(async_test_GroupQueueManager(8, 2))
    # do not test batch_size 12, because 12 % group_size != 0
    loop.run_until_complete(async_test_GroupQueueManager(24, 2))
    loop.run_until_complete(async_test_GroupQueueManager(32, 2))

    loop.run_until_complete(async_test_GroupQueueManager(16, 7))
    loop.run_until_complete(async_test_GroupQueueManager(8, 7))

    # test PipeGroupQueu
    loop.run_until_complete(async_test_GroupQueueManager(16, 1))
    loop.run_until_complete(async_test_GroupQueueManager(8, 1))
    loop.run_until_complete(async_test_GroupQueueManager(24, 1))
    loop.run_until_complete(async_test_GroupQueueManager(32, 1))

    # test sync training
    loop.run_until_complete(async_test_GroupQueueManager(16, 0))
    loop.run_until_complete(async_test_GroupQueueManager(8, 0))
    loop.run_until_complete(async_test_GroupQueueManager(24, 0))
    loop.run_until_complete(async_test_GroupQueueManager(32, 0))

if __name__ == "__main__":
    test_GroupQueueManager()
