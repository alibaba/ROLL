import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import ray

from roll.agentic.rollout.rollout_scheduler import EnvGroupQueue, GroupQueue

TEST_EXCEPTION = False

class AgenticConfig:
    pass

class EnvManagerConfig:
    pass

async def async_test_EnvGroupQueue_grpo():
    rollout_batch_size = 16

    config = AgenticConfig()
    config.async_generation_ratio = 2

    env_manager_config = EnvManagerConfig()
    env_manager_config.world_size = 1
    env_manager_config.env_groups = 2
    env_manager_config.group_size = 8
    env_manager_config.max_env_num_per_worker = 16
    env_manager_config.env_configs = {0: {0: {"group_id": 0}, 1: {"group_id": 1}}}

    train_env_num = env_manager_config.env_groups * env_manager_config.group_size
    traj_per_env = (rollout_batch_size + train_env_num - 1) // train_env_num
    env_manager_config.max_traj_per_env = traj_per_env

    env_num = env_manager_config.world_size * env_manager_config.max_env_num_per_worker

    env_output_queue = EnvGroupQueue.options(
        max_concurrency = env_num + 1
    ).remote(
        config,
        env_manager_config,
        "train"
    )

    quit = False

    def run_rollout_loop(thread_id, group_id, output_queue):
        if TEST_EXCEPTION:
            raise Exception("test exception")

        episode_id = 0
        while not quit:
            rollout = None
            ray.get(output_queue.put.remote(group_id, episode_id, 0, rollout))
            episode_id += 1
        print(f">>>>>>>>>>>>>>>>>>>>>> rollout loop {thread_id} finish")

    async def get_batch():
        nonlocal quit
        for i in range(10):
            print(f">>>>>>>>>>>>>>>>>>>>>> iter {i}")
            await env_output_queue.get_batch.remote(rollout_batch_size)
        quit = True
        for i in range(10):
            await env_output_queue.clear.remote(16)
            await asyncio.sleep(1)

    get_batch_task = asyncio.create_task(get_batch())

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

    await get_batch_task

def test_EnvGroupQueue_grpo():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_test_EnvGroupQueue_grpo())

if __name__ == "__main__":
    test_EnvGroupQueue_grpo()
