import asyncio
import json
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray._private import profiling
from tqdm import tqdm

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.functionals import append_to_dict, GenerateRequestType
from roll.utils.logging import get_logger

logger = get_logger()

class GroupQueue:
    def __init__(self, progress_bar: tqdm, group_size, max_group_num):
        self.group_size = group_size
        self.max_group_num = max_group_num
        self.clear(progress_bar)

    def clear(self, progress_bar):
        # assume no blocking put when clear called
        self.progress_bar = progress_bar
        self.groups: Dict[str, List[DataProto]] = {}
        self.inprogress = asyncio.Event()
        self.completed = asyncio.Semaphore(value=0)

    async def put(self, episode_id, start_step, rollout):
        if episode_id not in self.groups:
            while episode_id not in self.groups and len(self.groups) >= self.max_group_num:
                if self.inprogress.is_set():
                    self.inprogress.clear()
                await self.inprogress.wait()
            if episode_id not in self.groups:
                self.groups[episode_id] = []
        self.groups[episode_id].append(rollout)
        if len(self.groups[episode_id]) == self.group_size:
            self.completed.release()
            self.progress_bar.update(self.group_size)

    async def get(self):
        await self.completed.acquire()
        target = None
        for (episode_id, rollouts) in self.groups.items():
            if len(rollouts) >= self.group_size:
                target = min(episode_id, target) if target is not None else episode_id 
        assert target is not None
        ret = self.groups.pop(target)
        self.inprogress.set()
        return ret

@ray.remote
class EnvGroupQueue:
    def __init__(self, config, env_manager_config: EnvManagerConfig, mode):
        self.mode = mode
        self.env_manager_config = env_manager_config
        self.group_size = self.env_manager_config.group_size
        self.progress_bar = tqdm(desc=f"{self.mode} rollout progress(trajectory)", mininterval=self.env_manager_config.max_traj_per_env)
        self.wait_task = None
        self.exception = None
        self.pending_gets = set()

        if config.async_generation_ratio > 0 and self.mode == "train":
            # Async training use GroupQueue to implement rate limit.
            # There are at most 1 * async_generation_ratio in-progress steps.
            max_group_num = env_manager_config.max_traj_per_env * config.async_generation_ratio
        else:
            # Sync training do not need rate limit, and finished rollouts will never exceed limit
            max_group_num = env_manager_config.max_traj_per_env
        self.group_queue: Dict[int, GroupQueue] = {}
        for rank, rank_env_configs in env_manager_config.env_configs.items():
            for env_id, env_config in rank_env_configs.items():
                group_id = env_config["group_id"]
                if group_id not in self.group_queue:
                    self.group_queue[group_id] = GroupQueue(self.progress_bar, env_manager_config.group_size, max_group_num)

        # for debug
        self.total = 0
        self.waiting = 0

    def clear(self, batch_size):
        self.progress_bar = tqdm(total=batch_size, desc=f"{self.mode} rollout progress(trajectory)", mininterval=self.env_manager_config.max_traj_per_env)
        assert self.wait_task is None
        assert self.exception is None
        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        for group_queue in self.group_queue.values():
            group_queue.clear(self.progress_bar)

    def put_exception(self, exception):
        self.exception = exception
        if self.wait_task is not None:
            self.wait_task.cancel()

    def _check_exception(self):
        if self.exception is not None:
            raise self.exception

    async def put(self, group_id, episode_id, start_step, rollout: DataProto):
        self.waiting += 1
        await self.group_queue[group_id].put(episode_id, start_step, rollout)
        self.waiting -= 1
        self.total += 1

    async def get_batch(self, batch_size) -> List[DataProto]:
        """
        return completed rollouts group by group_id with least start_step
        """
        self._check_exception()
        # TODO: No need to get from every group queue, instead we can reuse 
        # a group queue as long as there are enough rollouts to avoid tail-latency?
        # But this will cause im-balance in episode_id.
        ret: List[DataProto] = []
        while len(ret) < batch_size:
            async def wait_a_episode():
                # Only wait for new episode when there are no pending GroupQueue.get,
                # this way we can avoid starvation of some env.
                if not self.pending_gets:
                    pending = set([asyncio.create_task(self.group_queue[group_id].get()) for group_id in self.group_queue])
                else:
                    pending = self.pending_gets
                    self.pending_gets = set()

                while pending and len(ret) < batch_size:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    while done and len(ret) < batch_size:
                        d = done.pop()
                        group_rollout = await d
                        assert len(group_rollout) == self.group_size
                        self.total -= len(group_rollout)
                        ret.extend(group_rollout)
                    assert (done and len(ret) >= batch_size) or (not done and len(ret) <= batch_size)
                    if done:
                        self.pending_gets.update(done)
                self.pending_gets.update(pending)

            assert self.wait_task is None
            self.wait_task = asyncio.create_task(wait_a_episode())
            try:
                await self.wait_task
            except asyncio.CancelledError:
                self._check_exception()
            self.wait_task = None
        assert len(ret) == batch_size
        return ret

@ray.remote
class RolloutScheduler:
    """
    Usage:
        sync:
            rollout_scheduler = RolloutScheduler()
            while True:
                model_update()
                ray.get(rollout_scheduler.get_batch.remote())
                rollout()
            ray.get(rollout_scheduler.stop.remote())

        async:
            rollout_scheduler = RolloutScheduler()
            while True:
                ray.get(rollout_scheduler.suspend.remote())
                model_update()
                ray.get(rollout_scheduler.resume.remote())
                ray.get(rollout_scheduler.get_batch.remote())
                rollout()
            ray.get(rollout_scheduler.stop.remote())

        sync and async (train and val) can exist simultaneously
    """
    def __init__(self, config, env_manager_config: EnvManagerConfig, resource_manager, infer_cluster, mode, collator=None):
        self.config = config
        self.env_manager_config = env_manager_config
        self.resource_manager = resource_manager
        self.infer_cluster = infer_cluster
        self.mode = mode

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        self.env_output_queue = EnvGroupQueue.options(
            max_concurrency = env_num + 1 # reserve extra one for get_batch
        ).remote(
            self.config,
            self.env_manager_config,
            mode
        )

        self.generate_scheduler = RequestScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                max_concurrency = env_num + 1 # reserve extra one for suspend/resume
            ).remote(infer_cluster=self.infer_cluster, pipeline_config=config)

        self.es_manager: Any = Cluster(
            name=self.env_manager_config.name,
            worker_cls=self.env_manager_config.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.env_manager_config,
        )
        self.es_manager.initialize(
            pipeline_config=self.config,
            generate_scheduler=self.generate_scheduler,
            output_queue=self.env_output_queue,
            collator=collator,
            mode=self.mode,
        )

        self.running = False
        self.rollout_refs = None # only used by async training

    async def start(self):
        if self.running:
            return
        assert self.rollout_refs is None
        self.running = True

        if self.config.async_generation_ratio > 0:
            data = DataProto()
            data.meta_info["global_step"] = 0
            data.meta_info["is_offload_states"] = False
            await asyncio.gather(
                *[
                    asyncio.wrap_future(ref.obj_ref.future())
                    for ref in self.infer_cluster.start_server(data, blocking=False)
                ]
            )
            self.alive_check_task = self.alive_check()

            if self.mode == "train":
                seed = self.config.seed
                self.rollout_refs: List[ray.ObjectRef] = self.es_manager.run_rollout_loop(None, seed, blocking=False)

    async def stop(self):
        if self.running == False:
            raise ValueError("rollout scheduler is not initialized")

        self.running = False

        if self.config.async_generation_ratio > 0:
            if self.mode == "train":
                assert self.rollout_refs is not None
                await asyncio.gather(
                    *[asyncio.wrap_future(ref.future()) for ref in self.es_manager.stop(blocking=False)],
                    *self.rollout_refs,
                )
            await asyncio.gather(
                self.alive_check_task,
                *[asyncio.wrap_future(ref.obj_ref.future()) for ref in self.infer_cluster.stop_server(data=DataProto(), blocking=False)],
            )

    async def suspend(self, global_step):
        if self.config.async_generation_ratio == 0 or self.mode != "train":
            return
        await self.generate_scheduler.suspend.remote()

    async def resume(self, global_step):
        if self.config.async_generation_ratio == 0 or self.mode != "train":
            return
        await asyncio.gather(
            *[
                asyncio.wrap_future(ref.future())
                for ref in self.es_manager.update_step(global_step, blocking=False)
            ]
        )
        await self.generate_scheduler.resume.remote()

    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]
        if not self.running:
            await self.start()

        if self.config.async_generation_ratio == 0 or self.mode != "train":
            if self.config.async_generation_ratio == 0:
                assert self.running
                await asyncio.gather(
                    *[
                        asyncio.wrap_future(ref.obj_ref.future())
                        for ref in self.infer_cluster.start_server(data, blocking=False)
                    ]
                )
                self.alive_check_task = self.alive_check()
            assert self.rollout_refs is None
            if self.mode == "train":
                seed = random.randint(0, 1000000)
            else:
                seed = self.config.seed
            rollout_refs: List[ray.ObjectRef] = self.es_manager.run_rollout_loop(global_step, seed, blocking=False)

        ref = self.env_output_queue.get_batch.remote(batch_size)
        data_batch: List[DataProto] = await asyncio.wrap_future(ref.future())
        metrics = {}
        [append_to_dict(metrics, meta_info.meta_info["metrics"]) for meta_info in data_batch]
        batch = DataProto.concat(data_batch)

        if self.config.async_generation_ratio == 0 or self.mode != "train":
            assert self.rollout_refs is None
            # TODO: abort running requests?
            await asyncio.gather(
                *[asyncio.wrap_future(ref.future()) for ref in self.es_manager.stop(blocking=False)],
                *rollout_refs
            )
            if self.config.async_generation_ratio == 0:
                assert self.running
                await self.stop() # only set self.running to False
                assert not self.running
                stop_server_tasks = [
                    asyncio.wrap_future(ref.obj_ref.future())
                    for ref in self.infer_cluster.stop_server(blocking=False)
                ]
                await asyncio.gather(
                    asyncio.wrap_future(self.env_output_queue.clear.remote(batch_size).future()),
                    self.alive_check_task,
                )
                gen_metrics = await asyncio.gather(*stop_server_tasks)
                gen_metrics = gen_metrics[0]
                metrics.update(gen_metrics.meta_info.pop("metrics", {}))

        batch.meta_info["metrics"] = metrics
        return batch

    # TODO: do not need alive_check if use async_generate
    async def alive_check(self):
        alive_check_interval = self.config.alive_check_interval
        while self.running:
            await asyncio.sleep(alive_check_interval)
            try:
                outputs: List[DataProto] = await asyncio.gather(
                    *[
                        asyncio.wrap_future(ref.future())
                        for ref in self.infer_cluster.add_request(
                                command=GenerateRequestType.ALIVE_CHECK, data=DataProto(), blocking=False)
                    ]
                )
            except Exception as e:
                if not self.running:
                    return
                self.env_output_queue.put_exception(e)
                return
            request_counts = {key: output.meta_info["request_counts"] for key, output in enumerate(outputs)}
            metrics = {"time": datetime.now().strftime("%Y%m%d-%H%M%S"), "metrics": request_counts}
            logger.debug(f"generate flow: {json.dumps(metrics)}")
