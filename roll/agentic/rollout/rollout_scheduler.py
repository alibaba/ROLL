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
    def __init__(self, progress_bar: tqdm, group_size, max_group_num, max_traj_per_env):
        self.group_size = group_size
        self.max_group_num = max_group_num
        self.max_traj_per_env = max_traj_per_env
        self.processed_episodes = set()
        self.clear(progress_bar)

    def prepare_clear(self):
        raise NotImplementedError

    def clear(self, progress_bar):
        raise NotImplementedError

    # assume episode_id start from 0 and increase monotonically (not across GroupQueue.clear)
    async def put(self, episode_id, start_step, rollout):
        raise NotImplementedError

    async def get(self):
        raise NotImplementedError

class BoundedGroupQueue(GroupQueue):
    def prepare_clear(self):
        self.quit = True
        self.inprogress.set()

    def clear(self, progress_bar):
        # assume no blocking put when clear called
        self.quit = False
        self.progress_bar = progress_bar
        self.groups: Dict[str, List[DataProto]] = {}
        self.inprogress = asyncio.Event()
        self.completed = asyncio.Semaphore(value=0)
        self.processed_episodes.clear()

    async def put(self, episode_id, start_step, rollout):
        if self.quit or episode_id in self.processed_episodes:
            return
        if episode_id not in self.groups:
            while episode_id not in self.groups and len(self.groups) >= self.max_group_num:
                self.inprogress.clear()
                await self.inprogress.wait()
                if self.quit:
                    return
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
        self.processed_episodes.add(target)
        self.inprogress.set()
        return ret

class PipeGroupQueue(GroupQueue):
    def prepare_clear(self):
        self.quit = True
        # release current waiting put
        for group in self.groups.values():
            for _, event in group:
                if event is not None:
                    event.set()

    def clear(self, progress_bar):
        # assume no more put when clear called
        self.quit = False
        self.progress_bar = progress_bar
        self.groups: Dict[str, List[tuple[DataProto, asyncio.Event|None]]] = {}
        self.completed = asyncio.Event()
        self.episode_ids = set()
        self.max_episode_id = None
        self.processed_episodes.clear()

    async def put(self, episode_id, start_step, rollout):
        if self.quit or episode_id in self.processed_episodes:
            return

        self.episode_ids.add(episode_id)
        assert len(self.episode_ids) <= self.max_traj_per_env
        self.max_episode_id = episode_id if self.max_episode_id is None else max(episode_id, self.max_episode_id)

        event = None
        if len(self.episode_ids) == self.max_traj_per_env and episode_id == self.max_episode_id:
            event = asyncio.Event()

        if episode_id not in self.groups:
            self.groups[episode_id] = []
            assert len(self.groups) <= self.max_traj_per_env
        self.groups[episode_id].append((rollout, event))
        self.completed.set()
        if event is not None:
            await event.wait()
            if self.quit:
                return

    async def get(self):
        target = None
        while target is None:
            for (episode_id, rollouts) in self.groups.items():
                if len(rollouts) >= self.group_size:
                    target = min(episode_id, target) if target is not None else episode_id
            if target is None:
                self.completed.clear()
                await self.completed.wait()
        group = self.groups.pop(target)
        self.processed_episodes.add(target)

        ret = [rollout for rollout, _ in group]
        events = [event for _, event in group]

        last_group = all(event is not None for event in events)
        assert last_group or all(event is None for event in events), f"last_group {last_group}, events {events}"
        if last_group:
            assert target == self.max_episode_id
            assert len(self.groups) == 0
            self.max_episode_id = None
            self.episode_ids.clear()
            for event in events:
                event.set()

        return ret

@ray.remote
class GroupQueueManager:
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
            # There are at most `async_generation_ratio - 1` in-progress steps.
            max_group_num = env_manager_config.max_traj_per_env * (config.async_generation_ratio - 1)
            if max_group_num == 0:
                queue_cls = PipeGroupQueue
            else:
                queue_cls = BoundedGroupQueue
        else:
            # Sync training do not need rate limit, and finished rollouts will never exceed limit
            max_group_num = env_manager_config.max_traj_per_env
            queue_cls = PipeGroupQueue # both GroupQueue and PipeGroupQueue are ok here
        self.group_queue: Dict[int, GroupQueue] = {}
        for rank, rank_env_configs in env_manager_config.env_configs.items():
            for env_id, env_config in rank_env_configs.items():
                group_id = env_config["group_id"]
                if group_id not in self.group_queue:
                    self.group_queue[group_id] = queue_cls(self.progress_bar, env_manager_config.group_size, max_group_num, env_manager_config.max_traj_per_env)

        # for debug
        self.total = 0
        self.waiting = 0

    def prepare_clear(self):
        for group_queue in self.group_queue.values():
            group_queue.prepare_clear()

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
        assert group_id in self.group_queue
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
                        group_rollout = group_rollout[:self.group_size]
                        self.total -= len(group_rollout)
                        ret.extend(group_rollout)
                    assert (done and len(ret) >= batch_size) or (not done and len(ret) <= batch_size)
                    if done:
                        self.pending_gets.update(done)
                self.pending_gets.update(pending)

            assert self.wait_task is None
            self._check_exception()
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
        actor_infer
        train_rollout_scheduler = RolloutScheduler(actor_infer)
        val_rollout_scheduler = RolloutScheduler(actor_infer)
        while True:
            ray.get(train_rollout_scheduler.suspend.remote()) # not neccessary in sync traing
            model_update()
            ray.get(train_rollout_scheduler.resume.remote()) # not neccessary in sync traing
            if val:
                ray.get(val_rollout_scheduler.get_batch.remote())
            ray.get(train_rollout_scheduler.get_batch.remote())
            rollout()
        ray.get(train_rollout_scheduler.stop.remote()) # not neccessary in sync traing
    """
    def __init__(self, config, env_manager_config: EnvManagerConfig, resource_manager, infer_cluster, mode, collator=None):
        self.config = config
        self.env_manager_config = env_manager_config
        self.resource_manager = resource_manager
        self.infer_cluster = infer_cluster
        self.mode = mode

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        self.env_output_queue = GroupQueueManager.options(
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

    async def _start_env_manager(self, global_step):
        assert self.running
        if self.config.async_generation_ratio > 0 and self.mode == "train" and self.rollout_refs is None:
            # async training will only call es_manager.run_rollout_loop once
            seed = self.config.seed
            self.rollout_refs: List[ray.ObjectRef] = self.es_manager.run_rollout_loop(None, seed, blocking=False)
        elif self.config.async_generation_ratio == 0 or self.mode != "train":
            # sync and async val will call es_manager.run_rollout_loop every time get_batch called
            assert self.rollout_refs is None
            if self.mode == "train":
                seed = random.randint(0, 1000000)
            else:
                seed = self.config.seed
            self.rollout_refs: List[ray.ObjectRef] = self.es_manager.run_rollout_loop(global_step, seed, blocking=False)

    async def _stop_env_manager(self, batch_size=None):
        assert self.running # generate_scheudler should be running to avoid block env manager
        # dry event env_output_queue, env_output_queue may block EnvManager
        await self.env_output_queue.prepare_clear.remote()
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in self.es_manager.stop(blocking=False)])
        await self.generate_scheduler.abort_request.remote()
        await asyncio.gather(*self.rollout_refs)
        self.rollout_refs = None
        # reset env_output_queue
        await self.env_output_queue.clear.remote(batch_size)

    async def stop(self):
        """
        Stop env manager for async training, called by user!!!
        """
        if self.config.async_generation_ratio > 0 and self.mode == "train" and self.rollout_refs is not None:
            await self._stop_env_manager()
            await self._stop_server()

    async def _stop_server(self):
        if not self.running:
            return
        self.running = False
        stop_server_tasks = [
            asyncio.wrap_future(ref.obj_ref.future())
            for ref in self.infer_cluster.stop_server(blocking=False)
        ]
        if self.config.async_generation_ratio == 0 or self.mode == "train":
            await asyncio.gather(
                self.alive_check_task,
            )
        gen_metrics = await asyncio.gather(*stop_server_tasks)
        gen_metrics = gen_metrics[0]
        return gen_metrics.meta_info.pop("metrics", {})

    async def suspend(self, global_step):
        if self.config.async_generation_ratio == 0 or self.mode != "train":
            return {}

        if not self.running:
            return {}
        # self.running will be set to False in self._stop_server

        await self.generate_scheduler.suspend.remote()
        return await self._stop_server()

    async def _start_server(self, global_step):
        if self.running:
            return
        self.running = True
        data = DataProto()
        data.meta_info["global_step"] = global_step
        data.meta_info["is_offload_states"] = self.config.async_generation_ratio == 0
        await asyncio.gather(
            *[
                asyncio.wrap_future(ref.obj_ref.future())
                for ref in self.infer_cluster.start_server(data, blocking=False)
            ],
        )
        if self.config.async_generation_ratio == 0 or self.mode == "train":
            self.alive_check_task = asyncio.create_task(self.alive_check())

    async def resume(self, global_step):
        if self.config.async_generation_ratio == 0 or self.mode != "train":
            return

        if self.running:
            return
        # self.running will be set to True in self._start_server

        await asyncio.gather(
            self._start_server(global_step),
            *[
                asyncio.wrap_future(ref.future())
                for ref in self.es_manager.update_step(global_step, blocking=False)
            ],
        )
        await self.generate_scheduler.resume.remote()

    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]

        await self._start_server(global_step)
        await self._start_env_manager(global_step)

        ref = self.env_output_queue.get_batch.remote(batch_size)
        data_batch: List[DataProto] = await asyncio.wrap_future(ref.future())
        metrics = {}
        [append_to_dict(metrics, meta_info.meta_info["metrics"]) for meta_info in data_batch]
        batch = DataProto.concat(data_batch)

        if self.config.async_generation_ratio == 0 or self.mode != "train":
            await self._stop_env_manager(batch_size)
            # stop server in both async val and sync training, assume train_rollout_manager is suspended or stopped
            actor_infer_metrics = await self._stop_server()
            if self.mode == "train":
                metrics.update(actor_infer_metrics)

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
