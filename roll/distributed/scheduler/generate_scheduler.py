import asyncio
import copy
import itertools
import random
import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, fields

from typing import Any, Union, Optional, Dict, List, Set
from contextlib import asynccontextmanager

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import set_seed

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.router import RouterManager
from roll.distributed.scheduler.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from roll.distributed.scheduler.reward_scheduler import RewardScheduler
from roll.distributed.scheduler.rollout_mock_mixin import RolloutMockMixin
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider
from roll.utils.taskgroups import TaskGroup # TODO use official TaskGroup after upgrade to python 3.11
from roll.utils.metrics.metrics_manager import DurationTracker
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger


logger = get_logger()


def expand_requests(data: DataProto, num_return_sequences, is_num_return_sequences_expand):
    """
    Args:
        data (DataProto) [IN|OUT]: 'num_return_sequences' will be overwritten
    """
    assert "generation_config" in data.meta_info, f"data {data.meta_info} should have key 'generation_config'"
    generation_config = data.meta_info["generation_config"]
    target_requests = []
    if is_num_return_sequences_expand:
        generation_config["num_return_sequences"] = 1
        for _ in range(num_return_sequences):
            target_requests.append(copy.deepcopy(data))
    else:
        generation_config["num_return_sequences"] = num_return_sequences
        target_requests.append(copy.deepcopy(data))
    return target_requests

def expand_responses(response: Optional[Union[DataProto, List[DataProto]]]) -> List[DataProto]:
    ret: List[DataProto] = []
    if response is None:
        return ret
    stack = deque([response])
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
        else:
            assert isinstance(current, DataProto)
            batch_size = current.batch.batch_size[0]
            assert batch_size >= 0
            if batch_size > 1:
                stack.extend(reversed([current[i] for i in range(batch_size)]))
            else:
                ret.append(current)
    return ret


@dataclass
class ExperienceItem:
    prompt_id: int
    domain: str = "default"
    sampling_start_step: Optional[int] = None
    data: Optional[DataProto] = None


class ItemsGroup:
    """
    items with the same starting step
    """
    def __init__(self, start_step):
        self.start_step: int = start_step
        self.running_prompts: Set[int] = set()
        self.finished_prompts: deque[List[ExperienceItem]] = deque()
        self.num_samples = 0
        self.event = asyncio.Event()

    def info(self):
        return (
            f"ItemsGroup {self.start_step}: "
            f"{len(self.running_prompts)=} "
            f"{len(self.finished_prompts)=} "
            f"{self.num_samples=} "
        )

    def empty(self) -> bool:
        return len(self.running_prompts) == 0 and len(self.finished_prompts) == 0

    def get_running_prompt_ids(self) -> Set[int]:
        return self.running_prompts

    def begin_prompt(self, prompt_id):
        assert prompt_id not in self.running_prompts
        self.running_prompts.add(prompt_id)

    def commit_prompt(self, prompt_id: int, result: List[ExperienceItem]):
        self.running_prompts.remove(prompt_id)
        assert prompt_id not in self.finished_prompts
        self.finished_prompts.append(result)
        self.num_samples += len(result)
        self.event.set()

    def abort_prompt(self, prompt_id: int):
        self.running_prompts.remove(prompt_id)
        assert prompt_id not in self.finished_prompts
        self.event.set()

    async def get_batch(self, expected_samples) -> List[List[ExperienceItem]]:
        """
        Get at most batch_size * num_return_sequences samples from finished prompts,
        blocking wait when there are running and has not collect enough responses.
        """
        assert expected_samples >= 0
        while self.num_samples < expected_samples and not len(self.running_prompts) == 0:
            self.event.clear()
            await self.event.wait()
        if self.num_samples <= expected_samples:
            result = list(self.finished_prompts)
            collected_samples = self.num_samples
            self.finished_prompts = deque()
            self.num_samples = 0
        else:
            result = []
            collected_samples = 0
            while collected_samples < expected_samples and self.finished_prompts:
                item = self.finished_prompts.popleft()
                result.append(item)
                collected_samples += len(item)
                self.num_samples -= len(item)
        assert sum(len(item) for item in self.finished_prompts) == self.num_samples
        assert collected_samples == sum(len(item) for item in result)
        # collected_samples may greater than expected_samples
        return result


class ReplayBuffer:
    """
    Provide a transactional interface to control running and finished prompts.

    Both sync and async training are supported (sync training is special case of async training).

    Limit running prompts (not aware of num_return_sequences) below batch_size
    or batch_size + max_additional_running_prompts.

    ReplayBuffer only control how many prompts can be sent at the same time and do not
    provide rate limit or load balance to ActorInfer or RewardWorker.

    Public interface:
        * advance_step: update current step and increate total batch size. (think of
            this as an epoch-based reclamation (or epoch-based garbage collection))
        * poll: poll for a prompt_id with implicit rate limit
        * begin: bind prompt_id to current step
        * commit/abort: accept or filter out responses of a prompt
        * gc: garbage collect outdated running or committed(finished) prompts
            (sync training will clear all stored but not used prompts)

    Usage: see tests/distributed/scheduler/test_generate_scheduler.py and RolloutContext.
    """
    def __init__(
        self,
        async_generation_ratio,
        is_use_additional_prompts,
        max_additional_running_prompts
    ):
        self.pid = 0
        self.current_step = None
        self.groups: Dict[int, ItemsGroup] = {}
        self.prompt_id_to_start_step: Dict[int, int] = {} # only store map info for running prompts

        self.batch_size = 0
        assert async_generation_ratio >= 0
        assert not is_use_additional_prompts or max_additional_running_prompts > 0
        self.async_generation_ratio = async_generation_ratio
        self.is_use_additional_prompts = is_use_additional_prompts
        self.max_additional_running_prompts = max_additional_running_prompts

        self._shutdown = False

        self.running_prompts = 0 # prompts in running state
        self.running_extra_prompts = 0 # additinal prompts in running state
        # only running_prompts not running_extra_prompts will add to completed_prompts at commit
        # so running_prompts + completed_prompts may less than used_prompts if is_use_additional_prompts is set
        self.completed_prompts = 0 # prompts in commit state
        self.event = asyncio.Event()
        self.advance_step_event = asyncio.Event()

    def info(self) -> str:
        group_info = [group.info() for group in self.groups.values()]
        return (
            f"ReplayBuffer: {self.current_step=} {self.batch_size=} "
            f"{self.used_prompts=}  {self.completed_prompts=} {self.running_prompts=} {self.running_extra_prompts=} "
            f"{group_info=} "
        )

    def _next_pid(self):
        pid = self.pid
        self.pid += 1
        return pid

    @property
    def used_prompts(self) -> int:
        return self.pid

    def shutdown(self):
        self._shutdown = True
        self.event.set()
        self.advance_step_event.set()

    def advance_step(self, step, batch_size):
        # step must increase monotonically
        assert not self.current_step or step > self.current_step
        assert step not in self.groups
        old_step = self.current_step
        old_batch_size = self.batch_size
        self.current_step = step
        # Must create ItemsGroup before get_batch. Otherwise,
        # if user call get_batch with min_step = current_step before
        # any process_new_prompt task is scheduled, get_batch will skip
        # current step and return empty list which breaks the postcondition
        # of get_enough_finished_prompts.
        self.groups[step] = ItemsGroup(start_step=step)
        if self.batch_size == 0 and self.async_generation_ratio > 0:
            # first step
            self.batch_size = math.ceil(batch_size * self.async_generation_ratio)
        self.batch_size += batch_size
        logger.info(f"advance_step from {old_step} to {self.current_step}, "
                    f"batch_size from {old_batch_size} to {self.batch_size}")
        self.event.set()
        self.advance_step_event.set()

    def _check_send_new_request(self) -> bool:
        if self.running_prompts + self.completed_prompts < self.batch_size:
            self.running_prompts += 1
            return True
        elif self.is_use_additional_prompts and self.batch_size > 0 and (self.running_prompts + self.running_extra_prompts < self.max_additional_running_prompts):
            # condition self.batch_size>0 ensure not at initialization stage
            self.running_extra_prompts += 1
            return True
        else:
            return False

    async def poll(self) -> int:
        """
        Will blocking wait when can not send new request and is not in shutdown stage.
        """
        prompt_id = self._next_pid()
        while True:
            if self._shutdown:
                raise asyncio.CancelledError
            elif self._check_send_new_request():
                self.prompt_id_to_start_step[prompt_id] = None
                return prompt_id
            self.event.clear()
            await self.event.wait()

    async def begin(self, prompt_id) -> int:
        """
        Blocking wait if group of current_step not exists (happen during garbage collection).

        Exception:
            asyncio.CancelledError: if prompt_id is aborted or at system shutdown stage.
        """
        assert self.current_step is not None
        while True:
            start_step = self.current_step
            if start_step in self.groups:
                assert start_step == next(reversed(self.groups))
                break
            elif self._shutdown:
                raise asyncio.CancelledError
            self.advance_step_event.clear()
            await self.advance_step_event.wait()
        if prompt_id not in self.prompt_id_to_start_step:
            raise asyncio.CancelledError
        assert self.prompt_id_to_start_step[prompt_id] is None, f"{prompt_id=} {self.prompt_id_to_start_step[prompt_id]=}"
        self.prompt_id_to_start_step[prompt_id] = start_step
        group = self.groups[start_step]
        group.begin_prompt(prompt_id)
        return start_step

    def _commit_prompt(self):
        assert self.running_prompts + self.running_extra_prompts > 0
        if self.running_prompts > 0:
            self.running_prompts -= 1
            self.completed_prompts += 1
        else:
            self.running_extra_prompts -= 1

    def commit(self, prompt_id, result: List[ExperienceItem]):
        try:
            if prompt_id not in self.prompt_id_to_start_step:
                # Prompt has finished or has been garbage collected.
                raise asyncio.CancelledError
            start_step = self.prompt_id_to_start_step.pop(prompt_id)
            if start_step is None:
                raise RuntimeError("Prompt has not been bind to step, cannot commit.")
            assert start_step in self.groups, "group must exits between begin and commit/abort"
            # assert len(result) == num_return_sequences
            assert all(item.sampling_start_step == start_step for item in result)
            self._commit_prompt()
            self.groups[start_step].commit_prompt(prompt_id, result)
        finally:
            self.event.set()

    def _abort_prompt(self):
        assert self.running_prompts + self.running_extra_prompts > 0
        if self.running_prompts > 0:
            self.running_prompts -= 1
        else:
            self.running_extra_prompts -= 1

    def abort(self, prompt_id):
        try:
            if prompt_id not in self.prompt_id_to_start_step:
                # Prompt has finished or has been garbage collected.
                return
            start_step = self.prompt_id_to_start_step.pop(prompt_id)
            if start_step is None:
                # Prompt has not been bind to step.
                self._abort_prompt()
                return
            assert start_step in self.groups, "group must exits between begin and commit/abort"
            self._abort_prompt()
            self.groups[start_step].abort_prompt(prompt_id)
        finally:
            self.event.set()

    async def get_batch(self, expected_samples: int, progress_bar) -> List[ExperienceItem]:
        """
        await on this function to wait for enough requests to be collected

        Assume self.groups will not be mutated during this function, i.e. min(self.groups,keys) and
        max(self.groups.keys) will not change and no iterator invalidation.
        """
        min_step = self.current_step - math.ceil(self.async_generation_ratio)
        # min_step_group must exist or min_step must less than min step of self.groups
        assert min_step in self.groups or next(iter(self.groups)) > min_step

        logger.info(f"ReplayBuffer get_batch: {self.current_step=} {min_step=} {expected_samples=}, {self.info()}")

        collected_samples = 0
        responses: List[List[DataProto]] = []

        # Iter on self.groups will go from small step to large step.
        for step, group in self.groups.items():
            if step < min_step:
                continue
            elif step == self.current_step:
                # special case: current running step, scheduler may have not send
                # new prompts (because of concurrency), we handle it outside this for loop.
                break
            elif step == min_step:
                if self.async_generation_ratio % 1 == 0:
                    expected = expected_samples - collected_samples
                else:
                    expected = int(expected_samples * (self.async_generation_ratio % 1))
            else:
                expected = expected_samples - collected_samples
            finished_prompts = await group.get_batch(expected_samples=expected)
            amount = sum(len(response) for response in finished_prompts)
            collected_samples += amount
            progress_bar.update(amount)
            responses.extend(finished_prompts)
            if collected_samples >= expected_samples:
                break

        # step == self.current_step, wait for scheduler to send enough new prompts
        while collected_samples < expected_samples:
            # There may be no running prompt at this time,
            # yield control to schedule process_new_prompt.
            await asyncio.sleep(0)
            finished_prompts = await group.get_batch(expected_samples=expected_samples-collected_samples)
            amount = sum(len(response) for response in finished_prompts)
            collected_samples += amount
            # If want to update progress_bar in a fine-grained manner, can call get_batch on the latest step
            # with expected_samples=num_return_sequences. But it will increase overhead.
            progress_bar.update(amount)
            responses.extend(finished_prompts)

        result: List[ExperienceItem] = []
        for response in responses:
            result.extend(response)
        assert len(result) == collected_samples
        return result

    def gc(self) -> List[int]:
        """
        Garbage collect old ItemsGroup.

        Return aborted prompt ids to scheduler(caller) and
        rely on scheduler(caller) to abort generate requests.

        Assume called after get_batch(step=current_step) and before advance_step(step=current_step + 1),
        and will garbage collect steps equal or less than current_step - min_start_step.

        User must handle return value of begin and commit to check whether should shutdown.
        """
        # Assume the following operations are atomic (no yiled).
        if self.current_step is None:
            assert not self.groups
            return []
        max_gc_step = self.current_step - math.ceil(self.async_generation_ratio)
        assert max_gc_step == self.current_step or self.async_generation_ratio > 0
        assert not self.groups or max_gc_step <= next(reversed(self.groups))
        aborted_prompts = []
        # Must remove group after all prompts are gracefully aborted.
        aborted_groups = []
        # Iter on self.groups will go from small step to large step.
        for step, group in self.groups.items():
            if step <= max_gc_step:
                aborted_prompts.extend(group.get_running_prompt_ids())
                aborted_groups.append(step)
        for prompt_id in aborted_prompts:
            self.abort(prompt_id)
        for step in aborted_groups:
            assert self.groups[step].empty() or self.is_use_additional_prompts, f"{step=} {self.groups[step]=}"
            del self.groups[step]
        logger.info(f"ReplayBuffer {self.current_step=} {max_gc_step=} garbage collect groups {aborted_groups}")
        return aborted_prompts


class DynamicSamplingScheduler(RolloutMockMixin):
    def __init__(
        self,
        pipeline_config,
        actor_cluster: Union[Any, Cluster],
        reward_clusters: Dict[str, Union[Any, Cluster]],
        dataset: Dataset,
        collect_fn_cls,
        collect_fn_kwargs,
        state: Dict[str, Any] = None,
        is_val: bool = False,
    ):
        self.pipeline_config = pipeline_config
        set_seed(seed=pipeline_config.seed)

        self.is_val = is_val
        if self.is_val:
            self.sequence_length = self.pipeline_config.val_sequence_length
            logger.info(f"validation generate scheduler sequence_length is: {self.sequence_length}")
        else:
            self.sequence_length = pipeline_config.sequence_length
            logger.info(f"training generate scheduler sequence_length is: {self.sequence_length}")

        # Initialize rollout mock mechanism from mixin (after is_val is set)
        self._init_rollout_mock()

        self.actor_cluster = actor_cluster

        self.reward_clusters = reward_clusters
        self.reward_worker_iters = {}
        for domain, cluster in reward_clusters.items():
            self.reward_worker_iters[domain] = itertools.cycle(cluster.workers)

        # metrics of a step
        self.generate_timer = {domain: DurationTracker() for domain in reward_clusters.keys()}
        self.reward_timer = {domain: DurationTracker() for domain in reward_clusters.keys()}

        self.request_id = uuid.uuid4()
        self.request_counter = 0

        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if state is not None and state.get("dataset_iter_count", 0) > 0:
            for _ in range(state["dataset_iter_count"]):
                self.get_next_dataset_item()
        self.dataset_epoch = 0
        self.dataset_iter = None
        self.dataset_iter_count = 0

        self.collect_fn_cls = collect_fn_cls
        self.collect_fn_kwargs = collect_fn_kwargs
        self.tokenizer = default_tokenizer_provider(model_args=self.actor_cluster.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.actor_cluster.worker_config.model_args)
        if "processor" in [f.name for f in fields(collect_fn_cls)]:
            collect_fn_kwargs["processor"] = self.processor
        self.collect_fn = self.collect_fn_cls(tokenizer=self.tokenizer, **self.collect_fn_kwargs)

        self.async_sending_task = None

        # Dynamic filter is supported no matter whether is_use_additional_prompts,
        # is_use_additional_prompts is required when using dynamic num_return_sequences.
        self.replay_buffer = ReplayBuffer(
            async_generation_ratio=self.pipeline_config.async_generation_ratio if not is_val else 0,
            is_use_additional_prompts=self.pipeline_config.is_use_additional_prompts if not is_val else False,
            max_additional_running_prompts=self.pipeline_config.max_additional_running_prompts if not is_val else 0,
        )

        self.router_manager = RouterManager(self.actor_cluster, router_args=self.pipeline_config.router_args,
                                            num_gpus_per_node=self.pipeline_config.num_gpus_per_node)
        self.router_client = None

        self.running_tasks: Dict[int, asyncio.Task] = {}

        # meta_info is reassigned every step
        self.meta_info = None

        udrl_cls = safe_import_class(pipeline_config.user_defined_rollout_loop_cls)
        assert udrl_cls
        self.udrl = udrl_cls()

        self.reward_scheduler = RewardScheduler()

    async def initialize(self):
        await self.router_manager.initialize()
        self.router_client = await RouterManager.create_client(self.router_manager)

        # async_sending_task is paused at start. But can not call self.pause_sampling directly here,
        # becauseuActorInfer.strategy has not been initialized yet and is not ready to serve abort_requests rpc.
        self.router_manager.suspend()

        # async_sending_task coroutine will last during the whole training process, only stop at shutdown or exception.
        # Because we do not need to pause all running prompts but only suspend generate requests, so that reward requests
        # still can run during model update.
        self.async_sending_task = asyncio.create_task(self.sending_request())

    def next_request_id(self):
        request_id = f"{self.request_id}_{self.request_counter}"
        self.request_counter += 1
        return request_id

    def gc(self):
        aborted_prompts = self.replay_buffer.gc()
        for prompt_id in aborted_prompts:
            if task := self.running_tasks.get(prompt_id, None):
                task.cancel()

    async def pause_sampling(self):
        self.router_manager.suspend()
        self.gc()
        await self.router_manager.abort_all()
        await self.router_manager.wait_complete()
        logger.info(f"sampling paused, replay_buffer info: {self.replay_buffer.info()}")

    async def shutdown(self):
        self.replay_buffer.shutdown()
        await self.router_manager.shutdown()
        self.gc()
        await self.async_sending_task

    async def get_batch_opt_level_0(self, data: DataProto, batch_size: int) -> DataProto:
        generation_config = copy.deepcopy(data.meta_info["generation_config"])
        completed_data: List[DataProto] = []
        query_use_count = 0
        if self.is_val:
            query_filter_fn = lambda data_list, config: True
        else:
            from roll.distributed.scheduler.user_defined_rollout_loop import query_filter
            query_filter_fn = query_filter
        query_filter_count = 0

        while len(completed_data) < batch_size:
            data_item_list = [self.get_next_dataset_item() for _ in range(batch_size)]
            collect_data = self.collect_fn(data_item_list)
            request_data: DataProto = DataProto.from_single_dict(collect_data, meta_info=data.meta_info)
            request_data.batch["prompt_id"] = torch.arange(request_data.batch.batch_size[0], device=request_data.batch.device)

            gen_batch = request_data.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
            gen_batch.meta_info = request_data.meta_info
            num_return_sequences = generation_config["num_return_sequences"]
            request_data = request_data.repeat(repeat_times=num_return_sequences)

            # Pad gen_batch to be divisible by dp_size to avoid errors
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_cluster.dp_size)
            batch: DataProto = self.actor_cluster.generate(gen_batch_padded)
            batch = unpad_dataproto(batch, pad_size * num_return_sequences)

            batch.union(other=request_data)
            batch.rename(old_keys="prompt_id", new_keys="origin_prompt_id")
            batch_rewards = await self.reward_scheduler.compute_rewards(data=batch, reward_clusters=self.reward_clusters, pipeline_config=self.pipeline_config)
            metrics = batch.meta_info.pop("metrics", {})
            metrics.update(batch_rewards.meta_info.pop("metrics", {}))

            batch.union(other=batch_rewards)

            batch.meta_info["metrics"] = metrics
            batch_grouped: Dict[str, DataProto] = batch.group_by("origin_prompt_id")
            for prompt_id, batch_item in batch_grouped.items():
                if query_filter_fn([batch_item], self.pipeline_config):
                    completed_data.append(batch_item)
                else:
                    query_filter_count += 1
            query_use_count += batch_size

        batch = DataProto.concat(completed_data[: batch_size])
        batch.meta_info["metrics"] = {
            f"scheduler/query_filter_count": query_filter_count,
            f"scheduler/collect_query_count": batch_size,
            f"scheduler/query_use_count": query_use_count,
        }
        return batch

    async def get_batch(self, data: DataProto, global_step: int, batch_size: int) -> DataProto:
        # MOCK MODE: Load pre-recorded data, skip rollout (from mixin)
        if self._should_load_mock(global_step):
            return await self._load_mock_batch(global_step)

        if self.pipeline_config.generate_opt_level == 0:
            return await self.get_batch_opt_level_0(data, batch_size)

        num_return_sequences = data.meta_info["generation_config"]["num_return_sequences"]
        self.meta_info = copy.deepcopy(data.meta_info)
        self.meta_info["collect_non_finish"] = self.pipeline_config.async_generation_ratio > 0

        assert self.router_manager.size() == 0, f"worker state: {self.router_manager.size()}"
        # Notice: self.replay_buffer.running_prompts may not be 0 because
        # pause_sampling only pause generate request but not reward request.

        self.replay_buffer.advance_step(step=global_step, batch_size=batch_size)
        logger.info(f"start sampling, {global_step=} {batch_size=}, {self.replay_buffer.info()}")
        self.router_manager.resume()

        bar_name = "-".join(self.reward_clusters.keys())
        progress_bar = tqdm(
            total=batch_size * num_return_sequences,
            desc=f"{bar_name} generate progress(prompt)",
            mininterval=int(batch_size * 0.1) + 1,
        )

        # await on both get_batch and async_sending_task to receive exception from async_sending_task as soon as possible
        get_task = asyncio.create_task(
            self.replay_buffer.get_batch(expected_samples=batch_size * num_return_sequences, progress_bar=progress_bar)
        )
        await asyncio.wait({get_task, self.async_sending_task}, return_when=asyncio.FIRST_COMPLETED)
        if self.async_sending_task.done():
            assert self.async_sending_task.exception() is not None
            await self.async_sending_task
        assert get_task.done()
        finished_items = await get_task

        if self.pipeline_config.is_use_additional_prompts:
            # Keep the first batch_size*num_return_sequences ExperienceItem now.
            assert len(finished_items) >= batch_size * num_return_sequences
            finished_items = finished_items[:batch_size * num_return_sequences]
        assert len(finished_items) == batch_size * num_return_sequences
        batch = self.collect_items_as_batch(finished_items=finished_items)

        if self.is_val or self.pipeline_config.async_generation_ratio <= 0:
            await self.pause_sampling()
            assert not self.replay_buffer.groups, f"{self.replay_buffer.groups=}"

        for domain in self.reward_clusters.keys():
            metrics = {}
            generate_stat = self.generate_timer[domain].log()
            metrics[f"scheduler/{domain}/time/generate/count"] = generate_stat["count"]
            metrics[f"scheduler/{domain}/time/generate/min"] = generate_stat["min"]
            metrics[f"scheduler/{domain}/time/generate/max"] = generate_stat["max"]
            metrics[f"scheduler/{domain}/time/generate/mean"] = generate_stat["mean"]
            reward_stat = self.reward_timer[domain].log()
            metrics[f"scheduler/{domain}/time/reward/count"] = reward_stat["count"]
            metrics[f"scheduler/{domain}/time/reward/min"] = reward_stat["min"]
            metrics[f"scheduler/{domain}/time/reward/max"] = reward_stat["max"]
            metrics[f"scheduler/{domain}/time/reward/mean"] = reward_stat["mean"]
            batch.meta_info["metrics"].update(metrics)

        # DUMP MODE: Save merged batch (from mixin)
        await self._maybe_dump_batch(batch, global_step)

        return batch

    def collect_items_as_batch(self, finished_items: List[ExperienceItem]) -> DataProto:
        collect_data_by_domain = defaultdict(list)
        data_off_policy_step = 0.0
        prompt_ids = set()
        for item in finished_items:
            collect_data_by_domain[item.domain].append(item.data)
            data_off_policy_step += self.replay_buffer.current_step - item.sampling_start_step
            prompt_ids.add(item.prompt_id)
        data_off_policy_step = data_off_policy_step / len(finished_items)

        collect_data_by_domain = {
            domain: DataProto.concat(data_list) for domain, data_list in collect_data_by_domain.items()
        }
        query_use_count = len(prompt_ids)
        collect_data_num = sum(data.batch.batch_size[0] for data in collect_data_by_domain.values())
        assert collect_data_num == len(finished_items)
        logger.info(f"total collect data: {collect_data_num}, collect queries: {query_use_count}")

        batch = DataProto.concat(list(collect_data_by_domain.values()))
        # TODO support response_filter_count and query_filter_count
        batch.meta_info.setdefault("metrics", {}).update({
            f"scheduler/collect_query_count": query_use_count,
            f"scheduler/query_use_count": query_use_count,
            f"scheduler/off_policy_ratio": data_off_policy_step,
        })

        metrics = {}
        for domain, response_batch in collect_data_by_domain.items():
            sequence_score = response_batch.batch["scores"]
            metrics[f"scheduler/{domain}/score/mean"] = torch.mean(sequence_score).detach().item()
            metrics[f"scheduler/{domain}/score/max"] = torch.max(sequence_score).detach().item()
            metrics[f"scheduler/{domain}/score/min"] = torch.min(sequence_score).detach().item()
        batch.meta_info["metrics"].update(metrics)

        # TODO shigao implement REPORT_LENGTH_AND_REWARDS (deleted at refactor)

        return batch

    async def sending_request(self):
        async with TaskGroup() as tg:
            while True:
                try:
                    prompt_id = await self.replay_buffer.poll()
                except:
                    logger.info(f"stop sending_request coroutine")
                    break
                task = tg.create_task(RolloutContext.process_new_prompt(scheduler=self, prompt_id=prompt_id))
                self.running_tasks[prompt_id] = task

            # The above loop only break at shutdown, it is safe to abort all infligh requests here.
            await self.router_manager.abort_all()
            # Implicitly wait until all running tasks finished when TaskGroup context exit.

    def get_next_dataset_item(self):
        if self.dataset_iter is None:
            random.seed(self.pipeline_config.seed + self.dataset_epoch)
            random.shuffle(self.indices)
            self.dataset_iter = iter(self.indices)
            logger.info(f"{'-'.join(self.reward_clusters.keys())} dataset epoch: {self.dataset_epoch}")

        try:
            dataset_item = self.dataset[next(self.dataset_iter)]
        except StopIteration:
            self.dataset_epoch += 1
            random.seed(self.pipeline_config.seed + self.dataset_epoch)
            random.shuffle(self.indices)
            self.dataset_iter = iter(self.indices)
            dataset_item = self.dataset[next(self.dataset_iter)]
            logger.info(f"{'-'.join(self.reward_clusters.keys())} dataset epoch: {self.dataset_epoch}")
        self.dataset_iter_count += 1
        return dataset_item

    def get_scheduler_state(self):
        return {"dataset_iter_count": self.dataset_iter_count}


class RolloutContext:
    """
    Helper class to manage life cycle of rollout of a prompt.
    Provide a context manager based interface to user and hide implementation
    details of DynamicSamplingScheduler, Router and ReplayBuffer from user.
    """

    @staticmethod
    async def process_new_prompt(
        scheduler: DynamicSamplingScheduler,
        prompt_id: int,
    ):
        num_return_sequences = scheduler.meta_info["generation_config"]["num_return_sequences"]
        context = RolloutContext(scheduler=scheduler, prompt_id=prompt_id, meta_info=scheduler.meta_info)
        success = False
        try:
            responses = await scheduler.udrl.process_new_prompt(context=context)
            if responses is None:
                logger.info(f"filter out prompt {prompt_id}")
                raise asyncio.CancelledError # abort this prompt
            responses = expand_responses(responses)
            assert (
                len(responses) == num_return_sequences or scheduler.replay_buffer.is_use_additional_prompts
            ), "is_use_additional_prompts is required when using dynamic num_return_sequences"
        except Exception as e:
            logger.warning(f"abort prompt {prompt_id} on exception {e}")
            raise
        else:
            success = True
        finally:
            scheduler.running_tasks.pop(prompt_id, None)

            # commit/abort should be put at last in finally block, because commit may raise exception
            if not success:
                scheduler.replay_buffer.abort(prompt_id)
            else:
                assert context.sampling_start_step is not None
                scheduler.replay_buffer.commit(
                    prompt_id,
                    [
                        ExperienceItem(
                            prompt_id=prompt_id,
                            domain=context.domain,
                            sampling_start_step=context.sampling_start_step,
                            data=response,
                        )
                        for response in responses
                    ],
                )

    def __init__(
        self,
        scheduler: DynamicSamplingScheduler,
        prompt_id: int,
        meta_info,
    ):
        # store reference to scheduler as a 'proxy'
        self._scheduler: DynamicSamplingScheduler = scheduler

        # export system/prompt level meta info and config to user
        self.prompt_id = prompt_id
        self.meta_info = copy.deepcopy(meta_info) # user may change config in meta_info
        self.pipeline_config = scheduler.pipeline_config
        self.is_val = scheduler.is_val
        self.sequence_length = scheduler.sequence_length
        self.prompt_length = scheduler.pipeline_config.prompt_length
        self.is_num_return_sequences_expand = scheduler.pipeline_config.is_num_return_sequences_expand

        # User can call reward worker of different domain in for a single data, but ExperienceItem.domain is bind to dataset
        self.domain = None

        # The following attributes are setted after generate and reward begin.
        self.sampling_start_step = None
        self.inflight_requests = None
        self._in_do_generate_and_reward = False

    def get_request_data(self, meta_info):
        assert not self._in_do_generate_and_reward and self.sampling_start_step is None

        # TODO allow user to get multiple data inside a RolloutContext?
        if getattr(self, "got_data", False):
            raise RuntimeError("Should call get_request_data only once."
                               " If want ot filter dataset item out, return None"
                               " and let framework to schedule another prompt")
        else:
            self.got_data: bool = True

        dataset_item = self._scheduler.get_next_dataset_item()
        domain = dataset_item.get("domain", "default")
        collect_data = self._scheduler.collect_fn([dataset_item])
        self.domain = domain
        return DataProto.from_single_dict(collect_data, meta_info=meta_info), domain

    @asynccontextmanager
    async def do_generate_and_reward(self, max_concurrency):
        assert not self._in_do_generate_and_reward and self.sampling_start_step is None
        self._in_do_generate_and_reward = True

        # Assume sampling_start_step of all samples of this prompt are the same, however
        # the real sampling_start_step can be different from self.sampling_start_step.
        self.sampling_start_step = await self._scheduler.replay_buffer.begin(prompt_id=self.prompt_id)
        self.inflight_requests = set()
        try:
            yield
        finally:
            assert not self.inflight_requests, f"User should gather all running requests: {self.inflight_requests}"
            self._in_do_generate_and_reward = False

    async def generate(
        self,
        req: DataProto,
        domain: str,
    ) -> DataProto:
        assert self._in_do_generate_and_reward
        with self._scheduler.generate_timer[domain].track():
            request_id = self._scheduler.next_request_id()
            req.meta_info["request_id"] = request_id
            logger.debug(f"generate_and_reward: {self.prompt_id=} {request_id} generate_request")
            self.inflight_requests.add(request_id)
            try:
                return await self._scheduler.router_client.generate_request(req=req, request_id=request_id, uid=self.prompt_id)
            finally:
                self.inflight_requests.remove(request_id)

    async def compute_rewards(
        self,
        req: DataProto,
        domain: str,
    ) -> DataProto:
        # reward worker得能支持单条数据计算, dynamic sampling对需要batch计算reward的需要注意...
        # 多域的时候,llm as judge, 需要单独为reward worker分配gpu
        assert self._in_do_generate_and_reward
        with self._scheduler.reward_timer[domain].track():
            reward_worker = next(self._scheduler.reward_worker_iters[domain])
            logger.debug(f"generate_and_reward: {self.prompt_id=} compute_rewards")
            output_count = req.batch.batch_size[0]
            req.non_tensor_batch["rollout_id"] = np.array([str(uuid.uuid4()) for _ in range(output_count)], dtype=object)
            return await reward_worker.compute_rewards.remote(req)

    async def abort_running_requests(self):
        """
        Abort all running requests.

        Notice: Both abort and partial rollout will return a unfinished response, user should distinguish
        these two cases by themselves to avoid dead loop (if do not check abort state in multi-turn rollout,
        may send generate request again).
        """
        assert self._in_do_generate_and_reward
        assert self.prompt_id is not None
        self._scheduler.router_manager.abort_requests(request_ids=list(self.inflight_requests), uid=self.prompt_id)
