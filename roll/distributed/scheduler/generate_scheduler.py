import asyncio
import copy
import itertools
import random
import math
import uuid
import time
from collections import defaultdict, deque
from dataclasses import dataclass, fields
from itertools import cycle

from typing import Any, Union, Optional, Dict, List, Set
from contextlib import asynccontextmanager

import numpy as np
import ray
import torch
from datasets import Dataset
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import set_seed

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto, collate_fn, pad_dataproto_to_divisor, unpad_dataproto
from roll.distributed.scheduler.reward_scheduler import RewardScheduler
from roll.distributed.scheduler.rollout_mock_mixin import RolloutMockMixin
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider
from roll.utils.functionals import (
    postprocess_generate,
    concatenate_input_and_output,
)
from roll.utils.taskgroups import TaskGroup # TODO use official TaskGroup after upgrade to python 3.11
from roll.utils.metrics.metrics_manager import DurationTracker
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger


logger = get_logger()


def is_report_data_finished(data: DataProto) -> bool:
    finish_reasons = data.meta_info.get("finish_reasons", [])
    assert isinstance(finish_reasons, list), f"{finish_reasons}"
    assert all(isinstance(finish_reason, str) for finish_reason in finish_reasons), f"{finish_reasons}"
    return not any(finish_reason == "abort" for finish_reason in finish_reasons)

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


class LoadBalancer:
    """
    Manage a bunch or workers (worker indexes). Limit the number of running requests of
    each dp below max_running_requests.

    Usage: see document of ReplayBuffer

    TODO support rescheduling to differente dp_rank after model update or support dynamic dp_rank.
    """

    class Lease:
        def __init__(self, load_balancer: "LoadBalancer", lease: int, dp_rank: int):
            self.mutex = asyncio.Lock()
            self.load_balancer = load_balancer
            self.lease = lease
            self.limit = lease
            self.running = 0
            self._dp_rank = dp_rank

        def __del__(self):
            # User must call clear or consume all lease to give back credit explicitly.
            assert self.lease == 0

        def clear(self):
            assert self.lease >= 0
            assert self.running == 0
            self.load_balancer._release(self._dp_rank, credit=self.lease)
            self.limit = 0
            self.lease = 0

        @asynccontextmanager
        async def lock(self, samplen):
            assert self.running + samplen <= self.limit # user should not over-subscribe lease
            self.running += samplen # must add outside the lock

            async with self.mutex:
                if self.lease < samplen:
                    extra = samplen - self.lease
                    await self.load_balancer._reacquire(self._dp_rank, credit=extra)
                    self.lease += extra
                    assert self.lease == samplen
                self.lease -= samplen

            try:
                # return dp_rank explicitly, may support dynamic dp_rank
                yield self._dp_rank
            finally:
                self.running -= samplen
                self.load_balancer._release(self._dp_rank, credit=samplen)

    def __init__(self, mp_rank_zero: Dict[int, any], max_running_requests: int):
        self.workers = {} # key: dp_rank, value: running_requests
        self.worker_acquire_event = {}
        for dp_rank in mp_rank_zero.keys():
            self.workers[dp_rank] = 0
            self.worker_acquire_event[dp_rank] = asyncio.Event()

        self.max_running_requests = max_running_requests
        self.running_request = 0
        self.acquire_event = asyncio.Event()

        self._suspend = False
        self.suspend_event = asyncio.Event()
        self.empty_event = asyncio.Event()

    async def acquire(self, credit: int) -> Lease:
        """
        Dispatching n sample of a prompt to the same worker using best fit strategy (using
        linear search for simplicity), blocking wait if no worker is available.
        """
        while True:
            while self._suspend:
                self.suspend_event.clear()
                await self.suspend_event.wait()

            target = -1
            for dp_rank, running_requests in self.workers.items():
                if running_requests >= self.max_running_requests:
                    continue
                if target == -1 or running_requests < self.workers[target]:
                    target = dp_rank
            if target != -1:
                # FIXME may send more than max_running_requests (i.e. workers[target] + credit > max_running_requests)
                self.workers[target] += credit
                self.running_request += credit
                return self.Lease(self, lease=credit, dp_rank=target)
            self.acquire_event.clear()
            await self.acquire_event.wait()

    async def _reacquire(self, dp_rank: int, credit: int) -> int:
        """
        For multi-turn rollout.
        """
        assert dp_rank in self.workers
        while True:
            while self._suspend:
                self.suspend_event.clear()
                await self.suspend_event.wait()

            if self.workers[dp_rank] < self.max_running_requests:
                self.workers[dp_rank] += credit
                self.running_request += credit
                return
            self.worker_acquire_event[dp_rank].clear()
            await self.worker_acquire_event[dp_rank].wait()

    def _release(self, dp_rank: int, credit: int = 1):
        assert credit >= 0
        self.workers[dp_rank] -= credit
        self.running_request -= credit
        assert self.workers[dp_rank] >= 0
        assert self.running_request >= 0
        self.acquire_event.set()
        self.worker_acquire_event[dp_rank].set()
        self.empty_event.set()

    def empty(self) -> bool:
        return sum(self.workers.values()) == 0

    def full(self) -> bool:
        return all(running_requests >= self.max_running_requests for running_requests in self.workers.values())

    def suspend(self):
        """
        Suspend all running requests.

        User calling acquire and suspended will be blocked after suspend.
        """
        if self._suspend:
            return
        self._suspend = True

    async def wait_complete(self):
        """
        Wait until all running requests are finished (no matter
        whether suspended or not).
        """
        while self.running_request > 0:
            self.empty_event.clear()
            await self.empty_event.wait()

    def resume(self):
        self._suspend = False
        self.suspend_event.set()


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

    Often used with LoadBalancer. ReplayBuffer control how many prompts can be
    sent at the same time, and LoadBalancer limit request to ActorInfer and RewardWorker.
    The real concurrency is limited by both ReplayBuffer and LoadBalancer.

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


class Scheduler:
    def __init__(self):
        self.request_id = uuid.uuid4()
        self.request_counter = 0

    def next_request_id(self):
        request_id = f"{self.request_id}_{self.request_counter}"
        self.request_counter += 1
        return request_id


@ray.remote
class GenerateScheduler(Scheduler):
    def __init__(self, pipeline_config=None):
        super().__init__()
        self.cluster: Union[Any, Cluster] = None
        self.pipeline_config = pipeline_config

        self.mp_rank_zero = {}
        self.max_running_requests = 128
        self.load_balancer = None

    async def generate(self, data: DataProto, actor_cluster: Union[Any, Cluster], pipeline_config) -> DataProto:
        assert self.pipeline_config is None or pipeline_config is self.pipeline_config
        if self.cluster is None:
            self.cluster = actor_cluster
            dp_ranks: List[int] = [rank_info.dp_rank for rank_info in self.cluster.worker_rank_info]
            for i, dp_rank in enumerate(dp_ranks):
                rank_info = self.cluster.get_rank_info(rank=i)
                if rank_info.tp_rank == 0 and rank_info.pp_rank == 0 and rank_info.cp_rank == 0:
                    self.mp_rank_zero[dp_rank] = self.cluster.workers[i]

        generate_opt_level = pipeline_config.generate_opt_level
        num_return_sequences = actor_cluster.worker_config.generating_args.num_return_sequences
        is_num_return_sequences_expand = pipeline_config.is_num_return_sequences_expand

        if generate_opt_level == 0 and is_num_return_sequences_expand:
            logger.warning("is_num_return_sequences_expand=True and generate_opt_level may reduce performance.")

        data.meta_info["is_num_return_sequences_expand"] = is_num_return_sequences_expand
        data.meta_info["num_return_sequences"] = num_return_sequences

        generation_config = self.cluster.worker_config.generating_args.to_dict()
        generation_config["num_return_sequences"] = num_return_sequences
        if is_num_return_sequences_expand:
            generation_config["num_return_sequences"] = 1
        data.meta_info["generation_config"] = generation_config

        if generate_opt_level == 0:
            if is_num_return_sequences_expand:
                batch_size = data.batch.batch_size[0]
                output_batch_size = batch_size * num_return_sequences
                input_ids = data.batch["input_ids"]
                attention_mask = data.batch["attention_mask"]
                position_ids = data.batch["position_ids"]
                input_ids = input_ids.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                attention_mask = (
                    attention_mask.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                )
                if position_ids.dim() == 3:  # (bsz, 3, seqlen)
                    # qwen2vl mrope, maybe use a placeholder and let model generate position_ids
                    position_ids = (
                        position_ids.unsqueeze(1)
                        .repeat(1, num_return_sequences, 1, 1)
                        .view(output_batch_size, *position_ids.shape[-2:])
                    )
                else:
                    position_ids = (
                        position_ids.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                    )

                non_tensor_batch = dict(
                    (k, np.repeat(v, num_return_sequences)) for k, v in data.non_tensor_batch.items()
                )

                data = DataProto(
                    batch=TensorDict(
                        {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
                        batch_size=output_batch_size,
                    ),
                    non_tensor_batch=non_tensor_batch,
                    meta_info=data.meta_info,
                )
            output_in_dp = await asyncio.gather(*[ref.obj_ref for ref in self.cluster.generate(data=data, blocking=False)])
            return DataProto.concat(output_in_dp)
        elif generate_opt_level == 1:
            return await self.generate_opt_level_1(data, num_return_sequences, is_num_return_sequences_expand)
        else:
            raise NotImplementedError(f"not support generate_opt_level {generate_opt_level}")

    async def generate_opt_level_1(self, data: DataProto, num_return_sequences, is_num_return_sequences_expand):
        batch_size = data.batch.batch_size[0]
        progress_bar = tqdm(total=batch_size, desc="generate progress(prompt)", mininterval=int(batch_size * 0.1) + 1)
        self.load_balancer = LoadBalancer(self.mp_rank_zero, self.max_running_requests)

        is_offload_states = data.meta_info.get("is_offload_states", True)
        await asyncio.gather(*[ref.obj_ref for ref in self.cluster.load_states(blocking=False)])

        tasks = []
        for data_index in range(batch_size):
            request_data = collate_fn([data[data_index]])
            request_data_list = expand_requests(data=request_data,
                num_return_sequences=num_return_sequences, is_num_return_sequences_expand=is_num_return_sequences_expand)

            prompt_requests = []
            for req in request_data_list:
                lease = await self.load_balancer.acquire(1)
                async def _generate_reqeust(data: DataProto, lease):
                    with lease.lock(1) as dp_rank:
                        request_id = self.next_request_id()
                        data.meta_info["request_id"] = request_id
                        data.meta_info["generation_config"] = data.meta_info["generation_config"]
                        response = await self.cluster.workers[dp_rank].generate_request.remote(data=request_data)
                        return response
                prompt_requests.append(asyncio.create_task(_generate_reqeust(data=req, lease=lease)))

            async def gather_one_prompt(requests):
                """
                gather requests of one prompt
                """
                responses = await asyncio.gather(*requests)
                progress_bar.update(1)
                return responses
            tasks.append(asyncio.create_task(gather_one_prompt(requests=prompt_requests)))
        assert self.load_balancer.empty()
        response_list = await asyncio.gather(*tasks)

        if is_offload_states:
            await asyncio.gather(*[ref.obj_ref for ref in self.cluster.offload_states(blocking=False)])
        response_ids_list_of_list = []
        eos_token_id = None
        pad_token_id = None
        for responses in response_list:
            response_ids_list = []
            for response in responses:
                eos_token_id = response.meta_info["eos_token_id"]
                pad_token_id = response.meta_info["pad_token_id"]
                response_ids_list.extend(response.meta_info["output_token_ids"])
            assert len(response_ids_list) == num_return_sequences
            response_ids_list_of_list.extend(response_ids_list)

        response_ids_list_of_list = [torch.tensor(token_ids) for token_ids in response_ids_list_of_list]
        output_tensor = pad_sequence(response_ids_list_of_list, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=data.batch["input_ids"],
            output_ids=output_tensor,
            num_return_sequences=num_return_sequences,
        )
        output: DataProto = postprocess_generate(
            prompts=data,
            output=output_tensor,
            num_return_sequences=num_return_sequences,
            sequence_length=self.pipeline_config.sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        return output


class DynamicSamplingScheduler(RolloutMockMixin, Scheduler):
    def __init__(self, pipeline_config=None):
        super().__init__()
        self.pipeline_config = pipeline_config
        set_seed(seed=pipeline_config.seed)

        self.sequence_length = pipeline_config.sequence_length

        self.actor_cluster = None
        self.mp_rank_zero = {}

        self.reward_clusters = None
        self.reward_worker_iters = None

        self.dataset = None
        self.indices = []
        self.dataset_epoch = 0
        self.dataset_iter = None
        self.dataset_iter_count = 0

        self.collect_fn_cls = None
        self.collect_fn_kwargs = None
        self.collect_fn = None
        self.tokenizer = None
        self.processor = None

        self.async_sending_task = None
        self.replay_buffer = None
        self.load_balancer = None
        self.running_requests = None
        self.running_tasks = None

        # metrics of a step
        self.generate_timer = None
        self.reward_timer = None

        # meta_info is reassigned every step
        self.meta_info = None

        udrl_cls = safe_import_class(pipeline_config.user_defined_rollout_loop_cls)
        assert udrl_cls
        self.udrl = udrl_cls()

        self.reward_scheduler = RewardScheduler()

    async def set_scheduler(
        self,
        actor_cluster: Union[Any, Cluster],
        reward_clusters: Dict[str, Union[Any, Cluster]],
        dataset: Dataset,
        collect_fn_cls,
        collect_fn_kwargs,
        state: Dict[str, Any] = None,
        is_val: bool = False,
    ):
        """
        GenerateScheduler可以由多个实例，不再局限于单例
        """
        self.is_val = is_val
        if self.is_val:
            self.sequence_length = self.pipeline_config.val_sequence_length
            logger.info(f"validation generate scheduler sequence_length is: {self.sequence_length}")
        else:
            logger.info(f"training generate scheduler sequence_length is: {self.sequence_length}")

        # Initialize rollout mock mechanism from mixin (after is_val is set)
        self._init_rollout_mock()

        self.actor_cluster = actor_cluster
        dp_ranks: List[int] = [rank_info.dp_rank for rank_info in self.actor_cluster.worker_rank_info]
        for i, dp_rank in enumerate(dp_ranks):
            rank_info = self.actor_cluster.get_rank_info(rank=i)
            if rank_info.tp_rank == 0 and rank_info.pp_rank == 0 and rank_info.cp_rank == 0:
                self.mp_rank_zero[dp_rank] = self.actor_cluster.workers[i]

        self.reward_clusters = reward_clusters
        self.reward_worker_iters = {}
        for domain, cluster in reward_clusters.items():
            self.reward_worker_iters[domain] = itertools.cycle(cluster.workers)

        self.generate_timer = {domain: DurationTracker() for domain in reward_clusters.keys()}
        self.reward_timer = {domain: DurationTracker() for domain in reward_clusters.keys()}

        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        # TODO: (async training) test resume
        if state is not None and state.get("dataset_iter_count", 0) > 0:
            for _ in range(state["dataset_iter_count"]):
                self.get_next_dataset_item()

        self.collect_fn_cls = collect_fn_cls
        self.collect_fn_kwargs = collect_fn_kwargs
        self.tokenizer = default_tokenizer_provider(model_args=self.actor_cluster.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.actor_cluster.worker_config.model_args)
        if "processor" in [f.name for f in fields(collect_fn_cls)]:
            collect_fn_kwargs["processor"] = self.processor
        self.collect_fn = self.collect_fn_cls(tokenizer=self.tokenizer, **self.collect_fn_kwargs)

        # Dynamic filter is supported no matter whether is_use_additional_prompts,
        # is_use_additional_prompts is required when using dynamic num_return_sequences.
        self.replay_buffer = ReplayBuffer(
            async_generation_ratio=self.pipeline_config.async_generation_ratio if not is_val else 0,
            is_use_additional_prompts=self.pipeline_config.is_use_additional_prompts if not is_val else False,
            max_additional_running_prompts=self.pipeline_config.max_additional_running_prompts if not is_val else 0,
        )
        self.load_balancer = LoadBalancer(self.mp_rank_zero, self.pipeline_config.max_running_requests)
        # dp_rank -> prompt_id -> request_ids
        self.running_requests: Dict[int, Dict[int, Set[str]]] = {dp_rank: defaultdict(set) for dp_rank in self.mp_rank_zero.keys()}
        self.running_tasks: Dict[int, asyncio.Task] = {}

        # async_sending_task is paused at start. But can not call self.pause_sampling directly here,
        # because ActorInfer.strategy has not been initialized yet and is not ready to serve abort_requests rpc.
        self.load_balancer.suspend()

        # async_sending_task coroutine will last during the whole training process, only stop at shutdown or exception.
        # Because we do not need to pause all running prompts but only suspend generate requests, so that reward requests
        # still can run during model update.
        self.async_sending_task = asyncio.create_task(self.sending_request())

    async def abort_running_requests(self):
        dp_requests = {}
        for dp_rank, prompt_requests in self.running_requests.items():
            dp_requests[dp_rank] = []
            for request_ids in prompt_requests.values():
                dp_requests[dp_rank].extend(request_ids)
        await asyncio.gather(
            *[
                self.actor_cluster.workers[dp_rank].abort_requests.remote(request_ids)
                for dp_rank, request_ids in dp_requests.items()
            ]
        )

    def gc(self):
        aborted_prompts = self.replay_buffer.gc()
        for prompt_id in aborted_prompts:
            if task := self.running_tasks.get(prompt_id, None):
                task.cancel()

    async def pause_sampling(self):
        self.load_balancer.suspend()
        self.gc()
        await self.abort_running_requests()
        await self.load_balancer.wait_complete()
        logger.info(f"sampling paused, replay_buffer info: {self.replay_buffer.info()}")

    async def shutdown(self):
        self.replay_buffer.shutdown()
        self.load_balancer.resume()
        self.gc()
        await self.abort_running_requests()
        await self.load_balancer.wait_complete()
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

        assert self.load_balancer.empty(), f"worker state: {self.load_balancer.workers}"
        assert all(len(requests) == 0 for prompt_requests in self.running_requests.values() for requests in prompt_requests.values())
        # Notice: self.replay_buffer.running_prompts may not be 0 because
        # pause_sampling only pause generate request but not reward request.

        self.replay_buffer.advance_step(step=global_step, batch_size=batch_size)
        logger.info(f"start sampling, {global_step=} {batch_size=}, {self.replay_buffer.info()}")
        self.load_balancer.resume()

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
        batch.meta_info["metrics"] = {
            f"scheduler/collect_query_count": query_use_count,
            f"scheduler/query_use_count": query_use_count,
            f"scheduler/off_policy_ratio": data_off_policy_step,
        }

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
        """
        See documentation of ReplyBuffer for recommended usage of ReplayBuffer and LoadBalancer.
        """
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
            await self.abort_running_requests()
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
    details of DynamicSamplingScheduler, LoadBalancer and ReplayBuffer from user.
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
        self._lease: LoadBalancer.Lease = None
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

        # All reuqest of the same prompt are scheduled to the same worker.
        # sample_params.n will take n credits rather than 1.
        # LoadBalancer.acquire will block until can send new request to actor infer.
        # Current implementation rely on the assumption that returned dp_rank is stable.
        self._lease = await self._scheduler.load_balancer.acquire(credit=max_concurrency)

        # Assume sampling_start_step of all samples of this prompt are the same, however
        # the real sampling_start_step can be different from self.sampling_start_step.
        try:
            sampling_start_step = await self._scheduler.replay_buffer.begin(prompt_id=self.prompt_id)
        except:
            self._lease.clear()
            raise
        self.sampling_start_step = sampling_start_step

        try:
            yield
        except:
            self._lease.clear()
            raise
        finally:
            assert (
                self.prompt_id not in self._scheduler.running_requests[self._lease._dp_rank] or
                len(self._scheduler.running_requests[self._lease._dp_rank][self.prompt_id]) == 0
            ), f"User should gather all running requests: {self._scheduler.running_requests[self._lease._dp_rank][self.prompt_id]=}"
            self._scheduler.running_requests[self._lease._dp_rank].pop(self.prompt_id, None)
            self._in_do_generate_and_reward = False

    async def generate(
        self,
        req: DataProto,
        domain: str,
    ) -> DataProto:
        assert self._in_do_generate_and_reward
        async with self._lease.lock(samplen=req.meta_info["generation_config"]["num_return_sequences"]) as dp_rank:
            with self._scheduler.generate_timer[domain].track():
                request_id = self._scheduler.next_request_id()
                req.meta_info["request_id"] = request_id
                logger.debug(f"generate_and_reward: {self.prompt_id=} {request_id} generate_request")
                self._scheduler.running_requests[dp_rank][self.prompt_id].add(request_id)
                try:
                    # InferWorker.generate_request only return data with finish_reason=="abort" on abort
                    # but not raise asyncio.CancelledError. This try finally block may be not necessary.
                    data = await self._scheduler.actor_cluster.workers[dp_rank].generate_request.remote(req)
                    # TODO ray.cancel(ref) on asyncio.CancelledError
                finally:
                    self._scheduler.running_requests[dp_rank][self.prompt_id].remove(request_id)
                assert data is not None
                return data

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
        dp_rank = self._lease._dp_rank
        request_ids = list(self._scheduler.running_requests[dp_rank][self.prompt_id])
        await self._scheduler.actor_cluster.workers[dp_rank].abort_requests.remote(request_ids)


@ray.remote
class RequestScheduler:
    def __init__(self, infer_cluster, pipeline_config, resource_manager):
        self.infer_cluster = infer_cluster
        self.pipeline_config = pipeline_config
        self.resource_manager = resource_manager
        self.request_id = uuid.uuid4()
        self.request_counter = 0
        self.src_rank2_dp_rank = {}
        self.request_id_2_dp_rank = {}
        self.request_id_2_src_rank: Dict[str, int] = {}  # Reverse lookup for abort
        self.running_requests: List[set[str]] = [set() for _ in range(self.infer_cluster.world_size)]
        self.worker_iter = itertools.cycle(range(self.infer_cluster.world_size))

        self.need_suspend = False
        self.suspend_notifier = asyncio.Event()
        self.empty_notifier = asyncio.Event()

        # Active DP ranks for request routing
        self.active_dp_ranks: Set[int] = set(range(self.infer_cluster.world_size))  # All ranks initially active
        self.routing_lock = asyncio.Lock()  # Protect routing updates

    async def generate_one_request(self, data: DataProto):
        await self._check_suspend()

        src_rank = data.meta_info["src_rank"]
        # Atomic routing assignment under lock to prevent TOCTOU race with shrink/expand
        async with self.routing_lock:
            # Least-loaded dispatch
            if src_rank not in self.src_rank2_dp_rank:
                dp_rank = self._get_least_active_dp_rank()
                self.src_rank2_dp_rank[src_rank] = dp_rank

        dp_rank = self.src_rank2_dp_rank[src_rank]
        request_id = f"{self.request_id}_{self.request_counter}"
        self.request_counter += 1
        data.meta_info["request_id"] = request_id

        self.request_id_2_dp_rank[request_id] = dp_rank
        self.request_id_2_src_rank[request_id] = src_rank
        self.running_requests[dp_rank].add(request_id)

        try:
            response_data = await self.infer_cluster.workers[dp_rank].generate_request.remote(data=data)
        finally:
            self.running_requests[dp_rank].remove(request_id)
            self.empty_notifier.set()
            # Cleanup tracking (on both success and abort paths)
            self.request_id_2_src_rank.pop(request_id, None)

        assert response_data is not None

        if not is_report_data_finished(response_data):
            return None

        # postprocess_generate, input_ids, attention_mask, left pad
        eos_token_id = response_data.meta_info["eos_token_id"]
        pad_token_id = response_data.meta_info["pad_token_id"]
        output_token_ids = response_data.meta_info["output_token_ids"]
        output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]

        output_logprobs = response_data.meta_info.get("output_logprobs", None)

        output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=data.batch["input_ids"], output_ids=output_tensor, num_return_sequences=len(output_tokens)
        )
        output: DataProto = postprocess_generate(
            prompts=data,
            output=output_tensor,
            num_return_sequences=len(output_tokens),
            sequence_length=output_tensor.shape[-1],
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            pad_to_seq_len=data.meta_info.get("pad_to_seq_len", True),
            output_logprobs=output_logprobs,
        )
        request_repeat = data.repeat(repeat_times=len(output_tokens))
        output.non_tensor_batch = request_repeat.non_tensor_batch
        output.meta_info = request_repeat.meta_info
        return output

    async def abort_request(self):
        await asyncio.gather(*(
            self.infer_cluster.workers[dp_rank].abort_requests.remote(list(self.running_requests[dp_rank]))
            for dp_rank in range(self.infer_cluster.world_size)
            if self.running_requests[dp_rank]
        ))


    async def _check_suspend(self):
        while self.need_suspend:
            await self.suspend_notifier.wait()

    def empty(self):
        return sum([len(running_requests) for running_requests in self.running_requests]) == 0

    async def suspend(self):
        if self.need_suspend:
            return
        self.suspend_notifier.clear()
        self.need_suspend = True
        await self.abort_request()
        while not self.empty():
            self.empty_notifier.clear()
            await self.empty_notifier.wait()

    def resume(self):
        if not self.need_suspend:
            return
        self.need_suspend = False
        self.suspend_notifier.set()

    def _get_gpus_for_dp_rank(self, dp_rank: int) -> List[int]:
        """Map DP rank to GPU IDs using cluster's device info.

        Args:
            dp_rank: Data parallel rank index (0 to dp_size-1)

        Returns:
            List of GPU IDs used by this DP rank's workers

        Example:
            # Pure DP: rank == dp_rank
            # DP rank 0 uses GPUs [0], DP rank 1 uses GPUs [1], etc.
            gpus = self._get_gpus_for_dp_rank(dp_rank=0)
            # Returns: [0]
        """
        # In agentic pipeline (pure DP): rank == dp_rank, so directly access rank2devices
        devices_info = self.infer_cluster.rank2devices[dp_rank]

        # Extract GPU IDs: gpu_id = node_rank * num_gpus_per_node + gpu_rank
        gpu_ids = []
        for device in devices_info:
            num_gpus_per_node = self.resource_manager.gpu_per_node
            gpu_id = device["node_rank"] * num_gpus_per_node + device["gpu_rank"]
            gpu_ids.append(gpu_id)

        return sorted(set(gpu_ids))  # Remove duplicates and sort

    def _get_least_active_dp_rank(self) -> int:
        """Find DP rank with fewest assigned src_ranks (environments).

        Returns:
            DP rank with minimum src_rank count from src_rank2_dp_rank

        Raises:
            RuntimeError: If no active ranks

        Note:
            Counts unique src_ranks (environments) per worker, not in-flight requests.
            With sticky mapping, one src_rank generates multiple sequential requests.
        """
        candidate_ranks = list(self.active_dp_ranks)
        if not candidate_ranks:
            raise RuntimeError("No active DP ranks")
        # todo optimization: (yangpeng) not efficient, better to use counter for this
        # Count src_ranks per dp_rank
        src_rank_count = defaultdict(int)
        for src_rank, dp_rank in self.src_rank2_dp_rank.items():
            if dp_rank in self.active_dp_ranks:
                src_rank_count[dp_rank] += 1

        # Return dp_rank with minimum src_rank count
        return min(candidate_ranks, key=lambda r: src_rank_count[r])

    def _clear_src_rank_mappings(self, src_ranks: Set[int]) -> None:
        """Clear sticky mappings to allow re-routing on retry."""
        for src_rank in src_ranks:
            self.src_rank2_dp_rank.pop(src_rank, None)

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        """Abort requests on shrinking workers, clear mappings for natural re-dispatch.

        Args:
            shrink_dp_ranks: DP ranks to remove from active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            ValueError: If shrink_dp_ranks empty/invalid/duplicates
            RuntimeError: If timeout or operation fails
        """
        # VAL: VAL_NON_EMPTY, VAL_TYPE_CHECK, VAL_INT_RANGE, VAL_NO_DUPLICATES
        if not shrink_dp_ranks:
            raise ValueError("shrink_dp_ranks cannot be empty")

        for rank in shrink_dp_ranks:
            if not isinstance(rank, int):
                raise TypeError(f"Expected int, got {type(rank)}")
            if not (0 <= rank < self.infer_cluster.world_size):
                raise ValueError(f"rank {rank} out of range")

        if len(shrink_dp_ranks) != len(set(shrink_dp_ranks)):
            raise ValueError(f"Duplicates in shrink_dp_ranks")

        # P0: LOCK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._rebalance_on_shrink(shrink_dp_ranks),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("rebalance_on_shrink timed out after 30s")

    async def _rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        """Internal implementation of shrink rebalancing.

        PRE-CONDITION: routing_lock MUST be held by caller (shrink_workers).
        This method does NOT acquire the lock internally to avoid double-lock deadlock.

        Args:
            shrink_dp_ranks: DP ranks to remove from active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            RuntimeError: If shrink operation fails
        """
        keep_ranks = list(self.active_dp_ranks - set(shrink_dp_ranks))
        if not keep_ranks:
            raise ValueError("Cannot shrink to zero active ranks")

        old_active_ranks = self.active_dp_ranks.copy()
        self.active_dp_ranks = set(keep_ranks)

        try:
            total_aborted = 0
            abort_futures = []

            for dp_rank in shrink_dp_ranks:
                request_ids = list(self.running_requests[dp_rank])
                if not request_ids:
                    continue

                total_aborted += len(request_ids)

                abort_futures.append(
                self.infer_cluster.workers[dp_rank].abort_requests.remote(request_ids)
                )



            await asyncio.gather(*abort_futures)

            while True:
                remain = sum(len(self.running_requests[dp_rank]) for dp_rank in shrink_dp_ranks)
                if remain == 0:
                    break
                logger.info(f"Shrink: waiting for {len(shrink_dp_ranks)} workers {remain=} to finish abort")
                await asyncio.sleep(3)

            # Clear ALL mappings pointing to shrinking workers (not just in-flight)
            shrink_dp_ranks_set = set(shrink_dp_ranks)
            src_ranks_to_remap = set([
                src_rank for src_rank, dp_rank in self.src_rank2_dp_rank.items()
                if dp_rank in shrink_dp_ranks_set
            ])
            self._clear_src_rank_mappings(src_ranks_to_remap)

            logger.info(
                f"Shrink: aborted {total_aborted} requests, "
                f"cleared {len(src_ranks_to_remap)} mappings"
            )

            return {"aborted": total_aborted, "remapped": len(src_ranks_to_remap)}

        except Exception as e:
            self.active_dp_ranks = old_active_ranks
            raise RuntimeError(f"Shrink failed: {e}") from e

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        """Add workers and rebalance via src_rank-level abort.

        Args:
            expand_dp_ranks: DP ranks to add to active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            ValueError: If expand_dp_ranks invalid
            RuntimeError: If timeout or operation fails
        """
        # VAL: VAL_NON_EMPTY, VAL_TYPE_CHECK, VAL_INT_RANGE, VAL_NO_DUPLICATES
        if not expand_dp_ranks:
            raise ValueError("expand_dp_ranks cannot be empty")
        for rank in expand_dp_ranks:
            if not isinstance(rank, int):
                raise TypeError(f"Expected int, got {type(rank)}")
            if not (0 <= rank < self.infer_cluster.world_size):
                raise ValueError(f"rank {rank} out of range")
        if len(expand_dp_ranks) != len(set(expand_dp_ranks)):
            raise ValueError(f"Duplicates in expand_dp_ranks")

        # P0: LOCK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._rebalance_on_expand(expand_dp_ranks),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("rebalance_on_expand timed out after 30s")

    async def _rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        """Internal implementation of expand rebalancing.

        PRE-CONDITION: routing_lock MUST be held by caller (expand_workers).
        This method does NOT acquire the lock internally to avoid double-lock deadlock.

        Algorithm: Round-robin selection across old workers
        1. Calculate proportional src_ranks to abort: src_ranks_to_keep = ceil(total * old_count / new_count)
        2. Group existing src_ranks by dp_rank (only old workers)
        3. Round-robin iterate over old workers using cycle()
        4. Select one src_rank at a time until remaining_to_abort reaches 0
        5. Abort ALL requests from selected src_ranks
        6. Clear src_rank mappings for reallocation to new workers

        Implementation Notes:
        - Uses cycle() for infinite round-robin iteration over old workers
        - Check at line 1146 (if not dp_rank in old_active_dp_ranks) is redundant
          since dp_rank_to_src_ranks already contains only old workers, but kept as defensive guard
        - Loop terminates when remaining_to_abort <= 0 or all worker lists are exhausted
        - If all workers exhausted before reaching target, loop may cycle indefinitely
          (no explicit check for empty state, but pop(0) will eventually empty all lists)

        Args:
            expand_dp_ranks: DP ranks to add to active set (already validated)

        Returns:
            {"aborted": count, "remapped": count} - count of src_ranks aborted/remapped

        Preconditions:
            - routing_lock MUST be held by caller
            - expand_dp_ranks validated (non-empty, int, in range, no duplicates)

        Postconditions:
            - active_dp_ranks updated with expand_dp_ranks
            - Selected src_ranks aborted and removed from mappings
            - Requests from aborted src_ranks reported as is_abort=True
        """
        # Calculate counts before updating active_dp_ranks
        old_dp_count = len(self.active_dp_ranks)
        old_active_dp_ranks = self.active_dp_ranks.copy()

        self.active_dp_ranks.update(expand_dp_ranks)
        new_dp_count = len(self.active_dp_ranks)

        total_src_ranks = len(self.src_rank2_dp_rank)
        if total_src_ranks == 0:
            return {"aborted": 0, "remapped": 0}

        # Proportional calculation
        src_ranks_to_keep = math.ceil(int(total_src_ranks * old_dp_count / new_dp_count))
        src_ranks_to_abort = total_src_ranks - src_ranks_to_keep

        if src_ranks_to_abort <= 0:
            logger.info("Expand: no rebalancing needed (src_ranks_to_abort <= 0)")
            return {"aborted": 0, "remapped": 0}

        # Group src_ranks by dp_rank (old workers only)
        dp_rank_to_src_ranks = defaultdict(list)
        for src_rank, dp_rank in self.src_rank2_dp_rank.items():
            if dp_rank in old_active_dp_ranks:
                dp_rank_to_src_ranks[dp_rank].append(src_rank)

        # Round-robin selection: iterate over old workers and select one src_rank at a time
        # todo optimization:(yangpeng) take uneven dp load into consideration and do dynamic load balancing, not just RR
        selected_src_ranks = []
        remaining_to_abort = src_ranks_to_abort
        for dp_rank in cycle(dp_rank_to_src_ranks.keys()):
            if not dp_rank in old_active_dp_ranks:
                continue

            if remaining_to_abort <= 0:
                break

            src_ranks_on_worker = dp_rank_to_src_ranks.get(dp_rank, [])
            if not src_ranks_on_worker:
                continue
            selected_src_ranks.append(src_ranks_on_worker.pop(0))

            remaining_to_abort -= 1

        # Remove from mapping and group by dp_rank for abort
        abort_by_dp_rank = defaultdict(list)
        for src_rank in selected_src_ranks:
            dp_rank = self.src_rank2_dp_rank.pop(src_rank)

            # Find request_id(s) for this src_rank
            for request_id, sr in self.request_id_2_src_rank.items():
                if sr == src_rank:
                    abort_by_dp_rank[dp_rank].append(request_id)

        # Send batched ABORT commands
        abort_futures = []
        total_aborted = 0
        for dp_rank, request_ids in abort_by_dp_rank.items():
            if not request_ids:
                continue

            total_aborted += len(request_ids)
            abort_futures.append(
                self.infer_cluster.workers[dp_rank].abort_requests.remote(request_ids)
            )


        await asyncio.gather(*abort_futures)

        logger.info(
            f"Expand: aborted {len(selected_src_ranks)} src_ranks, "
            f"cleared {len(selected_src_ranks)} mappings "
            f"(proportional: {old_dp_count}/{new_dp_count})"
        )

        return {"aborted": len(selected_src_ranks), "remapped": len(selected_src_ranks)}

    def _validate_target_gpus(self, target_gpus: List[int], mode: str) -> None:
        """Validate target_gpus input for shrink/expand operations.

        Args:
            target_gpus: List of GPU IDs to free (shrink) or restore (expand)
            mode: Operation mode ("shrink" or "expand")

        Raises:
            ValueError: If target_gpus is empty, has duplicates, or mode is invalid

        Example:
            self._validate_target_gpus([4, 5, 6, 7], mode="shrink")
            # Validates successfully

            self._validate_target_gpus([], mode="shrink")
            # Raises: ValueError("[shrink] target_gpus cannot be empty")

            self._validate_target_gpus([4, 4, 5], mode="expand")
            # Raises: ValueError("[expand] target_gpus has duplicates: [4, 4, 5]")
        """
        # VAL: VAL_NON_EMPTY
        if not target_gpus:
            raise ValueError(f"[{mode}] target_gpus cannot be empty")

        # VAL: VAL_NO_DUPLICATES
        if len(target_gpus) != len(set(target_gpus)):
            raise ValueError(f"[{mode}] target_gpus has duplicates: {target_gpus}")

        if mode not in ("shrink", "expand"):
            raise ValueError(f"Invalid mode: {mode}")

    def _validate_calculated_ranks(self, ranks: List[int], mode: str) -> None:
        """Validate calculated DP ranks against current active_dp_ranks state.

        Args:
            ranks: List of DP ranks calculated from target_gpus
            mode: Operation mode ("shrink" or "expand")

        Raises:
            ValueError: If ranks is empty, contains out-of-range values,
                       or violates state consistency (shrink: must be active,
                       expand: must be inactive)

        Example:
            # Shrink validation
            self.active_dp_ranks = {0, 1, 2, 3}
            self._validate_calculated_ranks([2, 3], mode="shrink")
            # Validates successfully (ranks 2, 3 are active)

            self._validate_calculated_ranks([4], mode="shrink")
            # Raises: ValueError("[shrink] DP rank 4 not active")

            # Expand validation
            self.active_dp_ranks = {0, 1}
            self._validate_calculated_ranks([2, 3], mode="expand")
            # Validates successfully (ranks 2, 3 are inactive)

            self._validate_calculated_ranks([0], mode="expand")
            # Raises: ValueError("[expand] DP rank 0 already active")
        """
        # VAL: VAL_NON_EMPTY
        if not ranks:
            raise ValueError(f"[{mode}] Calculated ranks list is empty")

        # VAL: VAL_INT_RANGE
        for dp_rank in ranks:
            if not (0 <= dp_rank < self.infer_cluster.world_size):
                raise ValueError(f"[{mode}] DP rank {dp_rank} out of range [0, {self.infer_cluster.world_size})")

        # AST: State consistency

        for dp_rank in ranks:
            if dp_rank not in self.active_dp_ranks:
                raise ValueError(f"DP rank {dp_rank} not active {mode=}")

    async def shrink_workers(self, target_gpus: List[int]) -> Dict[str, Any]:
        """Complete atomic shrink operation: validate → rebalance → offload → update routing.

        Orchestrates the full worker shrink process:
        1. Validates target_gpus input
        2. Calculates DP ranks to offload based on GPU overlap
        3. Validates calculated ranks against active state
        4. Atomically (under routing_lock):
           - Rebalances routing (aborts requests on shrinking workers)
           - Offloads model states from shrinking workers
        5. Returns metrics for monitoring

        Args:
            target_gpus: GPU IDs to free (e.g., [4, 5, 6, 7] to free second half of 8 GPUs)

        Returns:
            Metrics dict containing:
                - "aborted": Number of requests aborted during rebalancing
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "shrink_duration_ms": Total operation time in milliseconds
                - "offload_ranks": List of DP ranks that were offloaded

        Raises:
            ValueError: If target_gpus invalid (empty, duplicates) or
                       calculated ranks invalid (not active, out of range)
            RuntimeError: If rebalance or offload operations fail

        Example:
            # Shrink to free GPUs [4, 5, 6, 7] (second half of 8-GPU setup)
            result = await scheduler.shrink_workers([4, 5, 6, 7])
            # Returns: {"aborted": 10, "remapped": 5, "shrink_duration_ms": 2340.5, "offload_ranks": [2, 3]}

        Side Effects:
            - Updates active_dp_ranks (removes offload_ranks)
            - Aborts in-flight requests on shrinking workers
            - Clears src_rank mappings for remapped environments
            - Offloads model states from shrinking workers to CPU
        """
        start_time = time.time()

        # VAL: VAL_NON_EMPTY, VAL_NO_DUPLICATES
        self._validate_target_gpus(target_gpus, mode="shrink")
        # Calculate DP ranks to offload
        target_gpus = set(target_gpus)
        offload_ranks = [dp for dp in range(self.infer_cluster.world_size)
                         if set(self._get_gpus_for_dp_rank(dp)).intersection(target_gpus)]

        # VAL: VAL_NON_EMPTY, state consistency check
        self._validate_calculated_ranks(offload_ranks, mode="shrink")

        # Atomic operation under routing_lock
        async with self.routing_lock:
            # Rebalance (abort + update active_dp_ranks)
            result = await self.rebalance_on_shrink(offload_ranks)
        # release the lock before blocking offload so that active dp rank can work immediately
        # Offload states from target workers
        offload_refs = self.infer_cluster.offload_states_partial(offload_ranks, blocking=False)
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in offload_refs])

        return {**result, "shrink_duration_ms": (time.time() - start_time) * 1000,
                "offload_ranks": offload_ranks}

    async def expand_workers(self, target_gpus: List[int], skip_load: bool = False) -> Dict[str, Any]:
        """Complete atomic expand operation: validate → load → rebalance → update routing.

        Orchestrates the full worker expand process:
        1. Validates target_gpus input
        2. Calculates DP ranks to restore based on GPU overlap
        3. Validates calculated ranks against active state (skip if skip_load=True)
        4. Atomically (under routing_lock):
           - Loads model states on expanding workers (skip if skip_load=True)
           - Rebalances routing (proportionally redistributes requests)
        5. Returns metrics for monitoring

        Args:
            target_gpus: GPU IDs to restore (e.g., [4, 5, 6, 7] to restore second half of 8 GPUs)
            skip_load: If True, skip model loading and validation (use when model_update already loaded states).
                      This only updates active_dp_ranks to restore routing state without re-loading models.

        Returns:
            Metrics dict containing:
                - "aborted": Number of requests aborted during rebalancing (proportional redistribution)
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "expand_duration_ms": Total operation time in milliseconds
                - "load_ranks": List of DP ranks that were restored

        Raises:
            ValueError: If target_gpus invalid (empty, duplicates) or
                       calculated ranks invalid (already active, out of range)
            RuntimeError: If load or rebalance operations fail

        Example:
            # Expand to restore GPUs [4, 5, 6, 7] (second half of 8-GPU setup)
            result = await scheduler.expand_workers([4, 5, 6, 7])
            # Returns: {"aborted": 3, "remapped": 3, "expand_duration_ms": 1850.2, "load_ranks": [2, 3]}

            # After model_update already loaded states to all GPUs, just restore routing:
            result = await scheduler.expand_workers([4, 5, 6, 7], skip_load=True)

        Side Effects:
            - Updates active_dp_ranks (adds load_ranks)
            - Loads model states from CPU to expanding workers (unless skip_load=True)
            - Aborts some requests from old workers for proportional rebalancing
            - Clears src_rank mappings for rebalanced environments (will route to new workers)
        """
        start_time = time.time()

        # VAL: VAL_NON_EMPTY, VAL_NO_DUPLICATES
        self._validate_target_gpus(target_gpus, mode="expand")

        # Calculate DP ranks to restore
        target_gpus = set(target_gpus)
        load_ranks = [dp for dp in range(self.infer_cluster.world_size)
                      if set(self._get_gpus_for_dp_rank(dp)).issubset(target_gpus)]

        # VAL: VAL_NON_EMPTY, state consistency check
        # Skip validation when skip_load=True because ranks may already be "active" in cluster
        # (model states loaded by model_update) but not tracked in active_dp_ranks yet
        if not skip_load:
            self._validate_calculated_ranks(load_ranks, mode="expand")
            load_refs = self.infer_cluster.load_states_partial(load_ranks, blocking=False)
            await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in load_refs])

        # Atomic operation under routing_lock
        async with self.routing_lock:

            # Rebalance (update active_dp_ranks + conditional abort)
            result = await self.rebalance_on_expand(load_ranks)

        return {**result, "expand_duration_ms": (time.time() - start_time) * 1000,
                "load_ranks": load_ranks}
