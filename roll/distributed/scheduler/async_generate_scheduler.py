import copy
import enum
import itertools
import math
import queue
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

import ray
import torch
from datasets import Dataset
from ray import ObjectRef
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import set_seed

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import GlobalCounter
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import (
    GenerateRequestType,
    concatenate_input_and_output,
    postprocess_generate,
)
from roll.utils.logging import get_logger


logger = get_logger()


class ExpStatus(enum.Enum):
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    FINISHED = enum.auto()
    DELETED = enum.auto()


def is_report_data_finished(data: DataProto) -> bool:
    finish_reasons = data.meta_info.get("finish_reasons", [])
    # take no finish reason as finished
    return not any([finish_reason is None for finish_reason in finish_reasons])


@dataclass
class ExperienceItem:
    request_id: str
    prompt_id: int
    domain: str = "default"
    sampling_start_step: Optional[int] = None
    running_dp_rank: Optional[int] = None  # should be none when not running
    data: Optional[DataProto] = None
    status: ExpStatus = ExpStatus.RUNNING


@dataclass
class ItemsGroup:
    # items with the same starting step
    # item status index
    start_step: int
    by_request_id: Dict[str, ExperienceItem] = field(default_factory=dict)
    request_ids_by_prompt_id: Dict[int, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    running_request_ids: Set[str] = field(default_factory=set)
    pause_request_ids: Set[str] = field(default_factory=set)
    finished_ids_by_prompt_id: Dict[int, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # all requests for this prompt_id are finished, manually marked
    finished_prompt_ids: Set[int] = field(default_factory=set)

    def info(self):
        return (f"ItemsGroup {self.start_step}: {len(self.by_request_id)=}"
                f"{len(self.request_ids_by_prompt_id)=} {len(self.running_request_ids)=}"
                f"{len(self.pause_request_ids)=} {len(self.finished_prompt_ids)=}"
                f"{len(self.finished_ids_by_prompt_id)=}")

    def add_request_item(self, request_item: ExperienceItem):
        self.by_request_id[request_item.request_id] = request_item
        self.request_ids_by_prompt_id[request_item.prompt_id].add(request_item.request_id)
        request_item.status = ExpStatus.RUNNING
        self.running_request_ids.add(request_item.request_id)
        self.pause_request_ids.discard(request_item.request_id)

    def reset_status(self):
        for request_id in list(self.running_request_ids):
            item = self.by_request_id.get(request_id)
            if item:
                item.status = ExpStatus.PAUSED
                self.pause_request_ids.add(request_id)
        self.running_request_ids = set()

    def get_prompt_ids(self) -> Set[int]:
        return set(self.request_ids_by_prompt_id.keys())

    def pop_item(self, prompt_id: int) -> Optional[ExperienceItem]:
        request_id = self.request_ids_by_prompt_id[prompt_id].pop()
        return self.by_request_id.pop(request_id)

    def mark_prompt_id_finished(self, prompt_id: int, num_return_sequences: int):
        finished_ids = self.finished_ids_by_prompt_id.get(prompt_id, set())

        self.finished_prompt_ids.add(prompt_id)

        assert (
            sum(self.by_request_id.get(request_id).data.batch.batch_size[0] for request_id in finished_ids)
            >= num_return_sequences
        ), f"prompt_id {prompt_id} finished_ids {len(finished_ids)} num_return_sequences {num_return_sequences}"
        all_request_ids = self.request_ids_by_prompt_id[prompt_id]

        # remove extra request_ids and return them
        extra_request_ids = all_request_ids - finished_ids
        extra_request_ids = extra_request_ids.union(finished_ids - set(list(finished_ids)[:num_return_sequences]))
        extra_items = []
        for request_id in extra_request_ids:
            item = self.by_request_id.get(request_id)
            extra_items.append(item)
            item.status = ExpStatus.DELETED
            self.request_ids_by_prompt_id[prompt_id].discard(request_id)

            self.running_request_ids.discard(request_id)
            self.pause_request_ids.discard(request_id)
            self.finished_ids_by_prompt_id[prompt_id].discard(request_id)
        return extra_items

    def pop_prompt_id(self, prompt_id: int):
        running_items, finished_items = [], []
        request_ids = self.request_ids_by_prompt_id.pop(prompt_id, set())
        for request_id in request_ids:
            item = self.by_request_id.get(request_id)
            if item is None:
                continue
            if item.status == ExpStatus.FINISHED:
                finished_items.append(item)
            elif item.status == ExpStatus.RUNNING:
                running_items.append(item)
            item.status = ExpStatus.DELETED
            self.pause_request_ids.discard(request_id)
            self.running_request_ids.discard(request_id)

        self.finished_prompt_ids.discard(prompt_id)
        self.finished_ids_by_prompt_id.pop(prompt_id, None)
        return running_items, finished_items


@dataclass
class ReplayBuffer:
    groups: Dict[int, ItemsGroup] = field(default_factory=dict)
    prompt_id_to_start_step: Dict[int, int] = field(default_factory=dict)

    def info(self) -> str:
        return (
            f"ReplayBuffer: "
            f"by_request_id={sum(len(g.by_request_id) for g in self.groups.values())}, "
            f"request_ids_by_prompt_id={sum(len(g.request_ids_by_prompt_id) for g in self.groups.values())}, "
            f"running_request_ids={sum(len(g.running_request_ids) for g in self.groups.values())}, "
            f"pause_request_ids={sum(len(g.pause_request_ids) for g in self.groups.values())}, "
            f"finished_prompt_ids_by_start_step={sum(len(g.finished_prompt_ids) for g in self.groups.values())}, "
            f"finished_ids_by_prompt_id={sum(len(g.finished_ids_by_prompt_id) for g in self.groups.values())}"
        )

    def add_request_item(self, request_item: ExperienceItem):
        start_step = request_item.sampling_start_step
        group = self.groups.setdefault(start_step, ItemsGroup(start_step=start_step))
        self.prompt_id_to_start_step[request_item.prompt_id] = start_step
        group.add_request_item(request_item)

    def get_item(self, request_id: str) -> ExperienceItem:
        for group in self.groups.values():
            if request_id in group.by_request_id:
                return group.by_request_id[request_id]
        raise ValueError(f"request_id {request_id} not found")

    def prompt_num(self) -> int:
        return sum(len(group.request_ids_by_prompt_id) for group in self.groups.values())

    def running_request_num(self) -> int:
        return sum(len(group.running_request_ids) for group in self.groups.values())

    def _get_group_by_prompt_id(self, prompt_id: int) -> Optional[ItemsGroup]:
        start_step = self.prompt_id_to_start_step.get(prompt_id)
        if start_step is None:
            return None
        return self.groups.get(start_step)

    def report_item(self, data: DataProto, is_finished: bool = True):
        request_id = data.meta_info["request_id"]
        item = self.get_item(request_id)
        if item is None:
            raise ValueError(f"request_id {request_id} not found")

        group = self.groups[item.sampling_start_step]
        group.running_request_ids.discard(request_id)

        item.data = data
        item.running_dp_rank = None
        item.status = ExpStatus.FINISHED if is_finished else ExpStatus.PAUSED
        if is_finished:
            group.finished_ids_by_prompt_id[item.prompt_id].add(request_id)
        else:
            group.pause_request_ids.add(request_id)

    def get_finished_data_by_prompt_id(self, prompt_id: int) -> List[DataProto]:
        """
        get all finished data for a prompt_id
        """
        group = self._get_group_by_prompt_id(prompt_id)
        request_ids = group.finished_ids_by_prompt_id.get(prompt_id, set())
        return [group.by_request_id[request_id].data for request_id in request_ids]

    def mark_prompt_id_finished(self, prompt_id: int, num_return_sequences: int) -> list[ExperienceItem]:
        """
        mark a prompt_id as finished, pop extra items, to ensure only num_return_sequences items in finished_ids_by_prompt_id
        """
        group = self._get_group_by_prompt_id(prompt_id)
        return group.mark_prompt_id_finished(prompt_id, num_return_sequences)

    def pop_paused_item(self) -> Optional[ExperienceItem]:
        """
        pop a paused item, used to resume paused requests
        """
        for start_step in sorted(self.groups.keys()):
            group = self.groups[start_step]
            if len(group.pause_request_ids) > 0:
                request_id = group.pause_request_ids.pop()
                item = group.by_request_id.get(request_id)
                item.status = ExpStatus.RUNNING
                group.running_request_ids.add(request_id)
                return item
        return None

    def get_enough_finished_prompt_ids(
        self, total_prompt_num: int, min_step: Optional[int] = None, min_step_prompt_num: Optional[int] = None
    ) -> Optional[List[int]]:
        # if not enough return None
        if min_step is None:
            assert min_step_prompt_num is None
            if not (
                sum(len(group.finished_prompt_ids) for group in self.groups.values()) >= total_prompt_num
            ):
                return None
            return list(set().union(*(group.finished_prompt_ids for group in self.groups.values())))[: total_prompt_num]

        min_step_group = self.groups.get(min_step, None)
        if min_step_group is None:
            return None
        min_step_prompt_num = min(min_step_prompt_num, len(min_step_group.get_prompt_ids()))
        required_prompt_ids = list(min_step_group.finished_prompt_ids)[:min_step_prompt_num]
        if len(required_prompt_ids) < min_step_prompt_num:
            return None

        for start_step in sorted(self.groups.keys()):
            if start_step <= min_step:
                continue
            group = self.groups[start_step]
            wanted_prompt_num = min(total_prompt_num - len(required_prompt_ids), len(group.get_prompt_ids()))
            if wanted_prompt_num <= 0 or len(group.finished_prompt_ids) < wanted_prompt_num:
                break
            required_prompt_ids.extend(list(group.finished_prompt_ids)[:wanted_prompt_num])

        if len(required_prompt_ids) < total_prompt_num:
            return None

        assert len(required_prompt_ids) == total_prompt_num, f"{len(required_prompt_ids)=} {total_prompt_num=}"
        return required_prompt_ids

    def pop_finished_items_by_prompt_ids(self, prompt_ids: List[int], num_return_sequences: int) -> List[ExperienceItem]:
        items = []
        for prompt_id in prompt_ids:
            _, finished_items = self.pop_prompt_id(prompt_id)
            assert len(finished_items) == num_return_sequences
            items.extend(finished_items)
        return items

    def reset_status(self, min_step: Optional[int] = None, required_prompt_num: Optional[int] = None):
        # clear not needed prompt_ids, clear all when min_step is None
        for start_step in list(self.groups.keys()):
            group = self.groups[start_step]
            if min_step is None or start_step < min_step:
                group = self.groups.pop(start_step)
                del_prompt_ids = group.get_prompt_ids()
            elif start_step == min_step:
                del_prompt_ids = group.get_prompt_ids()
                del_num = len(del_prompt_ids) - required_prompt_num
                if del_num <= 0:
                    continue
                logger.info(f"randomly delete {del_num} prompt_ids from step: {start_step}")
                del_prompt_ids = random.sample(list(del_prompt_ids), del_num)
            else:  # start_step > min_step
                continue

            for prompt_id in del_prompt_ids:
                self.pop_prompt_id(prompt_id)

        for group in self.groups.values():
            group.reset_status()

    def is_prompt_id_finished(self, prompt_id: int) -> bool:
        group = self._get_group_by_prompt_id(prompt_id)
        return prompt_id in group.finished_prompt_ids

    def pop_prompt_id(self, prompt_id: int):
        group = self._get_group_by_prompt_id(prompt_id)
        self.prompt_id_to_start_step.pop(prompt_id, None)
        if group is None:
            return [], []
        return group.pop_prompt_id(prompt_id)


@ray.remote(concurrency_groups={"single_thread": 1, "multi_thread": 256})
class AsyncDynamicSamplingScheduler:
    def __init__(self, pipeline_config=None):
        self.pipeline_config = pipeline_config
        set_seed(seed=pipeline_config.seed)
        self.progress_bar: Optional[tqdm] = None
        self.request_counter = None
        self.mp_rank_zero = {}
        self.replay_buffer = ReplayBuffer()
        self.prompt_id_counter = itertools.count()
        self.response_batch_size: Optional[int] = None
        self.lock = threading.Lock()
        self.last_alive_check = time.time()
        self.dataset_iter_count = 0
        self.exception_queue = queue.Queue()
        self.dataset_epoch = 0

        self.alive_check_interval = self.pipeline_config.alive_check_interval
        self.max_additional_running_prompts = self.pipeline_config.max_additional_running_prompts
        self.is_use_additional_prompts = self.pipeline_config.is_use_additional_prompts

        self.actor_cluster = None
        self.reward_clusters = None
        self.reward_worker_iters = None
        self.dataset = None
        self.indices = []
        self.batch_size = None
        self.dataset_iter = None
        self.collect_fn_cls = None
        self.collect_fn_kwargs = None
        self.collect_fn = None
        self.tokenizer = None
        self.response_callback_fn = None
        self.generation_config = None

        self.async_sending_thread = None
        self.stop_sending_requests_event = threading.Event()

        self.global_step = 0
        self.init_global_step = None
        self.pre_send_dp_rank = 0
        self.filter_prompt_num = 0

    def set_scheduler(
        self,
        actor_cluster: Union[Any, Cluster],
        reward_clusters: Dict[str, Union[Any, Cluster]],
        dataset: Dataset,
        collect_fn_cls,
        collect_fn_kwargs,
        response_filter_fn=None,
        query_filter_fn=None,
        response_callback_fn=None,
        state: Dict[str, Any] = None,
        is_val: bool = False,
    ):
        self.is_val = is_val
        self.actor_cluster = actor_cluster
        self.reward_clusters = reward_clusters
        self.reward_worker_iters = {}
        for domain, cluster in reward_clusters.items():
            self.reward_worker_iters[domain] = itertools.cycle(cluster.workers)

        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        # TODO: test resume
        if state is not None and state.get("dataset_iter_count", 0) > 0:
            for _ in range(state["dataset_iter_count"]):
                self.get_next_dataset_item()

        self.collect_fn_cls = collect_fn_cls
        self.collect_fn_kwargs = collect_fn_kwargs
        self.tokenizer = default_tokenizer_provider(model_args=self.actor_cluster.worker_config.model_args)
        self.collect_fn = self.collect_fn_cls(tokenizer=self.tokenizer, **self.collect_fn_kwargs)
        self.response_callback_fn = response_callback_fn
        dp_ranks: List[int] = [rank_info.dp_rank for rank_info in self.actor_cluster.worker_rank_info]
        for i, dp_rank in enumerate(dp_ranks):
            rank_info = self.actor_cluster.get_rank_info(rank=i)
            if rank_info.tp_rank == 0 and rank_info.pp_rank == 0 and rank_info.cp_rank == 0:
                self.mp_rank_zero[dp_rank] = self.actor_cluster.workers[i]

        # TODO: support response_filter_fn
        if self.is_use_additional_prompts:
            self.query_filter_fn = query_filter_fn
        else:
            self.query_filter_fn = lambda data_list, config: True
            logger.info("use_additional_prompts is False, disable query and response filtering.")

        self.cluster_max_running_requests = self.pipeline_config.max_running_requests * self.actor_cluster.dp_size
        self.request_counter = GlobalCounter.options(
            name="DynamicSchedulerRequestCounter",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

    def reset_status(self):
        self.exception_queue = queue.Queue()
        min_start_step, required_prompt_num = self._get_min_step_and_prompt_num(for_keep=True)
        with self.lock:
            self.replay_buffer.reset_status(min_start_step, required_prompt_num)

        bar_name = "-".join(self.reward_clusters.keys())
        self.progress_bar = tqdm(
            total=self.batch_size,
            desc=f"{bar_name} generate progress(prompt)",
            mininterval=int(self.batch_size * 0.1) + 1,
        )
        self.stop_sending_requests_event.clear()

    def send_query_items(self, items: list[ExperienceItem]) -> list[ObjectRef]:
        # items will be send to the same worker
        refs = []
        dp_rank = next(self.get_available_dp_rank())

        for req_item in items:
            req_item.running_dp_rank = dp_rank
            req_item.data.meta_info["response_callback_fn"] = self.response_callback_fn
            req_item.data.meta_info["request_id"] = f"{req_item.request_id}_{self.global_step}"
            refs.append(
                self.actor_cluster.workers[dp_rank].add_request.remote(
                    command=GenerateRequestType.ADD, data=req_item.data
                )
            )
            req_item.data.meta_info.pop("response_callback_fn")
            req_item.data.meta_info["request_id"] = f"{req_item.request_id}"
        return refs

    def send_paused_requests(self) -> Optional[list[ObjectRef]]:
        with self.lock:
            paused_item = self.replay_buffer.pop_paused_item()
        if paused_item is None:
            return None
        return self.send_query_items([paused_item])

    def send_next_data_item(self, data: DataProto) -> list[ObjectRef]:
        # get a query from dataset
        prompt_id = next(self.prompt_id_counter)
        dataset_item = self.get_next_dataset_item()
        domain = dataset_item.get("domain", "default")
        collect_data = self.collect_fn([dataset_item])
        request_data: DataProto = DataProto.from_single_dict(collect_data, meta_info=data.meta_info)

        # replica, redundancy
        request_data_list = self.expand_requests(request_data)
        req_items: list[ExperienceItem] = []
        for req in request_data_list:
            request_id = ray.get(self.request_counter.get_value.remote())
            req.meta_info["prompt_id"] = prompt_id
            request_item = ExperienceItem(
                request_id=f"{request_id}",
                prompt_id=prompt_id,
                sampling_start_step=self.global_step,
                domain=domain,
                data=req,
            )
            req_items.append(request_item)

        with self.lock:
            for req_item in req_items:
                self.replay_buffer.add_request_item(req_item)
        return self.send_query_items(req_items)

    def _get_min_step_and_prompt_num(self, for_keep=False):
        if self.is_val:
            return None, None

        min_start_step = self.global_step - math.ceil(self.pipeline_config.async_generation_ratio)
        min_step_ratio = self.pipeline_config.async_generation_ratio % 1
        min_step_ratio = min_step_ratio if min_step_ratio > 0 else 1.0
        min_step_ratio += max(self.init_global_step - min_start_step, 0)
        if not for_keep:
            min_step_ratio = min(min_step_ratio, 1.0)

        required_prompt_num = round(self.batch_size * min_step_ratio)
        min_start_step = max(min_start_step, self.init_global_step)
        return min_start_step, required_prompt_num

    def get_batch(self, data: DataProto, batch_size: int) -> DataProto:
        if self.is_val:
            self.async_sending_thread.join()
            self.async_sending_thread = None
        self.batch_size = batch_size
        global_step = data.meta_info.get("global_step", 0)
        self.set_global_step(global_step)
        min_start_step, min_step_prompt_num = self._get_min_step_and_prompt_num()
        num_return_sequences = self.generation_config['num_return_sequences']

        logger.info(f"get batch by {min_start_step=}, {min_step_prompt_num=} {num_return_sequences=} {batch_size=}")
        while True:
            with self.lock:
                finished_prompt_ids = self.replay_buffer.get_enough_finished_prompt_ids(
                    batch_size, min_start_step, min_step_prompt_num
                )
            if finished_prompt_ids is not None:
                break
            self.check_response_callback()
            self.check_worker_alive(self.actor_cluster)
            time.sleep(1)

        with self.lock:
            finished_items = self.replay_buffer.pop_finished_items_by_prompt_ids(finished_prompt_ids, num_return_sequences)
        return self.collect_items_as_batch(finished_items=finished_items, prompt_ids=finished_prompt_ids)

    def collect_items_as_batch(self, finished_items: List[ExperienceItem], prompt_ids: List[int]) -> DataProto:
        collect_data_by_domain = defaultdict(list)
        data_off_policy_step = 0.0
        for item in finished_items:
            collect_data_by_domain[item.domain].append(item.data)
            data_off_policy_step += self.global_step - item.sampling_start_step
        data_off_policy_step = data_off_policy_step / len(finished_items)

        collect_data_by_domain = {
            domain: DataProto.concat(data_list) for domain, data_list in collect_data_by_domain.items()
        }
        query_use_count = len(prompt_ids)
        collect_data_num = sum(data.batch.batch_size[0] for data in collect_data_by_domain.values())
        logger.info(
            f"total collect data: {collect_data_num}, collect queries: {len(prompt_ids)} "
            f"used queries: {query_use_count} filter queries: {self.filter_prompt_num}"
        )

        batch = DataProto.concat(list(collect_data_by_domain.values()))
        batch.meta_info["metrics"] = {
            "scheduler/collect_query_count": len(prompt_ids),
            "scheduler/query_use_count": query_use_count,
            "scheduler/off_policy_ratio": data_off_policy_step,
            "scheduler/filter_query_count": self.filter_prompt_num,
        }
        self.filter_prompt_num = 0 # report here, so refresh here

        metrics = {}
        for domain, response_batch in collect_data_by_domain.items():
            sequence_score = response_batch.batch["scores"]
            metrics[f"scheduler/{domain}/score/mean"] = torch.mean(sequence_score).detach().item()
            metrics[f"scheduler/{domain}/score/max"] = torch.max(sequence_score).detach().item()
            metrics[f"scheduler/{domain}/score/min"] = torch.min(sequence_score).detach().item()

        batch.meta_info["metrics"].update(metrics)
        return batch

    def set_global_step(self, global_step: int):
        self.init_global_step = (
            min(global_step, self.init_global_step) if self.init_global_step is not None else global_step
        )
        self.global_step = global_step

    def pause_sampling(self, data: DataProto):
        if self.async_sending_thread is not None:
            self.stop_sending_requests_event.set()
            logger.info("waiting for async sending thread to finish...")
            self.async_sending_thread.join()
            self.async_sending_thread = None
        self.set_global_step(data.meta_info.get("global_step"))

        stop_refs = []
        for infer_worker in self.actor_cluster.workers:
            stop_refs.append(infer_worker.add_request.remote(command=GenerateRequestType.STOP, data=None))
        ray.get(stop_refs)
        logger.info("async sampling paused, waiting for all requests to be collected...")
        start_time = time.time()
        timeout = 120
        while True:
            if self.replay_buffer.running_request_num() == 0:
                break
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout after {timeout}s waiting for running requests to complete. "
                             f"Remaining running requests: {self.replay_buffer.running_request_num()}")
                break
            self.check_response_callback()
            time.sleep(1)
        logger.info(f"async sampling paused, replay_buffer info: {self.replay_buffer.info()}")

    def sending_request(self, data: DataProto):
        all_refs = []
        if self.is_val:
            for i in range(self.batch_size):
                all_refs.extend(self.send_next_data_item(data))
            ray.get(all_refs)
            logger.info(f"async validation send {self.batch_size} prompts.")
            return

        while True:
            refs = self.send_paused_requests()
            if refs is None:
                break
            all_refs.extend(refs)

        buffer_prompt_num = self.batch_size * max(1.0, self.pipeline_config.async_generation_ratio)
        if self.is_use_additional_prompts:
            buffer_prompt_num += self.max_additional_running_prompts

        send_prompt_num = 0
        while True:
            if self.stop_sending_requests_event.is_set():
                break
            if (
                self.replay_buffer.prompt_num() >= buffer_prompt_num
                or self.replay_buffer.running_request_num() >= self.cluster_max_running_requests
            ):
                time.sleep(1)
                continue
            all_refs.extend(self.send_next_data_item(data))
            send_prompt_num += 1
        ray.get(all_refs)
        logger.info(f"async sending thread send {send_prompt_num} prompts.")

    def start_sampling(self, data: DataProto, batch_size: int):
        # in async training, called after model update
        self.batch_size = batch_size

        global_step = data.meta_info.get("global_step", 0)
        self.set_global_step(global_step)
        logger.info(f"start async sampling, global_step: {global_step} {self.replay_buffer.info()}")

        self.generation_config = copy.deepcopy(data.meta_info["generation_config"])
        data.meta_info["collect_non_finish"] = True
        self.reset_status()
        logger.info(
            f"start async sampling, batch_size: {self.batch_size}, "
            f"num_return_sequences: {self.generation_config['num_return_sequences']}"
        )
        self.async_sending_thread = threading.Thread(target=self.sending_request, args=(data,))
        self.async_sending_thread.start()

    @ray.method(concurrency_group="multi_thread")
    def report_response(self, data: DataProto):
        try:
            data.meta_info["request_id"] = data.meta_info["request_id"].split("_")[0]
            experience_item = self.replay_buffer.get_item(data.meta_info["request_id"])
            prompt_id = experience_item.prompt_id
            num_return_sequences = self.generation_config["num_return_sequences"]

            is_finished = is_report_data_finished(data)
            batch = self.postprocess_output_ids(data) if is_finished else self.postprocess_paused_data(data)
            with self.lock:
                if not is_finished:
                    self.replay_buffer.report_item(batch, is_finished=is_finished)
                    return
                reward_worker = next(self.reward_worker_iters[experience_item.domain])

            # call reward
            rewards: DataProto = ray.get(reward_worker.compute_rewards.remote(batch))
            batch.union(rewards)

            with self.lock:
                self.replay_buffer.report_item(batch, is_finished=is_finished)
                if self.replay_buffer.is_prompt_id_finished(prompt_id):
                    return
                data_list = self.replay_buffer.get_finished_data_by_prompt_id(prompt_id)
            if not sum(data.batch.batch_size[0] for data in data_list) >= num_return_sequences:
                return

            need_prompt = self.query_filter_fn(data_list, self.pipeline_config)
            with self.lock:
                if need_prompt:
                    abort_items = self.replay_buffer.mark_prompt_id_finished(prompt_id, num_return_sequences)
                    self.progress_bar.update()
                else:
                    abort_items, _ = self.replay_buffer.pop_prompt_id(prompt_id)
                    self.filter_prompt_num += 1
                    logger.debug(f"prompt_id {prompt_id} is filtered, abort {len(abort_items)} requests")
            # abort uncompleted request
            self.abort_requests(abort_items)
        except Exception as e:
            self.exception_queue.put(e)

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

    def abort_requests(self, request_items: list[ExperienceItem]):
        abort_refs = []
        for item in request_items:
            dp_rank = item.running_dp_rank
            if dp_rank is None:
                continue
            abort_refs.append(
                self.actor_cluster.workers[dp_rank].add_request.remote(
                    command=GenerateRequestType.ABORT, data=DataProto(meta_info={"request_id": item.request_id})
                )
            )
        ray.get(abort_refs)

    def postprocess_paused_data(self, data: DataProto) -> DataProto:
        pre_data = self.replay_buffer.get_item(data.meta_info["request_id"]).data
        if "output_token_ids" not in data.meta_info:  # abort without inferred a token
            # too many this log means need more infer workers
            logger.info(f"received data without output_token_ids, request_id: {data.meta_info['request_id']}")
            return pre_data
        logger.debug(f"received paused data, request_id: {data.meta_info['request_id']}")

        assert len(data.meta_info["output_token_ids"]) == 1, (
            "async pipeline only support num_return_sequences=1 or is_num_return_sequences_expand=True"
        )

        # value: list[list[int|float]]
        for key in ["output_token_ids", "output_logprobs"]:
            cur_value = data.meta_info.pop(key)
            pre_value = pre_data.meta_info.get(f"pre_{key}", [[]] * len(cur_value))
            assert len(pre_value) == len(cur_value)
            pre_value = [pre_value[i] + cur_value[i] for i in range(len(pre_value))]
            data.meta_info[f"pre_{key}"] = pre_value
        new_batch = {**pre_data.batch}

        init_attention_mask = pre_data.batch.get("init_attention_mask", pre_data.batch["attention_mask"])
        new_batch["init_attention_mask"] = init_attention_mask
        new_batch["init_input_ids"] = pre_data.batch.get("init_input_ids", pre_data.batch["input_ids"])

        # concat pre output_ids and input_ids
        new_input_ids = concatenate_input_and_output(
            input_ids=new_batch["init_input_ids"],
            output_ids=torch.LongTensor(data.meta_info["pre_output_token_ids"]),
            num_return_sequences=len(data.meta_info["pre_output_token_ids"]),
        )
        new_batch["input_ids"] = new_input_ids

        new_attention_mask = torch.ones_like(new_input_ids, dtype=init_attention_mask.dtype)
        new_attention_mask[:, :init_attention_mask.shape[1]] = init_attention_mask
        new_batch["attention_mask"] = new_attention_mask

        max_new_tokens = self.pipeline_config.sequence_length - new_input_ids.shape[1]
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens {max_new_tokens} <= 0, init_input_ids {new_batch['init_input_ids'].shape}, "
            f"pre_output_token_ids {len(data.meta_info['pre_output_token_ids'][0])}")
        data.meta_info["max_new_tokens"] = max_new_tokens
        data = DataProto.from_dict(
            new_batch, non_tensors=pre_data.non_tensor_batch, meta_info={**pre_data.meta_info, **data.meta_info}
        )
        assert data.batch["init_attention_mask"].shape[1] == self.pipeline_config.prompt_length
        assert data.batch["init_input_ids"].shape[1] == self.pipeline_config.prompt_length
        return data

    def postprocess_output_ids(self, data: DataProto) -> DataProto:
        # postprocess_generate, input_ids, attention_mask, left pad
        request_id = data.meta_info["request_id"]
        request: DataProto = self.replay_buffer.get_item(request_id).data

        eos_token_id = data.meta_info["eos_token_id"]
        pad_token_id = data.meta_info["pad_token_id"]
        input_ids = request.batch.pop("init_input_ids", request.batch["input_ids"])
        request.batch["input_ids"] = input_ids
        request.batch["attention_mask"] = request.batch.pop("init_attention_mask", request.batch["attention_mask"])
        output_token_ids = data.meta_info["output_token_ids"]
        pre_output_token_ids = request.meta_info.pop("pre_output_token_ids", [[]] * len(output_token_ids))
        output_token_ids = [pre_output_token_ids[i] + output_token_ids[i] for i in range(len(pre_output_token_ids))]

        output_logprobs = data.meta_info.get("output_logprobs", None)
        if output_logprobs is not None:
            pre_output_logprobs = request.meta_info.get("pre_output_logprobs", [[]] * len(output_token_ids))
            output_logprobs = [pre_output_logprobs[i] + output_logprobs[i] for i in range(len(pre_output_logprobs))]

        output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]
        output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_tensor, num_return_sequences=len(output_tokens)
        )
        output: DataProto = postprocess_generate(
            prompts=request,
            output=output_tensor,
            num_return_sequences=len(output_tokens),
            sequence_length=self.pipeline_config.sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            output_logprobs=output_logprobs,
        )
        request_repeat = request.repeat(repeat_times=len(output_tokens))
        output.non_tensor_batch = request_repeat.non_tensor_batch
        output.meta_info = request_repeat.meta_info
        return output

    def expand_requests(self, data: DataProto):
        generate_opt_level = self.pipeline_config.generate_opt_level
        is_num_return_sequences_expand = self.pipeline_config.is_num_return_sequences_expand
        num_return_sequences = self.generation_config["num_return_sequences"]

        assert generate_opt_level > 0, (
            f"generate_opt_level {generate_opt_level} should > 0, " f"in dynamic sampling scheduler."
        )
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

    def check_worker_alive(self, cluster):
        current_time = time.time()
        if current_time - self.last_alive_check >= self.alive_check_interval:
            cluster.add_request(command=GenerateRequestType.ALIVE_CHECK, data=DataProto())
            self.last_alive_check = current_time
        if self.async_sending_thread is not None and not self.async_sending_thread.is_alive():
            raise RuntimeError("async sending thread is dead")

    def check_response_callback(self):
        if self.exception_queue.qsize() > 0:
            e = self.exception_queue.get()
            logger.error(f"report_response get exception {e}")
            raise e

    def get_available_dp_rank(self):
        while True:
            self.pre_send_dp_rank = (self.pre_send_dp_rank + 1) % len(self.mp_rank_zero)
            yield self.pre_send_dp_rank
