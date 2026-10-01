import asyncio
import copy
import json
import os
import time
import uuid
from contextlib import ExitStack
from functools import partial
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.configs import GeneratingArguments
from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.datasets.vlm_dataset_utils import create_pipeline_data_kwargs
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.router import is_report_data_finished
from roll.distributed.scheduler.user_defined_rollout_loop import (
    UserDefinedRolloutLoop,
    RolloutContext,
    expand_requests,
    postprocess_paused_data,
    query_filter,
    response_filter,
)
from roll.distributed.scheduler.protocol import DataProto, strip_multi_modal_for_reward
from roll.models.model_providers import default_processor_provider, get_extra_data_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
# from roll.pipeline.rlvr.timing import build_step_timing_log
from roll.datasets.dataset import update_dataset_domain
from roll.pipeline.rlvr.utils import dump_rollout_to_specific_path
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    batch_balance,
    compute_advantage,
    compute_token_reward,
    get_sample_level_mask,
    reduce_metrics,
    reward_postprocess,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager
from roll.utils.offload_states import OffloadStateType
from roll.utils.telemetry import get_tracer, inject_trace_context
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch


logger = get_logger()


def build_validation_batch_sizes(dataset_size: int, val_batch_size: int) -> List[int]:
    """Build validation rollout chunk sizes from the configured validation batch size."""
    if dataset_size <= 0:
        return []
    if val_batch_size <= 0:
        return [dataset_size]
    return [
        min(val_batch_size, dataset_size - start_idx)
        for start_idx in range(0, dataset_size, val_batch_size)
    ]


def build_validation_scheduler_step(global_step: int, batch_index: int, num_batches: int) -> int:
    """Build a monotonically increasing scheduler step for chunked validation."""
    return global_step * max(num_batches, 1) + batch_index


class FiltDataRolloutLoop(UserDefinedRolloutLoop):
    """
    custom to filter data whose length is larger than prompt_length
    """
    def reward_transfer_ctx(self, req: DataProto):
        """Strip large multi-modal tensors before Ray transfer to reward workers."""
        return strip_multi_modal_for_reward(req.non_tensor_batch, enabled=True)

    async def process_new_prompt(self, context: RolloutContext) -> Optional[DataProto | List[DataProto]]:
        num_return_sequences = context.meta_info["generation_config"]["num_return_sequences"]
        is_num_return_sequences_expand = context.is_num_return_sequences_expand

        ################# STEP 1: get and filter dataset
        request_data, domain = context.get_request_data(meta_info=context.meta_info)
        if request_data.batch["input_ids"].shape[1] > context.prompt_length:
            logger.error(
                f"prompt_id {context.prompt_id} is filtered, "
                f"since input length={request_data.batch['input_ids'].shape[1]} is larger than prompt_length={context.prompt_length}"
            )
            return
        request_data_list = expand_requests(
            data=request_data,
            num_return_sequences=num_return_sequences,
            is_num_return_sequences_expand=is_num_return_sequences_expand,
        )

        ################# STEP 2: spawn tasks to process requests, including generate, reward, and filter at response level
        # Must run inside RolloutContext.do_generate_and_reward context.
        # RolloutContext.do_generate_and_reward will wait until can send new request (controlled by LoadBalancer).
        # And at exit, RolloutContext will enforce there is no running requests.
        async with context.do_generate_and_reward(max_concurrency=num_return_sequences):
            if is_num_return_sequences_expand and num_return_sequences > 1:
                responses = await self._generate_expanded_group_and_reward(
                    context=context,
                    request_data_list=request_data_list,
                    domain=domain,
                )
                if responses is None:
                    return None
            else:
                responses_list: List[Optional[List[DataProto]]] = await asyncio.gather(
                    *[self._generate_and_reward(context=context, req=req, domain=domain) for req in request_data_list]
                )
                if not all(sublist is not None for sublist in responses_list):
                    return None
                responses: List[DataProto] = [
                    item for sublist in responses_list if sublist is not None for item in sublist
                ]
            # some quick methods to reduce store and transfer overhead of multi-modal data by ray
            # 1. remove multi_modal_inputs which is for training before generate and add it back after reward
            # 2. remove multi_modal_data which is for inference after generate
            # 3. change dtype of features in multi_modal_inputs to model dtype which uses lower bytes
            multi_modal_inputs = request_data.non_tensor_batch.get("multi_modal_inputs")
            for response in responses:
                response.non_tensor_batch.pop("multi_modal_data", None)
                response.meta_info.setdefault("metrics", {})["multi_modal_size"] = np.array(
                    [
                        (
                            multi_modal_inputs[0]["pixel_values"].numel()
                            if "pixel_values" in multi_modal_inputs[0]
                            else 0
                        )
                        + (
                            multi_modal_inputs[0]["pixel_values_videos"].numel()
                            if "pixel_values_videos" in multi_modal_inputs[0]
                            else 0
                        )
                        + (
                            multi_modal_inputs[0]["input_features"].numel()
                            if "input_features" in multi_modal_inputs[0]
                            else 0
                        )
                    ],
                    dtype=object,
                )
            # User can call RolloutContext.abort_running_requests to abort any running generate requests (generate will return a response
            # with finish_reason=="abort", user should distinguish this from partial rollout to avoid dead loop).
        # assert there is no running requests outside do_generate_and_reward context.

        ################# STEP 3: prompt level filter
        if not context.is_val and not query_filter(responses, context.pipeline_config):
            # TODO add metrics (query_filter_count)
            logger.debug(f"prompt_id {context.prompt_id} is filtered")
            return

        ################# STEP 4: return responses to commit to ReplayBuffer
        return responses

    async def _generate_expanded_group_and_reward(
        self,
        context: RolloutContext,
        request_data_list: List[DataProto],
        domain: str,
    ) -> Optional[List[DataProto]]:
        """Generate expanded samples independently, then score the complete prompt group once."""
        for _ in range(5):
            generated_batches = await asyncio.gather(
                *[
                    self._generate_without_reward(context=context, req=req, domain=domain)
                    for req in request_data_list
                ]
            )
            if not all(batch is not None for batch in generated_batches):
                return None

            group_batch = DataProto.concat([batch for batch in generated_batches if batch is not None])
            with self.reward_transfer_ctx(group_batch):
                rewards = await context.compute_rewards(req=group_batch, domain=domain)

            group_metrics = group_batch.meta_info.pop("metrics", {})
            group_metrics.update(rewards.meta_info.pop("metrics", {}))
            group_batch.union(rewards)
            group_batch.meta_info["metrics"] = group_metrics

            responses = [group_batch[[idx]] for idx in range(len(group_batch))]
            filtered_responses = [
                response
                for response in responses
                if context.is_val or response_filter(response, context.pipeline_config)
            ]
            if len(filtered_responses) == len(responses):
                return filtered_responses
            logger.debug(
                f"prompt_id {context.prompt_id} filtered "
                f"{len(responses) - len(filtered_responses)} responses after grouped reward; retrying group"
            )
        return filtered_responses

    async def _generate_without_reward(
        self,
        context: RolloutContext,
        req: DataProto,
        domain: str,
    ) -> Optional[DataProto]:
        """Run generation for one expanded request without invoking the reward worker."""
        with get_tracer("scheduler").start_as_current_span("generate_without_reward"):
            req = copy.deepcopy(req)
            collect_unfinished = req.meta_info.get("collect_unfinished", False)

            while True:
                data = await context.generate(req=req, domain=domain)

                if data is None:
                    return None
                if is_report_data_finished(data):
                    return self.postprocess_output_data(req, data, context.sequence_length)
                if not collect_unfinished:
                    logger.info(f"received unfinished response {context.prompt_id=}")
                    return None
                req = postprocess_paused_data(req, data, context.sequence_length, context.prompt_length)


class RLVRVLMPipeline(BasePipeline):
    """
    仅保留 RLVRVLMPipeline 的 rollout 部分，用于调试 rollout 性能。

    去掉了 actor_train / reference / critic / 训练 / checkpoint / advantage 等无关逻辑，
    每个 step 只做 generate（rollout），并输出 rollout 的耗时与吞吐指标。
    """

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        pipeline_config.user_defined_rollout_loop_cls = f"{self.__class__.__module__}.FiltDataRolloutLoop"

        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        # prepare dataset and collect_fn_kwargs
        train_data_kwargs = create_pipeline_data_kwargs(
            self.pipeline_config.actor_train.data_args, tokenizer=self.tokenizer, processor=self.processor
        )

        def _data_kwargs_helper(data_kwargs):
            dataset, collect_fn_kwargs = data_kwargs["dataset"], data_kwargs["collect_fn_kwargs"]
            assert "tag" in dataset.features, "dataset should include tag field to get domain"
            collect_fn_kwargs["extra_unpadded_keys"] = list(
                set(collect_fn_kwargs.get("extra_unpadded_keys", []) + ["domain"])
            )
            collect_fn_kwargs["extra_data_provider"] = collect_fn_kwargs.get(
                "extra_data_provider",
                get_extra_data_provider(
                    self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
                ),
            )
            collect_fn_kwargs["max_length"] = collect_fn_kwargs.get("max_length", self.pipeline_config.prompt_length)
            collect_fn_kwargs["padding"] = collect_fn_kwargs.get("padding", "max_length")
            collect_fn_kwargs["mm_feature_dtype"] = collect_fn_kwargs.get(
                "mm_feature_dtype", self.pipeline_config.actor_train.model_args.dtype
            )
            return data_kwargs

        train_data_kwargs = _data_kwargs_helper(train_data_kwargs)
        dataset, collect_fn_kwargs = train_data_kwargs["dataset"], train_data_kwargs["collect_fn_kwargs"]
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            desc="update_dataset_domain",
            load_from_cache_file=False,
        )

        self.domain_datasets: Dict[str, datasets.Dataset] = {}
        for domain in self.pipeline_config.actor_train.data_args.domain_interleave_probs.keys():
            self.domain_datasets[domain] = dataset.filter(
                lambda example, dom: example["domain"] == dom,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                fn_kwargs={"dom": domain},
            )
            assert len(self.domain_datasets[domain]) > 0, f"domain dataset {domain} has no data"

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        # rollout 只需要 actor_infer 与 reward clusters
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        download_clusters = [self.actor_infer]

        # key must be same as domain, which is used in DynamicSamplingScheduler to get corresponding reward
        self.rewards: Dict[str, Any] = {
            key: Cluster(
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }
        download_clusters.extend(self.rewards.values())
        self.download_models(*download_clusters)

        domain_ratios = self.pipeline_config.actor_train.data_args.domain_interleave_probs
        self.generate_schedulers: Dict[str, DynamicSamplingScheduler] = {}
        self.domain_batch_size = {}
        domain_list = list(domain_ratios.keys())
        accumulated = 0
        for i, domain in enumerate(domain_list):
            if i == len(domain_list) - 1:
                domain_batch_size = self.pipeline_config.rollout_batch_size - accumulated
            else:
                domain_batch_size = int(domain_ratios[domain] * self.pipeline_config.rollout_batch_size)
            accumulated += domain_batch_size
            generate_scheduler = ray.remote(DynamicSamplingScheduler).options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            ).remote(
                pipeline_config=self.pipeline_config,
                actor_cluster=self.actor_infer,
                reward_clusters={domain: self.rewards[domain]},
                dataset=self.domain_datasets[domain],
                collect_fn_cls=DataCollatorWithPaddingForMM,
                collect_fn_kwargs=collect_fn_kwargs,
                state=self.state.kv.get(f"scheduler_state_{domain}", None),
                get_data_item_kwargs=train_data_kwargs.get("get_data_item_kwargs", None),
            )
            self.generate_schedulers[domain] = generate_scheduler
            self.domain_batch_size[domain] = domain_batch_size

            assert domain_batch_size < len(self.domain_datasets[domain]), (
                f"domain_batch_size {domain_batch_size} must be "
                f"less than the number of domain datasets {len(self.domain_datasets[domain])}"
            )

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs = []
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        ray.get([scheduler.initialize.remote() for scheduler in self.generate_schedulers.values()])

    def get_generation_config(self, generating_args: Optional[GeneratingArguments] = None):
        generating_args = (
            generating_args if generating_args is not None else self.actor_infer.worker_config.generating_args
        )
        generation_config = generating_args.to_dict()
        if self.pipeline_config.async_pipeline:
            generation_config["logprobs"] = 1
        return generation_config

    @torch.no_grad()
    def run(self):
        metrics_mgr = MetricsManager()
        tracer = get_tracer("driver")

        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)

        metrics_mgr.timers["tps"] = tps_timer
        metrics_mgr.timers["actor_infer"] = actor_infer_timer
        metrics_mgr.timers["actor_infer_response"] = actor_infer_response_timer

        pre_step_total_time = 0

        # rollout 调试：actor_infer 与 reward 常驻，不做 offload/model_update
        self.actor_infer.load_states(blocking=True)
        for reward_cluster in self.rewards.values():
            reward_cluster.load_states()

        for global_step in range(self.pipeline_config.max_steps):
            self.global_step = global_step
            logger.info(f"rollout debug step {global_step} start...")

            metrics_mgr.clear_metrics()
            with (tps_timer, Timer(name="step_total", logger=None) as step_total_timer,
                  tracer.start_as_current_span("pipeline_step", attributes={"global_step": global_step})):
                logger.info(f"pre_step_total_time: {pre_step_total_time}")
                metrics_mgr.add_metric("time/step_total", pre_step_total_time)

                batch: DataProto = DataProto(
                    meta_info={
                        "global_step": global_step,
                        "collect_unfinished": self.pipeline_config.async_pipeline,
                        "max_steps": self.pipeline_config.max_steps,
                        "is_training": True,
                    }
                )
                batch.meta_info["generation_config"] = self.get_generation_config()

                # 按 domain group by 生成对应的 batch（rollout 核心逻辑）
                with actor_infer_timer, actor_infer_response_timer, Timer(
                    name="step_generate", logger=None
                ) as step_generate_timer, tracer.start_as_current_span("generate"):
                    domain_batches = {}
                    scheduler_refs = {}
                    for domain, scheduler in self.generate_schedulers.items():
                        inject_trace_context(batch.meta_info)
                        scheduler_refs[domain] = scheduler.get_batch.remote(
                            data=batch, global_step=global_step, batch_size=self.domain_batch_size[domain]
                        )
                    for domain, scheduler_ref in scheduler_refs.items():
                        domain_batch: DataProto = ray.get(scheduler_ref, timeout=self.pipeline_config.rpc_timeout)
                        metrics_mgr.add_domain_metrics(
                            domain, reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        )
                        domain_batches[domain] = domain_batch
                    generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
                    dump_rollout_to_specific_path(
                        self.pipeline_config.rollout_dump_dir, global_step, generate_output, self.tokenizer
                    )
                    generate_output.meta_info.pop("is_offload_states", None)

                    if not self.pipeline_config.async_pipeline:
                        ray.get([scheduler.pause_sampling.remote() for scheduler in self.generate_schedulers.values()])
                metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

                batch = generate_output

                # rollout 吞吐统计
                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_response_timer.push_units_processed(
                    n=torch.sum(batch.batch["response_mask"]).detach().item()
                )

                metrics = metrics_mgr.get_metrics()
                self.tracker.log(values=metrics, step=global_step)

                prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=False)
                responses = self.tokenizer.batch_decode(
                    generate_output.batch["responses"], skip_special_tokens=False
                )
                generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                logger.info(json.dumps(generate_examples, ensure_ascii=False))
                logger.info(json.dumps(metrics, ensure_ascii=False))

                DataProto.drop(batch)
                logger.info(f"rollout debug step {global_step} finished")
            pre_step_total_time = step_total_timer.last

        ray.get([scheduler.shutdown.remote() for scheduler in self.generate_schedulers.values()])
        logger.info("rollout debug pipeline complete!")
