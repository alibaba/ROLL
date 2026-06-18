import asyncio
import copy
import json
import os
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
from roll.distributed.scheduler.user_defined_rollout_loop import (
    UserDefinedRolloutLoop,
    RolloutContext,
    expand_requests,
    query_filter,
)
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider, get_extra_data_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.rlvr_pipeline import update_dataset_domain
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


class FiltDataRolloutLoop(UserDefinedRolloutLoop):
    """
    custom to filter data whose length is larger than prompt_length
    """
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
            responses_list: List[List[DataProto]] = await asyncio.gather(
                *[self._generate_and_reward(context=context, req=req, domain=domain) for req in request_data_list]
            )
            responses: List[DataProto] = [item for sublist in responses_list for item in sublist]
            # some quick methods to reduce store and transfer overhead of multi-modal data by ray
            # 1. remove multi_modal_inputs which is for training before generate and add it back after reward
            # 2. remove multi_modal_data which is for inference after generate
            # 3. change dtype of features in multi_modal_inputs to model dtype which uses lower bytes
            for response in responses:
                response.non_tensor_batch.pop("multi_modal_data", None)
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


class RLVRVLMPipeline(BasePipeline):
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
        # pipeline related data args
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
        # update domain field, DynamicSamplingScheduler requires
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

        self.val_dataset = None
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            val_data_kwargs = create_pipeline_data_kwargs(
                self.pipeline_config.validation.data_args,
                tokenizer=self.tokenizer,
                processor=self.processor,
                is_val=True,
            )
            val_data_kwargs = _data_kwargs_helper(val_data_kwargs)
            val_dataset, val_collect_fn_kwargs = val_data_kwargs["dataset"], val_data_kwargs["collect_fn_kwargs"]
            self.val_dataset = val_dataset
            self.val_dataset = self.val_dataset.map(
                partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                desc="update_val_dataset_domain",
                load_from_cache_file=False,
            )

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        download_clusters = [self.actor_train, self.actor_infer]
        if self.pipeline_config.enable_reference:
            self.reference: Any = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
            download_clusters.append(self.reference)
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
            download_clusters.append(self.critic)
        # key must be same as domain, which is used in DynamicSamplingScheduler
        # to get corresponding reward
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
                # enable the following line to use dataloader to speedup video loading
                get_data_item_kwargs=train_data_kwargs.get("get_data_item_kwargs", None),
            )
            self.generate_schedulers[domain] = generate_scheduler
            self.domain_batch_size[domain] = domain_batch_size

            assert domain_batch_size < len(self.domain_datasets[domain]), (
                f"domain_batch_size {domain_batch_size} must be "
                f"less than the number of domain datasets {len(self.domain_datasets[domain])}"
            )

        if self.val_dataset:
            val_pipeline_config = copy.deepcopy(self.pipeline_config)
            val_pipeline_config.is_use_additional_prompts = False
            self.val_generate_scheduler = ray.remote(DynamicSamplingScheduler).options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            ).remote(
                pipeline_config=val_pipeline_config,
                actor_cluster=self.actor_infer,
                reward_clusters=self.rewards,
                dataset=self.val_dataset,
                collect_fn_cls=DataCollatorWithPaddingForMM,
                collect_fn_kwargs=val_collect_fn_kwargs,
                is_val=True,
                # enable the following line to use dataloader to speedup video loading
                get_data_item_kwargs=val_data_kwargs.get("get_data_item_kwargs", None),
            )

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        if self.pipeline_config.enable_reference:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
        refs = []
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        ray.get([scheduler.initialize.remote() for scheduler in self.generate_schedulers.values()])
        if self.val_dataset:
            ray.get(self.val_generate_scheduler.initialize.remote())

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = {}
        for domain in self.rewards.keys():
            self.running[domain] = RunningMoments()

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
        actor_train_timer = _Timer(window_size=5)

        metrics_mgr.timers["tps"] = tps_timer
        metrics_mgr.timers["actor_infer"] = actor_infer_timer
        metrics_mgr.timers["actor_infer_response"] = actor_infer_response_timer
        metrics_mgr.timers["actor_train"] = actor_train_timer

        pre_step_total_time = 0
        if self.pipeline_config.async_pipeline:
            for reward_cluster in self.rewards.values():
                reward_cluster.load_states()

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            self.global_step = global_step
            logger.info(f"pipeline step {global_step} start...")

            metrics_mgr.clear_metrics()
            with (tps_timer, Timer(name="step_total", logger=None) as step_total_timer,
                tracer.start_as_current_span("pipeline_step", attributes={"global_step": global_step}),
                ExitStack() as defer,
            ):
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

                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                with Timer(name="step_stop_server", logger=None) as step_stop_server_timer, \
                        tracer.start_as_current_span("stop_server"):
                    if self.pipeline_config.async_pipeline:
                        ray.get([scheduler.pause_sampling.remote() for scheduler in self.generate_schedulers.values()])
                        self.actor_infer.offload_states(include=OffloadStateType.other_params)
                metrics_mgr.add_metric("time/step_stop_server", step_stop_server_timer.last)

                with Timer(name="step_model_update", logger=None) as step_model_update_timer, \
                        tracer.start_as_current_span("model_update"):
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics_mgr.add_metrics(model_update_metrics)
                    batch.meta_info["generation_config"] = self.get_generation_config()
                metrics_mgr.add_metric("time/step_model_update", step_model_update_timer.last)

                self.actor_infer.load_states(blocking=True)

                if not self.pipeline_config.async_pipeline:
                    for reward_cluster in self.rewards.values():
                        reward_cluster.load_states()

                if self.val_dataset and global_step % self.pipeline_config.eval_steps == 0:
                    with Timer(name="val_step", logger=None) as val_step_timer, \
                            tracer.start_as_current_span("validation"):
                        val_metrics = self.val(global_step=global_step)
                    metrics_mgr.add_metrics(val_metrics)
                    metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                # 要按domain group by生成对应的batch
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
                        for reward_cluster in self.rewards.values():
                            reward_cluster.offload_states()
                        self.actor_infer.offload_states()
                metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

                batch = generate_output
                defer.callback(lambda b=batch: DataProto.drop(b))

                # mark here to make megatron get_data_input broadcast with non_batch_tensor
                batch.meta_info["_broadcast_non_tensor_batch"] = True
                batch.meta_info["loss_mask_keys"] = ["response_mask", "final_response_mask"]
                batch.non_tensor_batch['sample_uuid'] = np.array([str(uuid.uuid4()) for _ in range(batch.batch.shape[0])], dtype=object)
                batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)

                with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer, \
                        tracer.start_as_current_span("cal_ref_log_probs"):
                    if self.pipeline_config.enable_reference:
                        batch_balance(batch, dp_size=self.reference.dp_size, minibatch_size=len(batch))
                        ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                        metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        batch = batch.union(ref_log_probs)
                metrics_mgr.add_metric("time/ref_log_probs_values", cal_ref_log_probs_timer.last)

                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer, \
                        tracer.start_as_current_span("cal_old_log_probs_values"):
                    batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                    if self.pipeline_config.enable_old_logprobs_recompute:
                        batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                        old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                        old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                        agg_entropy = agg_loss(
                            loss_mat=old_log_probs.batch["entropy"],
                            loss_mask=batch.batch["response_mask"][:, 1:],
                            loss_agg_mode="token-mean",
                        )
                        batch.meta_info["agg_entropy"] = agg_entropy

                        batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                        metrics_mgr.add_reduced_metrics(old_log_probs.meta_info.pop("metrics", {}))
                    else:
                        # Use zeros when optimization is enabled
                        batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics_mgr.add_reduced_metrics(values.meta_info.pop("metrics", {}))

                    # Mock ref_log_probs using old_log_probs if reference is disabled
                    if not self.pipeline_config.enable_reference:
                        batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                metrics_mgr.add_metric("time/old_log_probs", cal_old_logpb_timer.last)

                # group by domain to process reward
                # Restore original generate order before group_by, so that same-prompt responses
                # remain contiguous for correct GRPO group reward normalization (reshape by n_sample).
                batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                batch_list = []
                for domain, domain_batch in batch_grouped.items():
                    # 1. get sample level mask
                    with Timer(name="get_sample_level_mask", logger=None) as get_sample_level_mask_timer, \
                            tracer.start_as_current_span("get_sample_level_mask"):
                        domain_batch, mask_metrics = get_sample_level_mask(domain_batch, self.pipeline_config)
                        metrics_mgr.add_domain_metrics(domain, mask_metrics)
                    metrics_mgr.add_metric("time/get_sample_level_mask", get_sample_level_mask_timer.last)

                    # 2. process reward
                    with Timer(name="reward_postprocess", logger=None) as reward_postprocess_timer, \
                            tracer.start_as_current_span("reward_postprocess"):
                        domain_batch, response_level_metrics = reward_postprocess(
                            domain_batch, self.pipeline_config, self.running
                        )
                        metrics_mgr.add_domain_metrics(domain, response_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/reward_postprocess": reward_postprocess_timer.last})

                    # 3. compute token level rewards
                    with Timer(name="get_token_reward", logger=None) as get_token_reward_timer, \
                            tracer.start_as_current_span("get_token_reward"):
                        domain_batch, token_level_metrics = compute_token_reward(
                            domain_batch, self.pipeline_config, self.kl_ctrl
                        )
                        metrics_mgr.add_domain_metrics(domain, token_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_token_reward": get_token_reward_timer.last})

                    # 4. compute advantage
                    final_response_mask = domain_batch.batch["final_response_mask"].clone()
                    with Timer(name="compute_advantage", logger=None) as compute_advantage_timer, \
                            tracer.start_as_current_span("compute_advantage"):
                        domain_batch = compute_advantage(
                            data=domain_batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                            response_mask=final_response_mask,
                            pipeline_config=self.pipeline_config,
                        )
                        domain_metrics = reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        metrics_mgr.add_domain_metrics(domain, domain_metrics)
                        batch_list.append(domain_batch)
                    metrics_mgr.add_domain_metrics(domain, {"time/compute_advantage": compute_advantage_timer.last})

                batch = DataProto.concat(batch_list)

                if batch.batch["final_response_mask"].sum() == 0:
                    logger.info("Warning: final_response_mask.sum() == 0! Current step will be skipped.")
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 1)
                    metrics = metrics_mgr.get_metrics()
                    # do ckpt
                    self.state.step = global_step
                    self.state.log_history.append(metrics)
                    for domain, scheduler in self.generate_schedulers.items():
                        self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())
                    self.do_checkpoint(global_step=global_step)
                    self.tracker.log(values=metrics, step=global_step)
                    continue
                else:
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 0)

                batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                batch.pop("prompt_id")

                metrics_mgr.add_all_metrics(
                    global_step,
                    batch,
                    resource_manager=self.resource_manager,
                    actor_infer=self.actor_infer,
                    actor_train=self.actor_train,
                )
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                metrics_mgr.add_domain_all_metrics(global_step, batch_grouped)

                if self.pipeline_config.enable_old_logprobs_recompute:
                    batch, corr_metrics = apply_train_infer_correction_to_batch(self.pipeline_config, batch)
                    metrics_mgr.add_metrics(corr_metrics)

                with Timer(name="step_train", logger=None) as step_train_timer, \
                        tracer.start_as_current_span("train"):
                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                    with actor_train_timer:
                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            # Reorder data for DP rank load balancing
                            batch_balance_metrics = batch_balance(
                                batch,
                                dp_size=self.actor_train.dp_size,
                                minibatch_size=self.pipeline_config.actor_train.training_args.per_device_train_batch_size
                                * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps
                                * self.actor_train.dp_size,
                                logging_prefix="global_seqlen/actor_train",
                            )
                            metrics_mgr.add_metrics(batch_balance_metrics)
                            # update actor
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=actor_train_metrics_refs
                            )
                            metrics_mgr.add_reduced_metrics(actor_train_metrics.meta_info.pop("metrics", {}))

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                        metrics_mgr.add_reduced_metrics(critic_train_metrics.meta_info.pop("metrics", {}))

                metrics_mgr.add_metric("time/step_train", step_train_timer.last)

                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_response_timer.push_units_processed(
                    n=torch.sum(batch.batch["response_mask"]).detach().item()
                )
                actor_train_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                metrics = metrics_mgr.get_metrics()
                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)
                for domain, scheduler in self.generate_schedulers.items():
                    self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                self.do_checkpoint(global_step=global_step)

                self.tracker.log(values=metrics, step=global_step)

                if global_step % self.pipeline_config.logging_steps == 0:
                    if int(os.environ.get("RAY_PROFILING", "0")):
                        timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                        os.makedirs(timeline_dir, exist_ok=True)
                        ray.timeline(
                            filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                        )

                    prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(
                        generate_output.batch["responses"], skip_special_tokens=True
                    )
                    generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                    logger.info(json.dumps(generate_examples, ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
            pre_step_total_time = step_total_timer.last

        ray.get([scheduler.shutdown.remote() for scheduler in self.generate_schedulers.values()])
        if self.val_dataset:
            ray.get(self.val_generate_scheduler.shutdown.remote())

        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self, global_step):
        defer = ExitStack()
        val_metrics_mgr = MetricsManager()
        tracer = get_tracer("driver")
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer, \
                tracer.start_as_current_span("val_generate"):
            inject_trace_context(batch.meta_info)
            batch.meta_info["is_offload_states"] = False
            batch.meta_info["generation_config"] = self.pipeline_config.validation.generating_args.to_dict()
            batch.meta_info.update(
                {"global_step": self.global_step, "max_steps": self.pipeline_config.max_steps, "is_training": False}
            )
            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, global_step=global_step, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout,
            )
            generate_output.meta_info.pop("is_offload_states", None)
            val_metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

        batch = generate_output
        defer.callback(lambda b=batch: DataProto.drop(b))

        val_score_mean = batch.batch["scores"].detach().float().mean().item()
        val_metrics_mgr.add_metric("val_score/all/mean", val_score_mean)
        logger.info(json.dumps({"val_score/all/mean": val_score_mean}, ensure_ascii=False))

        epoch_batch = batch.pop(batch_keys=["scores"], non_tensor_batch_keys=["tag"])

        grouped_batch = epoch_batch.group_by("tag")
        for group_key, group_batch in grouped_batch.items():
            score_mean = group_batch.batch["scores"].mean().item()
            logger.info(f"val_score/{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_score", {f"{group_key}/mean": group_batch.batch["scores"].detach().float().mean().item()}
            )

        defer.close()
        return val_metrics_mgr.get_metrics()
