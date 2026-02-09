import json
import os.path
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.datasets.global_dataset import GlobalDatasetManager
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.utils import (
    agentic_compute_advantage,
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
    get_agentic_response_level_mask,
)
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    compute_token_reward,
    masked_mean,
    reduce_metrics,
    batch_balance
)
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType


logger = get_logger()


def is_lora_training(pipeline_config: AgenticConfig) -> bool:
    return pipeline_config.actor_train.model_args.lora_target is not None

class AgenticPipeline(BasePipeline):
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        self.use_ref_model = self.pipeline_config.enable_reference and (not is_lora_training(self.pipeline_config))

        # Derived configuration for partial GPU mode (auto-detected from device_mapping)
        self.partial_gpu_mode: bool = False

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        # INIT PHASE: Create Clusters
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

        if self.use_ref_model:
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

        # INIT PHASE: Create Reward Cluster (if device_mapping is configured)
        self.reward = None
        self.reward_scheduler = None
        if (
            self.pipeline_config.reward is not None
            and len(self.pipeline_config.reward.device_mapping) > 0
        ):
            self.reward: Any = Cluster(
                name=self.pipeline_config.reward.name,
                worker_cls=self.pipeline_config.reward.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reward,
            )
            download_clusters.append(self.reward)

        # INIT PHASE: Download Models
        self.download_models(*download_clusters)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        if self.reward:
            # Create reward scheduler as Ray named actor for environment managers to access
            self.reward_scheduler = RequestScheduler.options(
                name=f"RewardScheduler-{self.pipeline_config.reward.name}",
                get_if_exists=True,
                namespace=RAY_NAMESPACE,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                infer_cluster=self.reward,
                pipeline_config=self.pipeline_config,
                resource_manager=self.resource_manager,
            )
            logger.info(f"Created reward scheduler as Ray named actor: RewardScheduler-{self.pipeline_config.reward.name}")

        # INIT PHASE: Create RolloutSchedulers
        self.train_rollout_scheduler = ray.remote(RolloutScheduler).options(
            name="RolloutScheduler-train",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False)).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.train_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="train",
        )

        self.val_rollout_scheduler = ray.remote(RolloutScheduler).options(
            name="RolloutScheduler-val",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False)).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.val_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="val",
        )
        self.val_dataset_manager = GlobalDatasetManager.options(name=f"val_dataset_manager",
                                                                get_if_exists=True,
                                                                namespace=RAY_NAMESPACE).remote()
        # INIT PHASE: Initialize Clusters
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs = []
        if self.reward:
            # INIT PHASE: Initialize Reward Cluster
            refs.extend(self.reward.initialize(pipeline_config=self.pipeline_config, blocking=False))
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        if self.use_ref_model:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
        # INIT PHASE: Setup Operations
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = RunningMoments()

        # Validate partial GPU mode configuration and set self.partial_gpu_mode
        if self.pipeline_config.partial_gpu_mode:
            self.partial_gpu_mode = self._validate_partial_gpu_config()
        else:
            self.partial_gpu_mode = False

    @torch.no_grad()
    def run(self):
        # Calculate tokens-per-second system throughput
        tps_timer = _Timer(window_size=5)

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}

            # Add overall step timing
            with Timer(name="pipeline_step_total", logger=None) as step_timer:
                with tps_timer:
                    # PHASE 1: Offload States
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)

                    # PHASE 2: Suspend & Stop Server
                    # Suspend rollout scheduler to pause request processing
                    ray.get(self.train_rollout_scheduler.suspend.remote())

                    # Stop generation server if using async mode (will restart after model update)
                    if self.pipeline_config.async_pipeline:
                        self.actor_infer.offload_states(include=OffloadStateType.other_params)

                    # PHASE 3: Model Update
                    with Timer(name="model_update", logger=None) as model_update_timer:
                        model_update_metrics: Dict = self.model_update(global_step)
                    metrics["time/step_model_update"] =model_update_timer.last
                    metrics.update(model_update_metrics)

                    # PHASE 4: init kv cache
                    self.actor_infer.load_states()
                    if self.reward:
                        self.reward.load_states()

                    # PHASE 5: Expand Sampler (partial GPU mode, step > 0)
                    # Restore routing state: model_update loaded states to ALL GPUs, now update active_dp_ranks
                    # Step 0: active_dp_ranks initialized with all ranks {0,1,2,3}, no expand needed
                    # Step 1+: After shrink in previous iteration, active_dp_ranks was {2,3}.
                    #          model_update just loaded states to [0,1,2,3], so update routing state to match.
                    #          Use skip_load=True to avoid re-loading already-loaded model states.
                    if self.partial_gpu_mode and global_step > 0:
                        target_gpus = []
                        if hasattr(self.actor_train.worker_config, 'device_mapping') and self.actor_train.worker_config.device_mapping:
                            target_gpus.extend(self.actor_train.worker_config.device_mapping)
                        if self.pipeline_config.adv_estimator == "gae":
                            if hasattr(self.critic.worker_config, 'device_mapping') and self.critic.worker_config.device_mapping:
                                target_gpus.extend(self.critic.worker_config.device_mapping)

                        if target_gpus:
                            expand_metrics = ray.get(
                                self.train_rollout_scheduler.expand_sampler.remote(target_gpus, skip_load=True)
                            )
                            logger.info(f"Expand routing state (skip_load): {expand_metrics}")
                            metrics.update({"expand/" + k: v for k, v in expand_metrics.items()})

                    batch: DataProto = DataProto()
                    batch.meta_info = {"global_step": global_step}

                    # PHASE 6: Validation (every eval_steps) - Async
                    val_future = None
                    val_metrics = {}
                    with Timer(name="val", logger=None) as val_timer:
                        if self.pipeline_config.eval_steps > 0 and global_step % self.pipeline_config.eval_steps == 0:
                            # Submit val task to thread pool asynchronously
                            val_future = self.executor.submit(self.val, global_step)

                        # PHASE 7: Rollout Get Batch
                        with Timer(name="rollout", logger=None) as rollout_timer:
                            batch = ray.get(self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                            sample_uuids = [f"{traj_id}_{i}" for i, traj_id in enumerate(batch.non_tensor_batch['traj_id'])]
                            batch.non_tensor_batch['sample_uuid'] = np.array(sample_uuids, dtype=object)
                            if "get_batch_return_start_time" in batch.meta_info:
                                metrics["time/get_batch_cost_train"] = time.time() - batch.meta_info.pop("get_batch_return_start_time")
                            actor_infer_metrics = self.actor_infer.get_metrics()
                            metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))
                            metrics.update(compute_rollout_traj_metrics(batch))

                            dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

                        metrics["time/step_rollout"] = rollout_timer.last
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                        batch.meta_info["global_step"] = global_step
                        batch.meta_info["_broadcast_non_tensor_batch"] = True
                        batch.meta_info["loss_mask_keys"] = ["response_mask"]

                        # PHASE 8: Stop Server Sync (sync mode only) - Wait for async val to complete
                        if val_future is not None:
                            val_metrics = val_future.result()

                    if len(val_metrics) > 0:
                        metrics.update(val_metrics)
                        metrics["time/step_val"] = val_timer.last

                    if not self.pipeline_config.async_pipeline:
                        # Suspend scheduler before offload actor infer, because there may be
                        # some inflight redundant trajectories.
                        ray.get(self.train_rollout_scheduler.suspend.remote())
                        self.actor_infer.offload_states()
                        if self.reward:
                            self.reward.offload_states()

                    # PHASE 9: Shrink Sampler (partial GPU mode)
                    # Partial GPU overlap: Shrink sampler to free training GPUs before training phase
                    # This offloads actor_infer models from training GPUs (e.g., [0,1]) so they can be
                    # used by actor_train and critic for the training phase. After shrink, actor_infer
                    # only has models loaded on inference-dedicated GPUs (e.g., [2,3]).
                    #
                    # Example with actor_infer on [0,1,2,3], actor_train on [0,1]:
                    #   Before shrink: actor_infer has models on all GPUs [0,1,2,3]
                    #   After shrink: actor_infer offloads from [0,1], keeps models on [2,3]
                    #   During training: actor_train uses freed GPUs [0,1]
                    #   Next iteration: model_update reloads actor_infer to all GPUs [0,1,2,3]
                    elif self.partial_gpu_mode:
                        with Timer(name="cal_ref_log_probs", logger=None) as shrink_timer:
                            target_gpus = []
                            # Collect actor_train GPUs
                            if hasattr(self.actor_train.worker_config, 'device_mapping') and self.actor_train.worker_config.device_mapping:
                                target_gpus.extend(self.actor_train.worker_config.device_mapping)
                            # Collect critic GPUs if using GAE
                            if self.pipeline_config.adv_estimator == "gae":
                                if hasattr(self.critic.worker_config, 'device_mapping') and self.critic.worker_config.device_mapping:
                                    target_gpus.extend(self.critic.worker_config.device_mapping)

                            assert target_gpus, "cannot be empty"
                            shrink_metrics = ray.get(self.train_rollout_scheduler.shrink_sampler.remote(target_gpus))
                            logger.info(f"Shrink sampler: {shrink_metrics}")
                            metrics.update({"shrink/" + k: v for k, v in shrink_metrics.items()})
                        metrics["time/step_shrink"] = shrink_timer.last

                    batch = compute_discounted_returns(batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)

                    batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    # PHASE 11: Reference Log Probs
                    with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                        # TODO better the code structure, move the dynamic batching and sequence packing to worker/strategy
                        if self.pipeline_config.enable_reference:
                            worker_config = self.pipeline_config.reference if self.use_ref_model else self.pipeline_config.actor_train
                            worker = self.reference if self.use_ref_model else self.pipeline_config.actor_train
                            if worker_config.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    worker.dp_size,
                                    worker_config.max_tokens_per_microbatch_in_infer,
                                    worker_config.sequence_length_round_in_infer,
                                    worker_config.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    worker_config.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "reference/compute_log_probs",
                                )
                                metrics.update(dynamic_batching_metrics)
                            if not self.use_ref_model:
                                batch.meta_info["disable_adapter"] = True
                                batch.meta_info["is_offload_states"] = False
                                batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                                ref_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                            else:
                                batch_balance(batch, dp_size=self.reference.dp_size, minibatch_size=len(batch))
                                ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)

                            ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                            ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                            batch = batch.union(ref_log_probs)
                            avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                            metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                            metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                    metrics["time/step_ref_log_probs_values_reward"] = cal_timer.last

                    # PHASE 12: Old Log Probs & Values
                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        if self.pipeline_config.enable_reference and not self.use_ref_model:
                            batch.meta_info["disable_adapter"] = False
                        batch.meta_info["is_offload_states"] = False
                        if self.pipeline_config.enable_old_logprobs_recompute:
                            batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                                    self.pipeline_config.actor_train.sequence_length_round_in_infer,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "actor_train/compute_log_probs",
                                )
                                metrics.update(dynamic_batching_metrics)
                            old_log_probs: DataProto = self.actor_train.compute_log_probs(batch, blocking=True)
                            batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                            avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                            metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})
                            metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                            agg_entropy = agg_loss(
                                loss_mat=old_log_probs.batch["entropy"],
                                loss_mask=batch.batch["response_mask"][:, 1:],
                                loss_agg_mode="token-mean",
                            )
                            metrics.update({"critic/entropy/mean": agg_entropy.item()})
                        else:
                            batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

                        if self.pipeline_config.adv_estimator == "gae":
                            values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                        if self.pipeline_config.adv_estimator == "gae":
                            values = DataProto.materialize_concat(data_refs=values_refs)
                            batch = batch.union(values)
                            metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))

                        # Mock ref_log_probs using old_log_probs if reference cluster is disabled
                        if not self.pipeline_config.enable_reference:
                            batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                            avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                            metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})

                    metrics["time/step_old_log_probs_values"] = cal_old_logpb_timer.last

                    # TODO 当前这个还没用处
                    with Timer(name="cal_response_level_mask", logger=None) as timer:
                        # TODO 补充完善的过滤要求，不同环境需要维持统一过滤标识
                        batch, mask_metrics = get_agentic_response_level_mask(batch, self.pipeline_config)
                        metrics.update(mask_metrics)
                    metrics["time/step_cal_response_level_mask"] = timer.last

                    # PHASE 13: Advantage Computation
                    with Timer(name="cal_response_norm_rewards", logger=None) as timer:
                        # Rewards need to be processed after grouping
                        # We can group by tag(env_type)/traj_group_id(group)/batch(rollout_batch)... to compute rewards / advantages
                        # The compute_response_level_rewards function injects a response_level_rewards key into batch.batch.
                        batch, reward_metrics = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                        metrics.update(reward_metrics)
                    metrics["time/step_cal_norm_rewards"] = timer.last

                    with Timer(name="cal_token_reward", logger=None) as timer:
                        # Expand compute_response_level_rewards and add kl_penalty.
                        # batch, kl_metrics = apply_kl_penalty(data=batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty)
                        batch, token_level_metrics = compute_token_reward(batch, self.pipeline_config, self.kl_ctrl)
                        metrics.update(token_level_metrics)
                    metrics["time/step_cal_token_reward"] = timer.last

                    with Timer(name="compute_advantage", logger=None) as timer:
                        # Is the advantage calculated globally across the batch, or within each group?
                        batch = agentic_compute_advantage(
                            data=batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    metrics["time/step_adv"] = timer.last

                    if self.pipeline_config.enable_old_logprobs_recompute:
                        batch, corr_metrics = apply_train_infer_correction_to_batch(self.pipeline_config, batch,
                                                                                    update_mask_keys=batch.meta_info['loss_mask_keys'])
                        metrics.update(corr_metrics)

                    # PHASE 14: Training (critic + actor)
                    with Timer(name="train_timer", logger=None) as train_timer:
                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            batch_balance_metrics = batch_balance(batch, dp_size=self.actor_train.dp_size,
                                minibatch_size=self.actor_train.dp_size * self.pipeline_config.actor_train.training_args.per_device_train_batch_size *
                                self.pipeline_config.actor_train.training_args.gradient_accumulation_steps,
                                logging_prefix="global_seqlen/actor_train")
                            metrics.update(batch_balance_metrics)
                            # update actor
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                    self.pipeline_config.actor_train.sequence_length_round_in_train,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "actor_train/train_step",
                                )
                                metrics.update(dynamic_batching_metrics)
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                            metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                            metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                        tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    metrics["time/step_train"] = train_timer.last

                with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                    data_metrics = compute_train_data_metrics(batch=batch)

                metrics["time/step_compute_data_metrics"] = data_metrics_timer.last
                metrics.update(data_metrics)
                metrics["system/tps"] = tps_timer.mean_throughput
                metrics["system/samples"] = (global_step + 1) * self.pipeline_config.rollout_batch_size

                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)

                self.do_checkpoint(global_step=global_step)

                with Timer(name="log", logger=None) as log_timer:
                    if self.pipeline_config.logging_steps > 0 and global_step % self.pipeline_config.logging_steps == 0:
                        if int(os.environ.get("RAY_PROFILING", "0")):
                            timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                            os.makedirs(timeline_dir, exist_ok=True)
                            ray.timeline(
                                filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                            )

                        log_res = []
                        batch_grouped = batch.group_by(keys="traj_id")
                        for group_name, group_batch in batch_grouped.items():
                            if "step" in group_batch.non_tensor_batch.keys():
                                indices = torch.argsort(torch.from_numpy(group_batch.non_tensor_batch["step"].astype(np.int64)))
                                group_batch.reorder(indices)

                            prompt_mask = group_batch.batch["prompt_mask"]
                            non_prompt_mask = torch.logical_not(group_batch.batch["prompt_mask"]) * group_batch.batch["attention_mask"]
                            input_ids = group_batch.batch["input_ids"]
                            prompt_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(prompt_mask)]
                            response_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(non_prompt_mask)]
                            prompts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=False)
                            responses = self.tokenizer.batch_decode(response_ids_list, skip_special_tokens=False)
                            episode_scores = group_batch.non_tensor_batch["episode_scores"].tolist()
                            step_scores = group_batch.non_tensor_batch["step_scores"].tolist()
                            if isinstance(step_scores[0], np.ndarray):
                                step_scores = [t.tolist() for t in step_scores]

                            log_item = []
                            for prompt, response, episode_score, step_score in zip(
                                    prompts, responses, episode_scores, step_scores
                            ):
                                log_item.append(
                                    {
                                        "prompt": prompt,
                                        "response": response,
                                        "episode_score": episode_score,
                                        "step_score": step_score,
                                    }
                                )
                            log_res.append(log_item)
                            if len(log_res) >= 10:
                                break
                        logger.info(json.dumps(log_res, ensure_ascii=False))
                        logger.info(json.dumps(metrics, ensure_ascii=False))

                metrics["time/step_log"] = log_timer.last

            metrics["time/step_total"] = step_timer.last
            self.tracker.log(values=metrics, step=global_step)

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
            logger.info(f"epoch {global_step} finished")

        ray.get([
            self.train_rollout_scheduler.shutdown.remote(),
            self.val_rollout_scheduler.shutdown.remote(),
        ])


        logger.info("pipeline complete!")


    def val(self, global_step):
        batch = DataProto()
        metrics = {}
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["global_step"] = global_step
        ray.get(self.val_dataset_manager.reset.remote())
        eval_batch = ray.get(self.val_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.val_batch_size))

        if "get_batch_return_start_time" in eval_batch.meta_info:
            metrics["time/get_batch_cost_val"] = time.time() - eval_batch.meta_info.pop("get_batch_return_start_time")

        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, eval_batch)
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
        eval_score = get_episode_scores(eval_batch)
        eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
        eval_metrics["score/max"] = torch.max(eval_score).detach().item()
        eval_metrics["score/min"] = torch.min(eval_score).detach().item()

        batch_grouped = eval_batch.group_by(keys="tags")
        for group_name, group_batch in batch_grouped.items():
            traj_group_scores = []
            batch_traj_grouped = group_batch.group_by(keys="traj_group_id")
            for batch_traj_group_name, batch_traj_group in batch_traj_grouped.items():
                traj_group_score = get_episode_scores(batch_traj_group)
                traj_group_scores.append(traj_group_score.mean().item())
            eval_score = torch.tensor(traj_group_scores, dtype=torch.float)
            eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()

        metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
        logger.info(f"val_batch_size: {len(eval_batch)}")
        logger.info(f"val metrics: {metrics}")

        return metrics

    def adjust_batch(self, data: DataProto, mode="copy") -> DataProto:
        """
        ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/agent_system/multi_turn_rollout/utils.py#L86
        """
        actor_train_train_bsz = self.pipeline_config.actor_train.training_args.per_device_train_batch_size * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps * self.actor_train.dp_size
        actor_train_infer_bsz = self.pipeline_config.actor_train.infer_batch_size * self.actor_train.dp_size

        ref_infer_bsz = 1
        if hasattr(self, "reference"):
            ref_infer_bsz = self.pipeline_config.reference.infer_batch_size * self.reference.dp_size
        critic_train_bsz = 1
        critic_infer_bsz = 1
        if self.pipeline_config.adv_estimator == "gae":
            critic_train_bsz = self.pipeline_config.critic.training_args.per_device_train_batch_size * self.pipeline_config.critic.training_args.gradient_accumulation_steps * self.critic.dp_size
            critic_infer_bsz = self.pipeline_config.critic.infer_batch_size * self.critic.dp_size

        size_divide = np.lcm.reduce(np.array([actor_train_train_bsz, actor_train_infer_bsz, ref_infer_bsz, critic_infer_bsz, critic_train_bsz])).item()
        batch_size = data.batch.batch_size[0]
        threshold = batch_size % size_divide

        if threshold == 0:
            return data

        if mode == "auto":
            if threshold >= 0.5 * batch_size or  batch_size // size_divide == 0:
                mode = "copy"
            else:
                mode = "delete"
        elif mode == "random_sample":
            if batch_size < size_divide:
                mode = "copy"

        metrics = data.meta_info.get("metrics", {})
        metrics["system/batch_add_count"] = 0
        metrics["system/batch_remove_count"] = 0

        # 防止删除所有样本导致空批次
        if mode == "delete" and threshold >= batch_size:
            mode = "copy"

        if mode == "delete":
            remove_indices = np.random.choice(batch_size, threshold, replace=False)
            remove_indices = np.sort(remove_indices)
            keep_mask = np.ones(batch_size, dtype=bool)
            keep_mask[remove_indices] = False
            keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
            tensor_data = data.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
            adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
            metrics["system/batch_remove_count"] = len(remove_indices)
        elif mode == "copy":
            to_add = size_divide - threshold
            dup_indices = np.random.choice(batch_size, to_add, replace=True) if to_add > batch_size else np.random.choice(batch_size, to_add, replace=False)
            dup_proto = data.select_idxs(dup_indices)
            # TODO: set dup_proto response_mask to 0
            adjusted_batch = DataProto.concat([data, dup_proto])
            metrics["system/batch_add_count"] = to_add
        elif mode == "random_sample":
            select_indices = np.random.choice(batch_size, size_divide, replace=False)
            select_indices = np.sort(select_indices)
            adjusted_batch = data.select_idxs(select_indices)
            metrics["system/batch_remove_count"] = batch_size - size_divide
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        adjusted_batch.meta_info["metrics"] = metrics

        return adjusted_batch

    def _validate_partial_gpu_config(self) -> bool:
        """Derive partial_gpu_mode from device_mapping and validate all requirements.

        Universal validations (both Model A and B):
        - Reference colocation with actor_train

        Partial mode validations (Model B only - when train ⊂ infer):
        1. Minimum DP size (≥2)
        2. Async generation requirement (>0)
        3. Critic disjoint from actor_train
        4. Freed GPU capacity check
        5. TP/PP/EP compatibility
        6. At least 1 rank remains active

        Returns:
            partial_gpu_mode: True if train ⊂ infer (Configuration Model B),
                              False if train ∩ infer = ∅ (Configuration Model A)

        Raises:
            ValueError: Invalid configuration (device_mapping overlap, capacity issues,
                        DP size too small, missing async_generation_ratio, reference not colocated)
        """
        # rvst: yangpeng
        # Extract device mappings
        train_devices = set(self.actor_train.worker_config.device_mapping)
        infer_devices = set(self.actor_infer.worker_config.device_mapping)
        critic_devices = set(self.critic.worker_config.device_mapping) if hasattr(self, 'critic') and self.critic else set()
        ref_devices = set(self.reference.worker_config.device_mapping) if self.pipeline_config.enable_reference else set()
        reward_devices = set(self.reward.worker_config.device_mapping) if self.reward else set()

        # VAL: VAL_NON_EMPTY - ensure device_mapping not empty
        if not train_devices or not infer_devices:
            raise ValueError(
                f"device_mapping cannot be empty: "
                f"train={list(train_devices)}, infer={list(infer_devices)}"
            )

        # Universal validation: Reference must always colocate with actor_train (both Model A and B)
        # VAL: VAL_SUBSET (exact match) - reference colocation
        if self.pipeline_config.enable_reference:
            assert ref_devices == train_devices, (
                f"Reference device_mapping must match actor_train exactly: "
                f"ref={list(ref_devices)}, train={list(train_devices)}"
            )

        # Determine configuration mode
        if train_devices.isdisjoint(infer_devices):
            # Configuration Model A: Disjoint GPUs
            partial_gpu_mode = False
            logger.info("Detected Configuration Model A: Disjoint device_mapping, partial_gpu_mode=False")
            return partial_gpu_mode

        elif train_devices.issubset(infer_devices) and len(train_devices) < len(infer_devices):
            # Configuration Model B: Partial overlap
            partial_gpu_mode = True
            logger.info("Detected Configuration Model B: Subset device_mapping, partial_gpu_mode=True")

            # CRITICAL VALIDATIONS (6 checks for partial mode)

            # Validation 1: Minimum DP size
            # VAL: VAL_INT_RANGE(min=2, max=inf) - infer_dp_size
            infer_dp_size = self.actor_infer.worker_config.world_size
            assert infer_dp_size >= 2, (
                f"partial_gpu_mode requires actor_infer.dp_size >= 2, "
                f"got {infer_dp_size}"
            )

            # Validation 2: Async generation required
            # VAL: VAL_INT_RANGE(min=0.0, exclusive) - async_generation_ratio
            async_ratio = self.pipeline_config.async_generation_ratio
            assert async_ratio > 0, (
                f"partial_gpu_mode requires async_generation_ratio > 0, got {async_ratio}"
            )

            # Validation 3: Critic disjoint validation
            # VAL: VAL_SUBSET(critic_devices, infer_devices) + disjoint check
            if hasattr(self, 'critic') and self.critic is not None:
                assert critic_devices.issubset(infer_devices), (
                    f"Critic device_mapping must be subset of actor_infer: "
                    f"critic={list(critic_devices)}, infer={list(infer_devices)}"
                )
                assert critic_devices.isdisjoint(train_devices), (
                    f"Critic device_mapping must be disjoint from actor_train: "
                    f"critic={list(critic_devices)}, train={list(train_devices)}"
                )

            # Validation 4: Freed GPU capacity
            # VAL: VAL_INT_RANGE - freed GPU count check (no overlap)


            # Validation 5: TP/PP/EP compatibility
            # VAL: VAL_INT_RANGE(min=1) + device_mapping divisibility check
            # Extract TP and PP sizes from strategy config since workers aren't initialized yet
            infer_strategy_config = self.actor_infer.worker_config.strategy_args.strategy_config
            tp_size = infer_strategy_config.get("tensor_parallel_size", 1)
            pp_size = infer_strategy_config.get("pipeline_parallel_size", 1)

            assert tp_size >= 1 and pp_size >= 1, (
                f"tp_size and pp_size must be >= 1: tp={tp_size}, pp={pp_size}"
            )

            expected_gpu_count = tp_size * pp_size * infer_dp_size
            actual_gpu_count = len(infer_devices)
            assert expected_gpu_count == actual_gpu_count, (
                f"Parallelism configuration mismatch: "
                f"tp_size * pp_size * dp_size = {tp_size} * {pp_size} * {infer_dp_size} = {expected_gpu_count}, "
                f"but device_mapping has {actual_gpu_count} GPUs"
            )

            # Validation 6: At least 1 rank remains active
            # VAL: VAL_SUBSET, AST: AST_POSTCONDITION(remaining_ranks >= 1)
            gpus_per_dp_rank = tp_size * pp_size
            freed_gpus = train_devices | critic_devices
            freed_gpu_list = list(freed_gpus)
            self._validate_minimum_active_ranks(
                infer_dp_size, infer_devices, freed_gpu_list, gpus_per_dp_rank
            )

            logger.info(
                f"Partial GPU mode validated: infer_dp_size={infer_dp_size}, "
                f"freed_gpus={sorted(freed_gpus)}"
            )

            return partial_gpu_mode

        else:
            partial_gpu_mode = False
            assert len(train_devices) == len(infer_devices) + len(reward_devices),  "colocating mode"
            assert self.pipeline_config.async_generation_ratio == 0, "colocating mode only support sync/on-policy training"

            return partial_gpu_mode


    def _validate_minimum_active_ranks(
        self,
        infer_dp_size: int,
        infer_devices: set,
        freed_gpu_list: list,
        gpus_per_dp_rank: int
    ) -> None:
        """Validate at least 1 DP rank remains active after shrink.

        Args:
            infer_dp_size: Total DP size
            infer_devices: Infer device_mapping (as set for validation)
            freed_gpu_list: List of GPUs to free (train_devices | critic_devices)
            gpus_per_dp_rank: GPUs per DP rank (tp * pp)

        Raises:
            ValueError: If all ranks would be offloaded
        """
        # First validate that freed GPUs are subset of infer GPUs
        freed_gpu_set = set(freed_gpu_list)
        if not freed_gpu_set.issubset(infer_devices):
            raise ValueError(
                f"Freed GPUs (train + critic) must be subset of infer device_mapping: "
                f"freed={sorted(freed_gpu_list)}, infer={sorted(infer_devices)}"
            )

        # Convert infer_devices to ordered list to match DP rank assignment
        infer_devices_list = sorted(list(infer_devices))

        # Iterate through all DP ranks to find at least one that remains active
        # Each DP rank uses gpus_per_dp_rank consecutive GPUs from device_mapping
        at_least_one_active = False
        for dp_rank in range(infer_dp_size):
            # Get GPU range for this DP rank
            start_idx = dp_rank * gpus_per_dp_rank
            end_idx = start_idx + gpus_per_dp_rank
            dp_rank_gpus = set(infer_devices_list[start_idx:end_idx])

            # Check if this DP rank's GPUs are NOT in the freed set
            if dp_rank_gpus.isdisjoint(freed_gpu_set):
                at_least_one_active = True
                break

        if not at_least_one_active:
            raise ValueError(
                f"At least 1 DP rank must remain active after shrink. "
                f"All {infer_dp_size} DP ranks have at least one GPU in freed set. "
                f"infer_devices={sorted(infer_devices_list)}, freed_gpus={sorted(freed_gpu_list)}, "
                f"gpus_per_rank={gpus_per_dp_rank}"
            )

def get_episode_scores(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["episode_scores"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)

def get_traj_rollout_time(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["traj_rollout_time"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)

def get_traj_env_time(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["traj_env_time"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)


def compute_rollout_traj_metrics(batch) -> Dict:
    """
    Compute metrics for the rollout trajectory, before sample for train
    """
    episode_scores = get_episode_scores(batch)
    # fix: https://github.com/volcengine/verl/pull/60
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    prompt_mask = batch.batch["prompt_mask"].bool() # 首轮 prompt length
    prompt_lengths = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    non_prompt_mask = (torch.logical_not(batch.batch["prompt_mask"]) * batch.batch["attention_mask"]).float().sum(-1)

    metrics = {
        # score, sequence_score from env
        "rollout/score/mean": torch.mean(episode_scores).detach().item(),
        "rollout/score/max": torch.max(episode_scores).detach().item(),
        "rollout/score/min": torch.min(episode_scores).detach().item(),
        # response length
        "rollout/response_length/mean": torch.mean(response_length).detach().item(),
        "rollout/response_length/max": torch.max(response_length).detach().item(),
        "rollout/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "rollout/prompt_length/mean": torch.mean(prompt_lengths).detach().item(),
        "rollout/prompt_length/max": torch.max(prompt_lengths).detach().item(),
        "rollout/prompt_length/min": torch.min(prompt_lengths).detach().item(),
        # non-prompt length
        "rollout/non_prompt_length/mean": torch.mean(non_prompt_mask).detach().item(),
        "rollout/non_prompt_length/max": torch.max(non_prompt_mask).detach().item(),
        "rollout/non_prompt_length/min": torch.min(non_prompt_mask).detach().item(),
    }
    return metrics

def compute_train_data_metrics(batch):
    """
    Compute metrics on the training data.
    This is different from `rollout_traj`: `rollout_traj` contains trajectory data for the entire batch,
    while under `step_wise`, `train_batch` is sampled from `rollout_batch`, so the data distributions will differ.
    """
    # token_level_scores are per-token scores assigned by the reward model, possibly after normalization/clipping
    # score denotes the raw environment reward
    episode_scores = get_episode_scores(batch)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]
    # fix: https://github.com/volcengine/verl/pull/60
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    prompt_mask = batch.batch["prompt_mask"].bool() # 首轮 prompt length
    prompt_lengths = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    returns = batch.batch["returns"]
    non_prompt_mask = (torch.logical_not(batch.batch["prompt_mask"]) * batch.batch["attention_mask"]).float().sum(-1)

    # 从 batch 中提取 traj_rollout_time 相关指标
    # traj_rollout_times = []
    metrics = {
        # score, sequence_score from env
        "critic/score/mean": torch.mean(episode_scores).detach().item(),
        "critic/score/max": torch.max(episode_scores).detach().item(),
        "critic/score/min": torch.min(episode_scores).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": masked_mean(advantages, response_mask).detach().item(),
        "critic/advantages/max": torch.max(advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        "critic/advantages/min": torch.min(advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        # returns
        "critic/returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/returns/max": torch.max(returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        "critic/returns/min": torch.min(returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        # response length
        "tokens/response_length/mean": torch.mean(response_length).detach().item(),
        "tokens/response_length/max": torch.max(response_length).detach().item(),
        "tokens/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "tokens/prompt_length/mean": torch.mean(prompt_lengths).detach().item(),
        "tokens/prompt_length/max": torch.max(prompt_lengths).detach().item(),
        "tokens/prompt_length/min": torch.min(prompt_lengths).detach().item(),
        # prompt length(sys_obs)
        # "tokens/prompt_length_sys_obs/mean": torch.mean(prompt_lengths_sys_obs).detach().item(),
        # "tokens/prompt_length_sys_obs/max": torch.max(prompt_lengths_sys_obs).detach().item(),
        # "tokens/prompt_length_sys_obs/min": torch.min(prompt_lengths_sys_obs).detach().item(),
        # non-prompt length
        "tokens/non_prompt_length/mean": torch.mean(non_prompt_mask).detach().item(),
        "tokens/non_prompt_length/max": torch.max(non_prompt_mask).detach().item(),
        "tokens/non_prompt_length/min": torch.min(non_prompt_mask).detach().item(),
    }

    if "values" in batch.batch.keys():
        values = batch.batch["values"]
        # values
        metrics.update(
            {
                "critic/values/mean": masked_mean(values, response_mask).detach().item(),
                "critic/values/max": torch.max(values[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
                "critic/values/min": torch.min(values[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
            }
        )
    if "episode_rewards_norm" in batch.batch.keys():
        episode_rewards_norm = batch.batch["episode_rewards_norm"]
        step_rewards_norm = batch.batch["step_rewards_norm"]
        metrics.update({
            "critic/episode_rewards_norm/mean": episode_rewards_norm.mean().detach().item(),
            "critic/episode_rewards_norm/max": episode_rewards_norm.max().detach().item(),
            "critic/episode_rewards_norm/min": episode_rewards_norm.min().detach().item(),
            "critic/step_rewards_norm/mean": step_rewards_norm.mean().detach().item(),
            "critic/step_rewards_norm/max": step_rewards_norm.max().detach().item(),
            "critic/step_rewards_norm/min": step_rewards_norm.min().detach().item(),
        })
    return metrics

class GroupFilter:
    """
    User defined group filter.
    """
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        pass

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        """
        return True to filter out this group
        """
        return False
