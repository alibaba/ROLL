import json
import os.path
import random
import time
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.datasets.global_dataset import GlobalDatasetManager
from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.utils import (dump_rollout_render, compute_discounted_returns,
                                         compute_response_level_rewards, dump_rollout_trajectories, get_agentic_response_level_mask, agentic_compute_advantage)
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import (
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
    masked_mean,
    RunningMoments,
    compute_clip_fraction,
    agg_loss,
    compute_token_reward,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

logger = get_logger()


class AgenticPipeline(BasePipeline):
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

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
        self.download_models(*download_clusters)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        self.train_rollout_scheduler = ray.remote(RolloutScheduler).options(
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
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

        if self.pipeline_config.enable_reference:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
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
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)

                    ray.get(self.train_rollout_scheduler.suspend.remote())
                    if self.pipeline_config.async_generation_ratio > 0:
                        self.actor_infer.stop_server()

                    with Timer(name="model_update", logger=None) as model_update_timer:
                        model_update_metrics: Dict = self.model_update(global_step)
                    metrics["time/step_model_update"] =model_update_timer.last

                    metrics.update(model_update_metrics)
                    if self.pipeline_config.async_generation_ratio > 0:
                        self.actor_infer.start_server(data=DataProto(meta_info={"global_step": global_step, "is_offload_states": False}))
                    else:
                        self.actor_infer.start_server(data=DataProto(meta_info={"global_step": global_step, "is_offload_states": True}))

                    batch: DataProto = DataProto()
                    batch.meta_info = {"global_step": global_step}

                    if self.pipeline_config.eval_steps > 0 and global_step % self.pipeline_config.eval_steps == 0:
                        with Timer(name="val", logger=None) as val_timer:
                            metrics.update(self.val(global_step=global_step))
                        metrics["time/step_val"] = val_timer.last

                    with Timer(name="rollout", logger=None) as rollout_timer:
                        batch.meta_info["is_offload_states"] = True
                        batch = ray.get(self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                        sample_uuids = [f"{traj_id}_{i}" for i, traj_id in enumerate(batch.non_tensor_batch['traj_id'])]
                        batch.non_tensor_batch['sample_uuid'] = np.array(sample_uuids, dtype=object)
                        if "get_batch_return_start_time" in batch.meta_info:
                            metrics["time/get_batch_cost_train"] = time.time() - batch.meta_info.pop("get_batch_return_start_time")
                        actor_infer_metrics = self.actor_infer.get_metrics()
                        metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))

                        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

                    metrics["time/step_rollout"] = rollout_timer.last
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    batch.meta_info["global_step"] = global_step
                    if not (self.pipeline_config.async_generation_ratio > 0):
                        self.actor_infer.stop_server()

                    batch = compute_discounted_returns(batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)

                    batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                        if self.pipeline_config.enable_reference:
                            ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
                            ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                            ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                            batch = batch.union(ref_log_probs)
                            avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                            metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                            metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                    metrics["time/step_ref_log_probs_values_reward"] = cal_timer.last

                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        batch.meta_info["is_offload_states"] = False
                        if self.pipeline_config.enable_old_logprobs:
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

                    with Timer(name="train_timer", logger=None) as train_timer:
                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            # update actor
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                            metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                            metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                        tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    metrics["time/step_train"] = train_timer.last

                with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                    data_metrics = compute_data_metrics(batch=batch)

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
                            prompt_mask = group_batch.batch["prompt_mask"]
                            non_prompt_mask = torch.logical_not(group_batch.batch["prompt_mask"]) * group_batch.batch["attention_mask"]
                            input_ids = group_batch.batch["input_ids"]
                            prompt_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(prompt_mask)]
                            response_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(non_prompt_mask)]
                            prompts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=False)
                            responses = self.tokenizer.batch_decode(response_ids_list, skip_special_tokens=False)
                            episode_scores = group_batch.non_tensor_batch["episode_scores"].tolist()
                            step_scores = group_batch.non_tensor_batch["step_scores"].tolist()
                            if not isinstance(step_scores[0], float):
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

def compute_data_metrics(batch):
    # token_level_scores are per-token scores assigned by the reward model, possibly after normalization/clipping
    # score denotes the raw environment reward
    episode_scores = get_episode_scores(batch)
    try:
        traj_rollout_times = get_traj_rollout_time(batch)
        traj_env_times = get_traj_env_time(batch)
    except Exception as e:
        traj_rollout_times = torch.zeros(batch.batch.batch_size[0], dtype=torch.float32)
        traj_env_times = torch.zeros(batch.batch.batch_size[0], dtype=torch.float32)

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

        # # traj_rollout_time
        "env/traj_rollout_time/mean": torch.mean(traj_rollout_times).detach().item() if traj_rollout_times.numel() > 0 else 0.0,
        "env/traj_rollout_time/max": torch.max(traj_rollout_times).detach().item() if traj_rollout_times.numel() > 0 else 0.0,
        "env/traj_rollout_time/min": torch.min(traj_rollout_times).detach().item() if traj_rollout_times.numel() > 0 else 0.0,

        # traj_env_times
        "env/traj_env_time/mean": torch.mean(traj_env_times).detach().item() if traj_env_times.numel() > 0 else 0.0,
        "env/traj_env_time/max": torch.max(traj_env_times).detach().item() if traj_env_times.numel() > 0 else 0.0,
        "env/traj_env_time/min": torch.min(traj_env_times).detach().item() if traj_env_times.numel() > 0 else 0.0,

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
