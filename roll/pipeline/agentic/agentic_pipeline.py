import json
import os.path
from typing import Any, Dict, List, Callable

import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer

from roll.agentic.rollout.rollout_scheduler import RolloutScheduler
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.utils import dump_rollout_render
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.functionals import (
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
    masked_mean,
    RunningMoments,
    compute_clip_fraction,
    agg_loss,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

logger = get_logger()


class AgenticPipeline(BasePipeline):
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)
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
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )

        self.train_rollout_scheduler = RolloutScheduler(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.train_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="train",
        )
        self.val_rollout_scheduler = RolloutScheduler(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.val_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="val",
        )
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

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
        # Calculate tokens per second system throughput
        tps_timer = _Timer(window_size=5)

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}
            with tps_timer:
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                model_update_metrics: Dict = self.model_update(global_step)
                metrics.update(model_update_metrics)

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}

                if global_step % self.pipeline_config.eval_steps == 0:
                    batch.meta_info["is_offload_states"] = False
                    eval_batch = self.val_rollout_scheduler.get_batch(batch, self.pipeline_config.val_batch_size)
                    eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
                    eval_score = eval_batch.batch["scores"].sum(-1)
                    eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
                    eval_metrics["score/max"] = torch.max(eval_score).detach().item()
                    eval_metrics["score/min"] = torch.min(eval_score).detach().item()
                    metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})

                    if self.pipeline_config.render_save_dir:
                        self.executor.submit(
                            dump_rollout_render,
                            save_dir=self.pipeline_config.render_save_dir,
                            step=global_step,
                            frames=eval_batch.non_tensor_batch["frames"],
                            env_ids=eval_batch.non_tensor_batch["env_ids"],
                            tags=eval_batch.non_tensor_batch["tags"],
                            episode_scores=eval_batch.non_tensor_batch["episode_scores"],
                        )
                    del eval_batch

                with Timer(name="rollout", logger=None) as rollout_timer:
                    batch.meta_info["is_offload_states"] = True
                    batch = self.train_rollout_scheduler.get_batch(batch, self.pipeline_config.rollout_batch_size)
                    batch.non_tensor_batch.pop("frames")
                metrics["time/rollout"] = rollout_timer.last
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                batch.meta_info["global_step"] = global_step

                with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                    ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
                    ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                    ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                    batch = batch.union(ref_log_probs)
                    avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                    metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                metrics["time/ref_log_probs_values_reward"] = cal_timer.last

                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    batch.meta_info["is_offload_states"] = False
                    old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))
                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})

                    agg_entropy = agg_loss(
                        loss_mat=old_log_probs.batch["entropy"],
                        loss_mask=batch.batch["response_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )
                    metrics.update({"critic/entropy/mean": agg_entropy.item()})

                    metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                metrics["time/old_log_probs_values"] = cal_old_logpb_timer.last

                # Need to process rewards by group
                # Can group by tag(env_type)/traj_group_id(group)/batch(rollout_batch)... to calculate reward/adv
                batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
                with Timer(name="adv", logger=None) as timer:
                    grouping = self.pipeline_config.reward_normalization.grouping
                    batch_grouped: Dict[str, DataProto] = {"default": batch}
                    if grouping != "batch":
                        batch_grouped = batch.group_by(keys=grouping)
                    batch_list = []
                    for group_name, group_batch in batch_grouped.items():
                        score_norm_fn = get_score_normalize_fn(rn_cfg=self.pipeline_config.reward_normalization)
                        scores: torch.Tensor = group_batch.batch["scores"].clone()
                        penalty: torch.Tensor = group_batch.batch["penalty"]
                        acc_scores = scores.sum(dim=-1)
                        normalized_acc_scores = acc_scores + penalty
                        normalized_acc_scores = score_norm_fn(normalized_acc_scores)
                        group_batch.batch["response_level_rewards"] = normalized_acc_scores
                        if self.pipeline_config.reward_clip:
                            reward_clip_frac = compute_clip_fraction(
                                values=group_batch.batch["response_level_rewards"],
                                clip_max=self.pipeline_config.reward_clip,
                                clip_min=-self.pipeline_config.reward_clip,
                            )
                            metrics["critic/reward_clip_frac"] = reward_clip_frac
                            group_batch.batch["response_level_rewards"] = torch.clamp(
                                group_batch.batch["response_level_rewards"],
                                min=-self.pipeline_config.reward_clip,
                                max=self.pipeline_config.reward_clip,
                            )

                        group_batch, kl_metrics = apply_kl_penalty(
                            data=group_batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty
                        )
                        batch_list.append(group_batch)
                    batch = DataProto.concat(batch_list)
                    batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                    batch.pop("prompt_id")
                    # Is advantage calculated globally across batch or within groups?
                    batch = compute_advantage(
                        data=batch,
                        gamma=self.pipeline_config.gamma,
                        lambd=self.pipeline_config.lambd,
                        adv_estimator=self.pipeline_config.adv_estimator,
                        advantage_clip=self.pipeline_config.advantage_clip,
                        whiten_advantages=self.pipeline_config.whiten_advantages,
                        whiten_rewards=self.pipeline_config.whiten_rewards,
                    )

                metrics.update(kl_metrics)
                metrics["time/adv"] = timer.last

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

            data_metrics = compute_data_metrics(batch=batch)
            metrics.update(data_metrics)
            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["system/samples"] = (global_step + 1) * batch.batch.shape[0]

            # do ckpt
            self.state.step = global_step
            self.state.log_history.append(metrics)

            self.do_checkpoint(global_step=global_step)

            self.tracker.log(values=metrics, step=global_step)

            if global_step % self.pipeline_config.logging_steps == 0:
                if int(os.environ.get("RAY_PROFILING", "0")):
                    timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                    os.makedirs(timeline_dir, exist_ok=True)
                    ray.timeline(
                        filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                    )

                prompt_mask = batch.batch["prompt_mask"]
                non_prompt_mask = batch.batch["non_prompt_mask"]
                input_ids = batch.batch["input_ids"]
                prompt_ids = torch.where(
                    prompt_mask.bool(), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id)
                )
                response_ids = torch.where(
                    non_prompt_mask.bool(), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id)
                )

                generate_res = []
                prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                episode_scores = batch.non_tensor_batch["episode_scores"].tolist()
                llm_raw_text_list = batch.non_tensor_batch["llm_raw_text_list"].tolist()
                for prompt, prompt_id, response, response_id, episode_score, llm_raw_text in zip(
                    prompts, prompt_ids, responses, response_ids, episode_scores, llm_raw_text_list
                ):
                    generate_res.append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "episode_score": episode_score,
                            "llm_raw_text": llm_raw_text,
                        }
                    )
                logger.info(json.dumps(generate_res[:10], ensure_ascii=False))
                logger.info(json.dumps(metrics, ensure_ascii=False))

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
            logger.info(f"epoch {global_step} finished")
        logger.info("pipeline complete!")


def compute_data_metrics(batch):
    # token_level_scores are scores given by reward model to each token, possibly after norm/clip
    # score is the environment reward, raw value
    sequence_score = batch.batch["scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]
    # fix: https://github.com/volcengine/verl/pull/60
    prompt_mask = batch.batch["prompt_mask"].bool()
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    prompt_lengths = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    returns = batch.batch["returns"]

    metrics = {
        # score, sequence_score from env
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": masked_mean(advantages, response_mask).detach().item(),
        "critic/advantages/max": torch.max(advantages[response_mask]).detach().item(),
        "critic/advantages/min": torch.min(advantages[response_mask]).detach().item(),
        # returns
        "critic/returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/returns/max": torch.max(returns[response_mask]).detach().item(),
        "critic/returns/min": torch.min(returns[response_mask]).detach().item(),
        # response length
        "tokens/response_length/mean": torch.mean(response_length).detach().item(),
        "tokens/response_length/max": torch.max(response_length).detach().item(),
        "tokens/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "tokens/prompt_length/mean": torch.mean(prompt_lengths).detach().item(),
        "tokens/prompt_length/max": torch.max(prompt_lengths).detach().item(),
        "tokens/prompt_length/min": torch.min(prompt_lengths).detach().item(),
    }
    if "values" in batch.batch.keys():
        values = batch.batch["values"]
        # values
        metrics.update(
            {
                "critic/values/mean": masked_mean(values, response_mask).detach().item(),
                "critic/values/max": torch.max(values[response_mask]).detach().item(),
                "critic/values/min": torch.min(values[response_mask]).detach().item(),
            }
        )
    return metrics


def get_score_normalize_fn(rn_cfg) -> Callable:
    grouping, method = rn_cfg.grouping, rn_cfg.method
    if method == "mean_std":
        norm_func = lambda x: (
            (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            if x.std(dim=-1, keepdim=True).abs().max() > 1e-6
            else torch.zeros_like(x)
        )  # stable to bf16 than x.std()
    elif method == "mean":
        norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
    elif method == "asym_clip":
        norm_func = lambda x: (
            (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            if x.std(dim=-1, keepdim=True).abs().max() > 1e-6
            else torch.zeros_like(x)
        ).clamp(min=-1, max=3)
    elif method == "identity":
        norm_func = lambda x: x
    else:
        raise ValueError(f"Invalid normalization method: {method}")

    return norm_func
