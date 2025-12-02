import json
import os.path
import time
from itertools import count
from typing import Any

import ray
import torch
from codetiming import Timer

from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.utils import dump_rollout_trajectories
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.functionals import (
    reduce_metrics,
)
from roll.utils.logging import get_logger

logger = get_logger()


class AgenticRolloutPipeline(BasePipeline):
    """
    this is just for env rollout
    """
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        self.use_policy_model = self.pipeline_config.train_env_manager.llm_proxy.proxy_type == "policy"

        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.download_models(self.actor_infer)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        self.rollout_scheduler = ray.remote(RolloutScheduler).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.train_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="train",
        )

        if self.use_policy_model:
            self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

    @torch.no_grad()
    def run(self):

        for global_step in (count() if self.pipeline_config.max_steps == -1 else range(self.pipeline_config.max_steps)):
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}
            batch: DataProto = DataProto()
            batch.meta_info = {"global_step": global_step}

            with Timer(name="rollout", logger=None) as rollout_timer:
                if self.use_policy_model:
                    batch.meta_info["is_offload_states"] = True
                    self.actor_infer.start_server(data=batch)
                batch = ray.get(self.rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                if batch is None:
                    break

                if "get_batch_return_start_time" in batch.meta_info:
                    metrics["time/get_batch_cost_train"] = time.time() - batch.meta_info.pop("get_batch_return_start_time")
                actor_infer_metrics: DataProto = self.actor_infer.get_metrics()
                metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))

            metrics["time/step_rollout"] = rollout_timer.last
            eval_metrics = reduce_metrics(batch.meta_info.get("metrics", {}))
            eval_score = batch.batch["scores"].sum(-1)
            eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
            eval_metrics["score/max"] = torch.max(eval_score).detach().item()
            eval_metrics["score/min"] = torch.min(eval_score).detach().item()

            batch_grouped = batch.group_by(keys="tags")
            for group_name, group_batch in batch_grouped.items():
                eval_score = group_batch.batch["scores"].sum(-1)
                eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
                eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
                eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()
                group_eval_metrics = reduce_metrics(group_batch.meta_info.get("metrics", {}))
                eval_metrics.update({f"{group_name}/{k}": v for k, v in group_eval_metrics.items()})

            metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
            batch.meta_info["global_step"] = global_step
            metrics["system/samples"] = (global_step + 1) * batch.batch.shape[0]

            self.tracker.log(values=metrics, step=global_step)

            dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

            if global_step % self.pipeline_config.logging_steps == 0:
                if int(os.environ.get("RAY_PROFILING", "0")):
                    timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                    os.makedirs(timeline_dir, exist_ok=True)
                    ray.timeline(
                        filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                    )

                prompt_mask = batch.batch["prompt_mask"]
                non_prompt_mask = torch.logical_not(batch.batch["prompt_mask"])
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
                for prompt, prompt_id, response, response_id, episode_score in zip(
                    prompts, prompt_ids, responses, response_ids, episode_scores
                ):
                    generate_res.append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "episode_score": episode_score,
                        }
                    )
                logger.info(json.dumps(generate_res[:10], ensure_ascii=False))
                logger.info(json.dumps(metrics, ensure_ascii=False))

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
        ray.get(self.rollout_scheduler.shutdown.remote())
        logger.info("pipeline complete!")
