import copy
from typing import Any, Dict

import torch
from codetiming import Timer

from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.base_worker import ActorWorker, InferWorker
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger

logger = get_logger()


class ModelUpdatePipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)

        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_train.model_args,
        )
        self.pipeline_config.set_max_steps(max_steps=1024)
        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=InferWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )

        actor_attn_worker_config: WorkerConfig = copy.deepcopy(self.pipeline_config.actor_infer)
        actor_attn_worker_config.strategy_args = StrategyArguments(strategy_name="hf_infer", strategy_config={})
        actor_attn_worker_config.name = "actor_attn"
        self.actor_attn: Any = Cluster(
            name=actor_attn_worker_config.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=actor_attn_worker_config,
        )

        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.actor_attn.initialize(pipeline_config=self.pipeline_config, blocking=True)

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_attn,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

    @torch.no_grad()
    def run(self):
        global_step = 0
        metric_list = []

        with Timer() as timer:
            model_update_metrics: Dict = self.model_update(global_step)
        model_update_metrics["time/model_update"] = timer.last

        metric_list.append(model_update_metrics)
        global_step += 1

        prompts = [
            "Compared with Google, Microsoft",
            "据悉，美国总统",
            "接天莲叶无穷碧，",
            "中国的首都是北京，而美国的",
            "Artificial intelligence is transforming industries such as",
            "在过去的十年中，科技的快速发展使得",
            "The Great Wall of China is a famous landmark that",
            "COVID-19 pandemic has impacted global economies, leading to",
            "Machine learning algorithms can improve efficiency in",
            "近年来，全球气候变化引发了人们对",
            "The exploration of Mars is a significant step for",
            "在文化交流中，中西方的差异让人们",
            "Sustainable energy sources are crucial for combating",
            "在文学的创作中，诗歌常常与",
            "The rise of social media has changed how we connect with",
            "科技在日常生活中扮演着越来越重要的角色，例如",
        ]

        batch_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)

        batch: DataProto = DataProto.from_single_dict(batch_dict)
        batch.batch["position_ids"] = torch.clip(
            torch.cumsum(batch.batch["attention_mask"], dim=-1) - 1, min=0, max=None
        )
        generate_output: DataProto = self.actor_infer.generate(batch)
        actor_attn_output: DataProto = self.actor_attn.generate(batch)
        prompt_ids = generate_output.batch["prompts"]
        response_ids = generate_output.batch["responses"]
        attn_response_ids = actor_attn_output.batch["responses"]
        generate_res = []
        prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        attn_responses = self.tokenizer.batch_decode(attn_response_ids, skip_special_tokens=True)
        for prompt, response, attn_response in zip(prompts, responses, attn_responses):
            generate_res.append({"prompt": prompt, "response": response, "attn_responses": attn_response})
        metric_list.append(generate_res)

        logger.info("pipeline complete!")
        return metric_list
