# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Any, List

import ray
import torch

from roll.pipeline.base_worker import ActorWorker
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config

logger = get_logger()


class ComputeLogprobsPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)

        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.reference.model_args,
            template_name=self.pipeline_config.reference.data_args.template,
        )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)

    @torch.no_grad()
    def run(self):
        global_step = 0
        metric_list = []

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

        gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        gen_batch.meta_info = {"global_step": global_step}

        generate_output: DataProto = self.reference.generate(data=gen_batch)
        generate_output.meta_info.pop("metrics", {})
        batch.batch = generate_output.batch
        batch = batch.union(generate_output)

        ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
        ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
        ref_log_probs.meta_info.pop("metrics", {})
        batch = batch.union(ref_log_probs)

        prompt_ids = generate_output.batch["prompts"]
        response_ids = generate_output.batch["responses"]
        generate_res = []
        prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        for prompt, response, ref_log_prob in zip(prompts, responses, batch.batch["ref_log_probs"]):
            generate_res.append({"prompt": prompt, "response": response, "ref_log_prob": ref_log_prob.tolist()})
        metric_list.append(generate_res)

        logger.info("pipeline complete!")
        return metric_list


if __name__ == "__main__":
    init()

    ppo_config = make_baseline_config(config_path="./log_probs", config_name="log_probs_config")

    pipeline = ComputeLogprobsPipeline(ppo_config)
    metric_list = pipeline.run()

    output_file = "compute_log_probs.json"
    with open(output_file, "w") as f:
        json.dump(metric_list, f, ensure_ascii=False)
