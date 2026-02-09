import json
from typing import Any, Dict, List

import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.datasets.loader import get_dataset
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.base_worker import ActorWorker
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import \
    make_baseline_config

logger = get_logger()


class ComputeLogprobsPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.reference.model_args,
        )
        self.dataset = get_dataset(
            tokenizer=self.tokenizer,
            data_args=self.pipeline_config.actor_infer.data_args,
        )
        data_collator = DataCollatorWithPaddingForPaddedKeys(
            tokenizer=self.tokenizer,
            max_length=self.pipeline_config.prompt_length,
            padding="max_length",
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.pipeline_config.rollout_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

    @torch.no_grad()
    def run(self):
        global_step = 0
        results = []

        for batch_dict in tqdm(self.dataloader):
            logger.info(f"pipeline step {global_step} start...")

            batch_dict: Dict
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
            gen_batch.meta_info = {"global_step": global_step}

            generate_output: DataProto = self.actor_infer.generate(data=gen_batch)
            generate_output.meta_info.pop("metrics", {})
            batch.batch = generate_output.batch
            batch = batch.union(generate_output)

            # TODO: megatron 和 hf logprobs计算结果差异很大，中间应该有问题，待后续排查
            ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
            ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
            ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
            if "entropy" in ref_log_probs.batch.keys():
                del ref_log_probs.batch["entropy"]
            ref_log_probs.meta_info.pop("metrics", {})
            batch = batch.union(ref_log_probs)

            hf_log_probs: DataProto = self.actor_infer.compute_log_probs(batch)
            hf_log_probs.rename(old_keys="log_probs", new_keys="hf_log_probs")
            if "entropy" in hf_log_probs.batch.keys():
                del hf_log_probs.batch["entropy"]
            hf_log_probs.meta_info.pop("metrics", {})
            batch = batch.union(hf_log_probs)
            response_mask = batch.batch["response_mask"]

            prompt_ids = generate_output.batch["prompts"]
            response_ids = generate_output.batch["responses"]
            prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            count = 0
            sum_diff_max = 0.0
            sum_diff_mean = 0.0
            for prompt, response, ref_log_prob, hf_log_prob, one_response_mask, attn_mask in zip(
                prompts,
                responses,
                batch.batch["ref_log_probs"],
                batch.batch["hf_log_probs"],
                response_mask,
                batch.batch["attention_mask"],
            ):
                diff_mean = (ref_log_prob - hf_log_prob).abs().sum().item() / one_response_mask.sum().item()
                diff_max = (ref_log_prob - hf_log_prob).abs().max().item()
                sum_diff_max += diff_max
                sum_diff_mean += diff_mean
                count += 1
                results.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "diff_max": diff_max,
                        "diff_mean": diff_mean,
                        "ref_log_prob": ref_log_prob.tolist(),
                        "hf_log_prob": hf_log_prob.tolist(),
                        "attn_mask": attn_mask.tolist(),
                    }
                )
            logger.info(f"avg_diff_max: {sum_diff_max / count}, avg_diff_mean: {sum_diff_mean / count}")
            diff_max = (batch.batch["ref_log_probs"] - batch.batch["hf_log_probs"]).abs().max()
            diff_mean = (batch.batch["ref_log_probs"] - batch.batch["hf_log_probs"]).abs().sum() / response_mask[
                :, 1:
            ].sum()
            logger.info(f"diff_max: {diff_max}, diff_mean: {diff_mean}")

        logger.info("pipeline complete!")
        return results


if __name__ == "__main__":
    init()

    ppo_config = make_baseline_config(config_path="./log_probs", config_name="log_probs_megatron_config")

    pipeline = ComputeLogprobsPipeline(ppo_config)
    metric_list = pipeline.run()

    output_file = "compute_log_probs_megatron.json"
    with open(output_file, "w") as f:
        for m in metric_list:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
