import json
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.datasets.loader import get_dataset
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.base_worker import ActorWorker
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config


logger = get_logger()


class LogProbsCmpPipeline(BasePipeline):

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        set_seed(self.pipeline_config.seed)
        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_train.model_args,
        )
        self.dataset = get_dataset(
            tokenizer=self.tokenizer,
            data_args=self.pipeline_config.actor_train.data_args,
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
        max_steps = len(self.dataloader) * self.pipeline_config.actor_train.training_args.num_train_epochs
        self.pipeline_config.set_max_steps(max_steps=max_steps)
        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
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
        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)

    def run(self):
        global_step = 0

        results = []
        for batch_dict in tqdm(self.dataloader):
            logger.info(f"pipeline step {global_step} start...")

            batch_dict: Dict
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.meta_info = {"global_step": global_step}

            gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
            gen_batch.meta_info = {"global_step": global_step}
            generate_output: DataProto = self.actor_infer.generate(data=gen_batch)

            batch.batch = generate_output.batch
            batch = batch.union(generate_output)

            batch.meta_info["disable_adapter"] = True
            logprobs_train = self.actor_train.compute_log_probs(batch)
            batch.meta_info["disable_adapter"] = False
            logprobs_train_adapter = self.actor_train.compute_log_probs(batch)
            logprobs_hf = self.actor_infer.compute_log_probs(batch)
            logprobs_ref = self.reference.compute_log_probs(batch)

            prompt_ids = generate_output.batch["prompts"]
            prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            for prompt, logprobs_train, logprobs_train_adapter, logprob_hf, logprobs_ref in zip(
                prompts,
                logprobs_train.batch["log_probs"],
                logprobs_train_adapter.batch["log_probs"],
                logprobs_hf.batch["log_probs"],
                logprobs_ref.batch["log_probs"],
            ):
                result = {
                    "prompt": prompt,
                    "logprobs_train": logprobs_train.tolist(),
                    "logprobs_train_adapter": logprobs_train_adapter.tolist(),
                    "logprob_hf": logprob_hf.tolist(),
                    "logprobs_ref": logprobs_ref.tolist(),
                }
                print(result)
                assert torch.allclose(logprobs_train, logprobs_ref)
                results.append(result)

        return results


if __name__ == "__main__":
    ppo_config = make_baseline_config(config_path="./log_probs", config_name="log_probs_lora_cmp_config")

    init()

    pipeline = LogProbsCmpPipeline(ppo_config)
    results = pipeline.run()

    output_file = "logprobs_cmp.json"
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False)
