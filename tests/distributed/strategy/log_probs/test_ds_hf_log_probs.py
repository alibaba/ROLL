import json
from typing import Any, Dict

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
from tests.distributed.strategy.make_baseline_config import \
    make_baseline_config

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

            logprobs_zero3_ne = self.actor_train.compute_log_probs(batch)
            logprobs_hf = self.actor_infer.compute_log_probs(batch)
            logprobs_zero3_eq = self.reference.compute_log_probs(batch)

            prompt_ids = generate_output.batch["prompts"]
            response_ids = generate_output.batch["responses"]
            prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            # Compute per-sample differences
            count = 0
            sum_diff_zero3ne_hf_max = 0.0
            sum_diff_zero3ne_hf_mean = 0.0
            sum_diff_zero3eq_hf_max = 0.0
            sum_diff_zero3eq_hf_mean = 0.0
            
            for prompt, response, logprob_zero3_ne, logprob_hf, logprob_zero3_eq in zip(
                prompts,
                responses,
                logprobs_zero3_ne.batch["log_probs"],
                logprobs_hf.batch["log_probs"],
                logprobs_zero3_eq.batch["log_probs"],
            ):
                # Compute differences
                diff_zero3ne_hf_max = (logprob_zero3_ne - logprob_hf).abs().max().item()
                diff_zero3ne_hf_mean = (logprob_zero3_ne - logprob_hf).abs().mean().item()
                diff_zero3eq_hf_max = (logprob_zero3_eq - logprob_hf).abs().max().item()
                diff_zero3eq_hf_mean = (logprob_zero3_eq - logprob_hf).abs().mean().item()
                
                sum_diff_zero3ne_hf_max += diff_zero3ne_hf_max
                sum_diff_zero3ne_hf_mean += diff_zero3ne_hf_mean
                sum_diff_zero3eq_hf_max += diff_zero3eq_hf_max
                sum_diff_zero3eq_hf_mean += diff_zero3eq_hf_mean
                count += 1
                
                result = {
                    "prompt": prompt,
                    "response": response,
                    "diff_zero3ne_hf_max": diff_zero3ne_hf_max,
                    "diff_zero3ne_hf_mean": diff_zero3ne_hf_mean,
                    "diff_zero3eq_hf_max": diff_zero3eq_hf_max,
                    "diff_zero3eq_hf_mean": diff_zero3eq_hf_mean,
                    "logprob_zero3_ne": logprob_zero3_ne.tolist(),
                    "logprob_hf": logprob_hf.tolist(),
                    "logprob_zero3_eq": logprob_zero3_eq.tolist(),
                }
                results.append(result)
            
            # Log average differences for this batch
            logger.info(
                f"Batch {global_step} - ZeRO3(ne) vs HF: "
                f"avg_diff_max={sum_diff_zero3ne_hf_max / count:.6f}, "
                f"avg_diff_mean={sum_diff_zero3ne_hf_mean / count:.6f}"
            )
            logger.info(
                f"Batch {global_step} - ZeRO3(eq) vs HF: "
                f"avg_diff_max={sum_diff_zero3eq_hf_max / count:.6f}, "
                f"avg_diff_mean={sum_diff_zero3eq_hf_mean / count:.6f}"
            )
            
            global_step += 1

        logger.info("pipeline complete!")
        
        # Compute and log overall statistics
        if results:
            overall_zero3ne_hf_max = sum(r["diff_zero3ne_hf_max"] for r in results) / len(results)
            overall_zero3ne_hf_mean = sum(r["diff_zero3ne_hf_mean"] for r in results) / len(results)
            overall_zero3eq_hf_max = sum(r["diff_zero3eq_hf_max"] for r in results) / len(results)
            overall_zero3eq_hf_mean = sum(r["diff_zero3eq_hf_mean"] for r in results) / len(results)
            
            logger.info("=" * 80)
            logger.info("Overall Statistics:")
            logger.info(
                f"  ZeRO3(ne) vs HF: avg_diff_max={overall_zero3ne_hf_max:.6f}, "
                f"avg_diff_mean={overall_zero3ne_hf_mean:.6f}"
            )
            logger.info(
                f"  ZeRO3(eq) vs HF: avg_diff_max={overall_zero3eq_hf_max:.6f}, "
                f"avg_diff_mean={overall_zero3eq_hf_mean:.6f}"
            )
            logger.info("=" * 80)
        
        return results


if __name__ == "__main__":
    ppo_config = make_baseline_config(config_path="./log_probs", config_name="log_probs_cmp_config")

    init()

    pipeline = LogProbsCmpPipeline(ppo_config)
    results = pipeline.run()

    output_file = "logprobs_cmp.json"
    with open(output_file, "w") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    
    logger.info(f"Results saved to {output_file}")
