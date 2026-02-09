import json
from typing import Any, Dict

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
from roll.pipeline.base_worker import ActorWorker, InferWorker
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config

logger = get_logger()


class TestFSDPLogProbsPipeline(BasePipeline):
    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)

        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_train.model_args,
        )

        # Load dataset
        self.dataset = get_dataset(
            tokenizer=self.tokenizer,
            data_args=self.pipeline_config.actor_train.data_args,
        )

        # Create data collator
        data_collator = DataCollatorWithPaddingForPaddedKeys(
            tokenizer=self.tokenizer,
            max_length=self.pipeline_config.prompt_length,
            padding="max_length",
        )

        # Create dataloader
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.pipeline_config.rollout_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
        )

        max_steps = len(self.dataloader) * self.pipeline_config.actor_train.training_args.num_train_epochs
        self.pipeline_config.set_max_steps(max_steps=max_steps)

        # Initialize clusters
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
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )

        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)

    @torch.no_grad()
    def run(self):
        """
        Compare log probs between FSDP2 strategy and HF reference implementation.
        Similar to test_ds_hf_log_probs.py logic.
        """
        global_step = 0
        results = []

        for batch_dict in tqdm(self.dataloader):
            logger.info(f"pipeline step {global_step} start...")

            batch_dict: Dict
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.meta_info = {"global_step": global_step}

            # Generate responses using actor_infer
            gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
            gen_batch.meta_info = {"global_step": global_step}
            generate_output: DataProto = self.actor_infer.generate(data=gen_batch)

            # Combine generated output with original batch
            batch.batch = generate_output.batch
            batch = batch.union(generate_output)

            if self.pipeline_config.actor_train.model_args.lora_target is not None:
                batch.meta_info["disable_adapter"] = True
                logprobs_fsdp_disable_adapter = self.actor_train.compute_log_probs(batch)
                batch.meta_info["disable_adapter"] = False
                logprobs_fsdp_enable_adapter = self.actor_train.compute_log_probs(batch)
                logprobs_fsdp = logprobs_fsdp_enable_adapter
            else:
                logprobs_fsdp = self.actor_train.compute_log_probs(batch)
                logprobs_fsdp_disable_adapter = None
                logprobs_fsdp_enable_adapter = None

            # Compute log probs from reference (should also use HF)
            logprobs_ref = self.reference.compute_log_probs(batch)

            # Extract prompt and response for logging
            prompt_ids = generate_output.batch["prompts"]
            response_ids = generate_output.batch["responses"]
            prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            # Compare FSDP vs HF and FSDP vs Reference
            count = 0
            sum_diff_max = 0.0
            sum_diff_mean = 0.0

            # Statistics for adapter enable/disable comparison
            sum_diff_adapter_enable_disable_max = 0.0
            sum_diff_adapter_enable_disable_mean = 0.0
            count_adapter = 0

            # Statistics for FSDP vs HF comparison
            sum_diff_fsdp_hf_max = 0.0
            sum_diff_fsdp_hf_mean = 0.0
            count_fsdp_hf = 0

            # Prepare logprobs lists
            logprobs_fsdp_list = logprobs_fsdp.batch["log_probs"]
            logprobs_ref_list = logprobs_ref.batch["log_probs"]

            # Prepare adapter logprobs if available
            logprobs_fsdp_enable_list = None
            logprobs_fsdp_disable_list = None
            if logprobs_fsdp_enable_adapter is not None and logprobs_fsdp_disable_adapter is not None:
                logprobs_fsdp_enable_list = logprobs_fsdp_enable_adapter.batch["log_probs"]
                logprobs_fsdp_disable_list = logprobs_fsdp_disable_adapter.batch["log_probs"]

            for idx, (prompt, response, logprob_fsdp, logprob_ref) in enumerate(
                zip(
                    prompts,
                    responses,
                    logprobs_fsdp_list,
                    logprobs_ref_list,
                )
            ):
                # Compare FSDP (with adapter enabled) vs FSDP (with adapter disabled)
                if logprobs_fsdp_enable_list is not None and logprobs_fsdp_disable_list is not None:
                    logprob_enable = logprobs_fsdp_enable_list[idx]
                    logprob_disable = logprobs_fsdp_disable_list[idx]
                    diff_adapter_max = (logprob_enable - logprob_disable).abs().max().item()
                    diff_adapter_mean = (logprob_enable - logprob_disable).abs().mean().item()
                    sum_diff_adapter_enable_disable_max += diff_adapter_max
                    sum_diff_adapter_enable_disable_mean += diff_adapter_mean
                    count_adapter += 1
                    adapter_diff_max = diff_adapter_max
                    adapter_diff_mean = diff_adapter_mean
                else:
                    adapter_diff_max = None
                    adapter_diff_mean = None

                # Compare FSDP vs HF (if both have values)
                if logprob_fsdp is not None and logprob_ref is not None:
                    diff_fsdp_hf_max = (logprob_fsdp - logprob_ref).abs().max().item()
                    diff_fsdp_hf_mean = (logprob_fsdp - logprob_ref).abs().mean().item()
                    sum_diff_fsdp_hf_max += diff_fsdp_hf_max
                    sum_diff_fsdp_hf_mean += diff_fsdp_hf_mean
                    count_fsdp_hf += 1
                else:
                    diff_fsdp_hf_max = None
                    diff_fsdp_hf_mean = None

                # Original comparison (FSDP vs HF, kept for backward compatibility)
                diff_max = diff_fsdp_hf_max if diff_fsdp_hf_max is not None else 0.0
                diff_mean = diff_fsdp_hf_mean if diff_fsdp_hf_mean is not None else 0.0
                sum_diff_max += diff_max
                sum_diff_mean += diff_mean
                count += 1

                result = {
                    "prompt": prompt,
                    "response": response,
                    "diff_max": diff_max,
                    "diff_mean": diff_mean,
                    "logprob_fsdp": logprob_fsdp.tolist(),
                    "logprob_ref": logprob_ref.tolist(),
                }

                # Add adapter comparison if available
                if adapter_diff_max is not None:
                    result["diff_adapter_enable_disable_max"] = adapter_diff_max
                    result["diff_adapter_enable_disable_mean"] = adapter_diff_mean

                # Add explicit FSDP vs HF comparison if available
                if diff_fsdp_hf_max is not None:
                    result["diff_fsdp_hf_max"] = diff_fsdp_hf_max
                    result["diff_fsdp_hf_mean"] = diff_fsdp_hf_mean

                results.append(result)

            # Log statistics
            if count > 0:
                logger.info(f"avg_diff_max: {sum_diff_max / count}, avg_diff_mean: {sum_diff_mean / count}")

            if count_adapter > 0:
                logger.info(
                    f"avg_diff_adapter_enable_disable_max: {sum_diff_adapter_enable_disable_max / count_adapter}, "
                    f"avg_diff_adapter_enable_disable_mean: {sum_diff_adapter_enable_disable_mean / count_adapter}"
                )

            if count_fsdp_hf > 0:
                logger.info(
                    f"avg_diff_fsdp_hf_max: {sum_diff_fsdp_hf_max / count_fsdp_hf}, "
                    f"avg_diff_fsdp_hf_mean: {sum_diff_fsdp_hf_mean / count_fsdp_hf}"
                )
            global_step += 1

        logger.info("pipeline complete!")
        return results


def test_fsdp_log_probs_full():
    init()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_config")
    pipeline = TestFSDPLogProbsPipeline(config)
    results = pipeline.run()

    output_file = "test_fsdp_log_probs_full.json"
    with open(output_file, "w") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Test FSDP (full) completed, results saved to {output_file}")


def test_fsdp_log_probs_lora():
    init()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_lora_config")
    pipeline = TestFSDPLogProbsPipeline(config)
    results = pipeline.run()

    output_file = "test_fsdp_log_probs_lora.json"
    with open(output_file, "w") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Test FSDP (LoRA) completed, results saved to {output_file}")


def test_fsdp_log_probs_cp():
    init()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_cp_config")

    if torch.cuda.device_count() < 8:
        logger.warning(f"Skipping CP test, need at least 8 GPUs (have {torch.cuda.device_count()})")
        return

    pipeline = TestFSDPLogProbsPipeline(config)
    results = pipeline.run()

    output_file = "test_fsdp_log_probs_cp.json"
    with open(output_file, "w") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Test FSDP (CP) completed, results saved to {output_file}")


def test_fsdp_log_probs_cp_rmpad():
    init()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_cp_rmpad_config")
    pipeline = TestFSDPLogProbsPipeline(config)
    results = pipeline.run()

    output_file = "test_fsdp_log_probs_cp_rmpad.json"
    with open(output_file, "w") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Test FSDP (CP+RMpad) completed, results saved to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "full":
            test_fsdp_log_probs_full()
        elif test_name == "lora":
            test_fsdp_log_probs_lora()
        elif test_name == "cp":
            test_fsdp_log_probs_cp()
        elif test_name == "cp_rmpad":
            test_fsdp_log_probs_cp_rmpad()
        else:
            logger.error(f"Unknown test: {test_name}. Use 'full', 'lora', or 'cp'.")
    else:
        test_fsdp_log_probs_full()
