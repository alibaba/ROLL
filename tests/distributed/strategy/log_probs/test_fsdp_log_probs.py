import json
import os
import time
from typing import Any, Dict

import pytest
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
from roll.platforms import current_platform
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config

logger = get_logger()


def _available_ray_nodes_for_config(num_gpus_per_node: int) -> int:
    """Count Ray nodes that have enough devices for the configured per-node requirement."""
    count = 0
    for node in ray.nodes():
        if int(node["Resources"].get(current_platform.ray_device_key, 0)) >= num_gpus_per_node:
            count += 1
    return count


def _skip_if_cluster_insufficient(config: RLVRConfig, test_name: str) -> None:
    required_nodes = getattr(config, "num_nodes", None)
    required_gpus_per_node = getattr(config, "num_gpus_per_node", 1)
    if required_nodes is None:
        return
    available_nodes = _available_ray_nodes_for_config(required_gpus_per_node)
    if available_nodes < required_nodes:
        pytest.skip(
            f"{test_name} requires {required_nodes} Ray nodes with >= {required_gpus_per_node} "
            f"{current_platform.ray_device_key} each, but only {available_nodes} available in CI."
        )


def _reset_model_download_cache_actor() -> None:
    from roll.utils import checkpoint_manager

    checkpoint_manager.shared_storage = None


def _looks_like_local_path(path: str) -> bool:
    return (
        os.path.isabs(path)
        or path.startswith((".", "~"))
        or "\\" in path
        or path.count("/") != 1
    )


def _skip_if_local_model_unavailable(config: RLVRConfig, test_name: str) -> None:
    model_paths = set()
    for config_name in ("actor_train", "actor_infer", "reference"):
        worker_config = getattr(config, config_name, None)
        model_args = getattr(worker_config, "model_args", None)
        model_path = getattr(model_args, "model_name_or_path", None)
        if model_path:
            model_paths.add(str(model_path))

    for model_path in sorted(model_paths):
        if _looks_like_local_path(model_path) and not os.path.isdir(os.path.expanduser(model_path)):
            pytest.skip(f"{test_name} requires local model path {model_path}, but it is not available on this CI node.")


def _data_files_exist(data_args) -> bool:
    file_names = getattr(data_args, "file_name", None)
    if file_names is None:
        return False
    if isinstance(file_names, str):
        file_names = [file_names]

    dataset_dir = os.path.expanduser(str(getattr(data_args, "dataset_dir", ".") or "."))
    for file_name in file_names:
        path = os.path.expanduser(str(file_name))
        if os.path.exists(path):
            continue
        if os.path.exists(os.path.join(dataset_dir, path)):
            continue
        return False
    return True


def _make_synthetic_vlm_dataset(processor, size: int = 2):
    from PIL import Image
    from torch.utils.data import Dataset

    from roll.pipeline.rlvr.rlvr_vlm_pipeline import format_prompt

    prompt = format_prompt("What color is the image?", processor, use_image=True)
    size = max(1, int(size))

    class SyntheticVLMDataset(Dataset):
        def __len__(self):
            return size

        def __getitem__(self, index):
            return {
                "images": [Image.new("RGB", (64, 64), (255, 255, 255))],
                "prompt": prompt,
                "ground_truth": "white",
                "image_flag": True,
                "tag": "synthetic",
            }

    return SyntheticVLMDataset()


def _cleanup_pipeline(pipeline) -> None:
    for cluster_name in ("actor_train", "actor_infer", "reference"):
        cluster = getattr(pipeline, cluster_name, None)
        for worker in getattr(cluster, "workers", []) or []:
            ray.kill(worker, no_restart=True)
    resource_manager = getattr(pipeline, "resource_manager", None)
    if resource_manager is not None:
        resource_manager.destroy_placement_group()
    time.sleep(1)


def _run_pipeline_and_cleanup(pipeline, *args, **kwargs):
    try:
        return pipeline.run(*args, **kwargs)
    finally:
        _cleanup_pipeline(pipeline)


class FSDPLogProbsPipeline(BasePipeline):
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
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )

        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
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
            batch.meta_info = {"global_step": global_step, "loss_mask_keys": ["response_mask"]}
            batch.batch["response_mask"] = batch.batch["attention_mask"].clone()

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

            # These tests validate logprob computation, not generation. Use the
            # collated token sequence directly to avoid depending on vLLM startup.
            prompt_ids = batch.batch["input_ids"]
            response_ids = batch.batch["input_ids"]
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
            break

        logger.info("pipeline complete!")
        return results


def test_fsdp_log_probs_full():
    init()
    _reset_model_download_cache_actor()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_config")
    _skip_if_cluster_insufficient(config, "test_fsdp_log_probs_full")
    pipeline = FSDPLogProbsPipeline(config)
    results = _run_pipeline_and_cleanup(pipeline)

    output_file = "test_fsdp_log_probs_full.json"
    with open(output_file, "w") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Test FSDP (full) completed, results saved to {output_file}")


def test_fsdp_log_probs_lora():
    init()
    _reset_model_download_cache_actor()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_lora_config")
    _skip_if_cluster_insufficient(config, "test_fsdp_log_probs_lora")
    pipeline = FSDPLogProbsPipeline(config)
    results = _run_pipeline_and_cleanup(pipeline)

    output_file = "test_fsdp_log_probs_lora.json"
    with open(output_file, "w") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Test FSDP (LoRA) completed, results saved to {output_file}")


def test_fsdp_log_probs_cp():
    init()
    _reset_model_download_cache_actor()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_cp_config")
    _skip_if_cluster_insufficient(config, "test_fsdp_log_probs_cp")

    device_count = current_platform.device_count()
    if device_count < 8:
        pytest.skip(f"Need at least 8 {current_platform.ray_device_key} devices, got {device_count}.")

    pipeline = FSDPLogProbsPipeline(config)
    results = _run_pipeline_and_cleanup(pipeline)

    output_file = "test_fsdp_log_probs_cp.json"
    with open(output_file, "w") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Test FSDP (CP) completed, results saved to {output_file}")


def test_fsdp_log_probs_cp_rmpad():
    init()
    _reset_model_download_cache_actor()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_cp_rmpad_config")
    _skip_if_cluster_insufficient(config, "test_fsdp_log_probs_cp_rmpad")
    pipeline = FSDPLogProbsPipeline(config)
    results = _run_pipeline_and_cleanup(pipeline)

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
