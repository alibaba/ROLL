import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Apply model patches before importing anything that uses the model
os.environ["AUTO_APPLY_MODEL_PATCHES"] = "1"
from tests.distributed.strategy.log_probs.apply_model_patch import (
    apply_qwen3vl_megatron_patches,
    apply_qwen3vl_patches,
)

apply_qwen3vl_patches()
apply_qwen3vl_megatron_patches()

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider, get_extra_data_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.base_worker import ActorWorker, InferWorker
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger
from tests.distributed.strategy.log_probs.analyze_layer_divergence import analyze_divergence
from tests.distributed.strategy.make_baseline_config import make_baseline_config

logger = get_logger()


def _actorworker_set_capture_env(self, env: Dict[str, str]):
    """
    Test-only helper executed inside Ray workers.
    - Sets capture env vars used by `layer_states_capture.py`
    - Ensures model patches are applied inside the worker process (not just the driver)
    """
    for k, v in env.items():
        os.environ[k] = str(v)
    # Apply patches inside the worker process so FSDP2/HF forwards get instrumented.
    try:
        from tests.distributed.strategy.log_probs.apply_model_patch import (
            apply_qwen3vl_megatron_patches,
            apply_qwen3vl_patches,
        )

        apply_qwen3vl_patches()
        apply_qwen3vl_megatron_patches()
    except Exception:
        pass


# Monkeypatch onto ActorWorker so we can call it on Ray actors from this test.
setattr(ActorWorker, "set_capture_env", _actorworker_set_capture_env)


def _set_capture_env_on_cluster(cluster: Cluster, save_dir: Path, prefix: str, step: int, batch_idx: int):
    env = {
        "LAYER_STATES_SAVE_DIR": str(save_dir),
        "LAYER_STATES_PREFIX": str(prefix),
        "LAYER_STATES_STEP": str(step),
        "LAYER_STATES_BATCH": str(batch_idx),
    }
    ray.get([w.set_capture_env.remote(env) for w in cluster.workers])


def _as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def save_inputs_and_embeddings(data: DataProto, save_dir: Path, prefix: str, global_step: int, batch_idx: int = 0):
    """Save input tensors for comparison."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save input_ids, attention_mask, position_ids
    for key in ["input_ids", "attention_mask", "position_ids", "response_mask"]:
        if key in data.batch:
            save_path = save_dir / f"{prefix}_step{global_step}_batch{batch_idx}_{key}.pt"
            torch.save(data.batch[key].cpu().detach(), save_path)

    # Save multi_modal_data if present
    if "multi_modal_data" in data.non_tensor_batch:
        mm_data = data.non_tensor_batch["multi_modal_data"]
        save_path = save_dir / f"{prefix}_step{global_step}_batch{batch_idx}_multi_modal_data.json"
        mm_metadata = {}
        if isinstance(mm_data, (list, tuple)):
            for i, sample_mm in enumerate(mm_data):
                if isinstance(sample_mm, dict):
                    for k, v in sample_mm.items():
                        if isinstance(v, torch.Tensor):
                            key_name = f"sample{i}_{k}"
                            mm_metadata[key_name] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                            tensor_path = save_dir / f"{prefix}_step{global_step}_batch{batch_idx}_mm_{key_name}.pt"
                            torch.save(v.cpu().detach(), tensor_path)
        with open(save_path, "w") as f:
            json.dump(mm_metadata, f, indent=2)


class TestFSDPVLMLayerStatesPipeline(BasePipeline):
    def __init__(self, pipeline_config: RLVRConfig, output_dir: str = "./layer_states_output"):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)
        if self.processor is None:
            raise RuntimeError("VLM layer states test requires a processor (AutoProcessor).")
        # Follow RLVRVLMPipeline: ensure these are not None
        img_proc = getattr(self.processor, "image_processor", None)
        if img_proc is not None:
            model_args = self.pipeline_config.actor_train.model_args
            if getattr(img_proc, "max_pixels", None) is None:
                img_proc.max_pixels = getattr(model_args, "max_pixels", 1024 * 1024)
            if getattr(img_proc, "min_pixels", None) is None:
                img_proc.min_pixels = getattr(model_args, "min_pixels", 56 * 56)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        # Dataset
        self.dataset = self._build_dataset_or_skip()

        data_collator = DataCollatorWithPaddingForMM(
            tokenizer=self.tokenizer,
            processor=self.processor,
            extra_data_provider=get_extra_data_provider(
                self.pipeline_config.actor_train.model_args.model_name_or_path,
                processor=self.processor,
            ),
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

    def _build_dataset_or_skip(self):
        data_args = self.pipeline_config.actor_train.data_args
        file_names = _as_list(getattr(data_args, "file_name", None))

        if file_names and all(os.path.exists(p) for p in file_names):
            import datasets

            from roll.pipeline.rlvr.rlvr_math_vlm_pipeline import encode_function, get_dataset

            features = datasets.Features(
                {
                    "image": datasets.Sequence(feature=datasets.Image(decode=True)),
                    "prompt": datasets.Value("string"),
                    "ground_truth": datasets.Value("string"),
                    "image_flag": datasets.Value("bool"),
                    "tag": datasets.Value("string"),
                }
            )
            return get_dataset(data_args, encode_function, self.processor, features)

    @torch.no_grad()
    def run(self, max_batches: Optional[int] = None):
        """
        Run the pipeline and capture layer states using environment variables.

        Args:
            max_batches: Maximum number of batches to process (None for all)
        """
        global_step = 0
        results = []

        # Create output directories
        fsdp_dir = self.output_dir / "fsdp2"
        hf_dir = self.output_dir / "hf"
        inputs_dir = self.output_dir / "inputs"
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, batch_dict in enumerate(tqdm(self.dataloader)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            logger.info(f"vlm layer states pipeline step {global_step} batch {batch_idx} start...")

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.meta_info = {"global_step": global_step, "_broadcast_non_tensor_batch": True}

            # Save inputs and embeddings
            save_inputs_and_embeddings(batch, inputs_dir, "input", global_step, batch_idx)

            # Generate responses using actor_infer (vLLM)
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["multi_modal_data"],
            )
            gen_batch.meta_info = {"global_step": global_step}
            generate_output: DataProto = self.actor_infer.generate(data=gen_batch)

            # Merge generated full sequences back
            batch.batch = generate_output.batch
            batch = batch.union(generate_output)

            _set_capture_env_on_cluster(
                self.actor_train,
                save_dir=fsdp_dir,
                prefix="fsdp2",
                step=global_step,
                batch_idx=batch_idx,
            )
            logprobs_fsdp = self.actor_train.compute_log_probs(batch)

            _set_capture_env_on_cluster(
                self.reference,
                save_dir=hf_dir,
                prefix="hf",
                step=global_step,
                batch_idx=batch_idx,
            )
            logprobs_ref = self.reference.compute_log_probs(batch)

            # Directly compare saved inputs/embeddings/layer states for this step/batch.
            analysis_out = analysis_dir / f"divergence_step{global_step}_batch{batch_idx}.json"
            analyze_divergence(
                fsdp_dir=fsdp_dir,
                hf_dir=hf_dir,
                inputs_dir=inputs_dir,
                output_file=analysis_out,
                global_step=global_step,
                batch_idx=batch_idx,
                threshold=1e-5,
            )

            lp_fsdp = logprobs_fsdp.batch["log_probs"]
            lp_ref = logprobs_ref.batch["log_probs"]
            mask = batch.batch["response_mask"][:, 1:].to(torch.bool)

            diff = (lp_fsdp - lp_ref).abs()
            diff_max = diff[mask].max().item() if mask.any() else 0.0
            diff_mean = diff[mask].mean().item() if mask.any() else 0.0

            results.append(
                {
                    "global_step": global_step,
                    "batch_idx": batch_idx,
                    "diff_max": diff_max,
                    "diff_mean": diff_mean,
                }
            )
            logger.info(f"vlm logprob diff_max={diff_max:.6f}, diff_mean={diff_mean:.6f}")

            global_step += 1

        logger.info("vlm layer states pipeline complete!")

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        return results


def test_fsdp_vlm_layer_states_cp2():
    init()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_vlm_cp2_config")
    pipeline = TestFSDPVLMLayerStatesPipeline(config, output_dir="./layer_states_output")
    results = pipeline.run(max_batches=1)  # Start with 1 batch for testing

    logger.info(f"Test FSDP VLM layer states (CP2) completed, results saved to {pipeline.output_dir}")


if __name__ == "__main__":
    test_fsdp_vlm_layer_states_cp2()
