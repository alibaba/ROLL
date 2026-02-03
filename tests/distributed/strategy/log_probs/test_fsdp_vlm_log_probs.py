import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider, get_extra_data_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.base_worker import ActorWorker, InferWorker
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config

logger = get_logger()


def _as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


class TestFSDPVLMLogProbsPipeline(BasePipeline):
    """
    VLM logprob precision test:
    - use VLM processor + DataCollatorWithPaddingForMM (same data path as RLVRVLMPipeline)
    - generate with vLLM (actor_infer)
    - compare compute_log_probs between FSDP2 (actor_train) and HF (reference)
    """

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        # ------------------------------------------------------------------
        # Qwen3-VL precision debug dumps (rank-0 only inside each Ray actor process).
        # We must pass env vars via Ray runtime_env (worker_config.system_envs), not via driver os.environ.
        dump_root = os.path.abspath(
            os.getenv(
                "QWEN3_VL_TEST_DUMP_ROOT",
                os.path.join(self.pipeline_config.output_dir or ".", "qwen3_vl_dumps"),
            )
        )
        os.makedirs(dump_root, exist_ok=True)
        self.pipeline_config.actor_train.system_envs["QWEN3_VL_DUMP_DIR"] = os.path.join(dump_root, "actor_train")
        self.pipeline_config.reference.system_envs["QWEN3_VL_DUMP_DIR"] = os.path.join(dump_root, "reference")

        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)
        if self.processor is None:
            raise RuntimeError("VLM logprob test requires a processor (AutoProcessor).")
        # Follow RLVRVLMPipeline: ensure these are not None, otherwise qwen2_vl smart_resize will crash.
        img_proc = getattr(self.processor, "image_processor", None)
        if img_proc is not None:
            model_args = self.pipeline_config.actor_train.model_args
            if getattr(img_proc, "max_pixels", None) is None:
                img_proc.max_pixels = getattr(model_args, "max_pixels", 1024 * 1024)
            if getattr(img_proc, "min_pixels", None) is None:
                img_proc.min_pixels = getattr(model_args, "min_pixels", 56 * 56)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        # Dataset: prefer real VLM dataset if paths exist; otherwise skip (this is a GPU-heavy test anyway).
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
        # self.actor_infer: Any = Cluster(
        #     name=self.pipeline_config.actor_infer.name,
        #     worker_cls=InferWorker,
        #     resource_manager=self.resource_manager,
        #     worker_config=self.pipeline_config.actor_infer,
        # )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )

        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
        # self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)

    def _build_dataset_or_skip(self):
        # Follow RLVRVLMPipeline data path when real files exist.
        data_args = self.pipeline_config.actor_train.data_args
        file_names = _as_list(getattr(data_args, "file_name", None))

        # Common pattern in configs: absolute paths only exist on the training machine.
        if file_names and all(os.path.exists(p) for p in file_names):
            import datasets

            from roll.pipeline.rlvr.rlvr_math_vlm_pipeline import encode_function, get_dataset

            features = datasets.Features(
                {
                    # only support single image temporarily since sglang usage
                    # "image": datasets.Image(decode=True),
                    "image": datasets.Sequence(feature=datasets.Image(decode=True)),
                    "prompt": datasets.Value("string"),
                    "ground_truth": datasets.Value("string"),
                    # for text and multi-modal mixed data usage, indicating valid image
                    "image_flag": datasets.Value("bool"),
                    # for area seperated validation, dummy currently
                    "tag": datasets.Value("string"),
                }
            )
            return get_dataset(data_args, encode_function, self.processor, features)

    @torch.no_grad()
    def run(self):
        global_step = 0
        results = []

        for batch_dict in tqdm(self.dataloader):
            logger.info(f"vlm logprob pipeline step {global_step} start...")

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.meta_info = {"global_step": global_step, "_broadcast_non_tensor_batch": True}
            batch.batch["response_mask"] = batch.batch["attention_mask"].clone()

            # Generate responses using actor_infer (vLLM). Needs multi_modal_data for VLM prompts.
            # gen_batch = batch.pop(
            #     batch_keys=["input_ids", "attention_mask", "position_ids"],
            #     non_tensor_batch_keys=["multi_modal_data"],
            # )
            # gen_batch.meta_info = {"global_step": global_step}
            # generate_output: DataProto = self.actor_infer.generate(data=gen_batch)

            # Merge generated full sequences back with original (keeps multi_modal_inputs for HF/FSDP forward).
            # batch.batch = generate_output.batch
            # batch = batch.union(generate_output)

            # Compute log probs from FSDP2 and HF reference.
            logprobs_fsdp = self.actor_train.compute_log_probs(batch)
            logprobs_ref = self.reference.compute_log_probs(batch)

            # layer_states = self.actor_train.compute_layer_state(batch)
            # layer_states_ref = self.reference.compute_layer_state(batch)
            # breakpoint()

            lp_fsdp = logprobs_fsdp.batch["log_probs"]
            lp_ref = logprobs_ref.batch["log_probs"]

            diff = (lp_fsdp - lp_ref).abs()
            diff_max = diff.max().item()
            diff_mean = diff.mean().item()

            results.append(
                {
                    "global_step": global_step,
                    "diff_max": diff_max,
                    "diff_mean": diff_mean,
                }
            )
            logger.info(f"vlm logprob diff_max={diff_max:.6f}, diff_mean={diff_mean:.6f}")

            global_step += 1
            break

        logger.info("vlm logprob pipeline complete!")
        return results


def test_fsdp_vlm_log_probs_cp2():
    init()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_vlm_cp2_config")
    pipeline = TestFSDPVLMLogProbsPipeline(config)
    results = pipeline.run()

    output_file = "test_fsdp_vlm_log_probs_cp2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        for m in results:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Test FSDP VLM log probs (CP2) completed, results saved to {output_file}")


if __name__ == "__main__":
    test_fsdp_vlm_log_probs_cp2()
