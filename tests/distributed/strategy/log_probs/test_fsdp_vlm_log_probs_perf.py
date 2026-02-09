import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
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


def get_timer_stats():
    """Get timer statistics from the context parallel utilities."""
    try:
        from roll.utils.context_parallel.globals import get_timer, log_timer_stats, clear_timer_stats
        return {
            "available": True,
            "timers": log_timer_stats(),
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def get_memory_stats():
    """Get GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


class TestFSDPVLMLogProbsPipeline(BasePipeline):
    """
    VLM logprob precision test with performance statistics:
    - use VLM processor + DataCollatorWithPaddingForMM (same data path as RLVRVLMPipeline)
    - compare compute_log_probs between FSDP2 (actor_train) and HF (reference)
    - measure timing, memory usage, and communication overhead
    """

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        # ------------------------------------------------------------------
        # Qwen3-VL precision debug dumps (rank-0 only inside each Ray actor process).
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
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )

        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
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
    def run(self):
        global_step = 0
        results = []
        
        # Clear timer stats before starting
        try:
            from roll.utils.context_parallel.globals import clear_timer_stats
            clear_timer_stats()
        except:
            pass

        for batch_dict in tqdm(self.dataloader):
            logger.info(f"vlm logprob pipeline step {global_step} start...")

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.meta_info = {"global_step": global_step, "_broadcast_non_tensor_batch": True}
            batch.batch["response_mask"] = batch.batch["attention_mask"].clone()

            # Get initial memory stats
            mem_before = get_memory_stats()

            # Time FSDP2 compute_log_probs
            start_fsdp = time.time()
            logprobs_fsdp = self.actor_train.compute_log_probs(batch)
            time_fsdp = time.time() - start_fsdp

            # Get memory stats after FSDP2
            mem_after_fsdp = get_memory_stats()

            # Get timer stats after FSDP2
            timer_stats_fsdp = get_timer_stats()

            # Clear timers for reference run
            try:
                from roll.utils.context_parallel.globals import clear_timer_stats
                clear_timer_stats()
            except:
                pass

            # Time HF reference compute_log_probs
            start_ref = time.time()
            logprobs_ref = self.reference.compute_log_probs(batch)
            time_ref = time.time() - start_ref

            # Get memory stats after reference
            mem_after_ref = get_memory_stats()

            # Get timer stats after reference (should be minimal)
            timer_stats_ref = get_timer_stats()

            # Compute correctness metrics
            lp_fsdp = logprobs_fsdp.batch["log_probs"]
            lp_ref = logprobs_ref.batch["log_probs"]

            diff = (lp_fsdp - lp_ref).abs()
            diff_max = diff.max().item()
            diff_mean = diff.mean().item()
            diff_std = diff.std().item()

            # Check if results are numerically equivalent
            is_correct = diff_max < 1e-5

            # Batch statistics
            batch_size = batch.batch["input_ids"].size(0)
            seq_len = batch.batch["input_ids"].size(1)
            num_tokens = (batch.batch["attention_mask"].sum()).item()

            # Speedup calculation
            speedup = time_ref / time_fsdp if time_fsdp > 0 else 0

            result = {
                "global_step": global_step,
                "correctness": {
                    "diff_max": diff_max,
                    "diff_mean": diff_mean,
                    "diff_std": diff_std,
                    "is_correct": is_correct,
                },
                "performance": {
                    "time_fsdp_seconds": time_fsdp,
                    "time_ref_seconds": time_ref,
                    "speedup": speedup,
                    "tokens_per_second_fsdp": num_tokens / time_fsdp if time_fsdp > 0 else 0,
                    "tokens_per_second_ref": num_tokens / time_ref if time_ref > 0 else 0,
                },
                "memory": {
                    "before_gb": mem_before.get("allocated_gb", 0),
                    "after_fsdp_gb": mem_after_fsdp.get("allocated_gb", 0),
                    "after_ref_gb": mem_after_ref.get("allocated_gb", 0),
                    "fsdp_memory_increase_gb": mem_after_fsdp.get("allocated_gb", 0) - mem_before.get("allocated_gb", 0),
                },
                "batch_info": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_tokens": num_tokens,
                },
                "communication": {
                    "fsdp_timer_stats": timer_stats_fsdp,
                    "ref_timer_stats": timer_stats_ref,
                }
            }

            results.append(result)
            
            logger.info(f"Step {global_step}:")
            logger.info(f"  Correctness: diff_max={diff_max:.6f}, diff_mean={diff_mean:.6f}, is_correct={is_correct}")
            logger.info(f"  Performance: FSDP={time_fsdp:.4f}s, Ref={time_ref:.4f}s, Speedup={speedup:.2f}x")
            logger.info(f"  Throughput: FSDP={result['performance']['tokens_per_second_fsdp']:.0f} tok/s, Ref={result['performance']['tokens_per_second_ref']:.0f} tok/s")
            
            if timer_stats_fsdp.get("available"):
                logger.info(f"  Communication stats: {timer_stats_fsdp.get('timers', {})}")

            global_step += 1
            break  # Only run one step for testing

        logger.info("vlm logprob pipeline complete!")
        return results


def test_fsdp_vlm_log_probs_cp2_with_perf():
    """Test VLM logprobs with CP2 and comprehensive performance statistics."""
    init()
    config = make_baseline_config(config_path="./log_probs", config_name="log_probs_fsdp_vlm_cp2_config")
    pipeline = TestFSDPVLMLogProbsPipeline(config)
    results = pipeline.run()

    output_file = "test_fsdp_vlm_log_probs_cp2_with_perf.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Test FSDP VLM log probs (CP2) with performance stats completed!")
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    if results:
        r = results[0]
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Correctness: {r['correctness']['is_correct']} (diff_max={r['correctness']['diff_max']:.6f})")
        logger.info(f"Speedup: {r['performance']['speedup']:.2f}x")
        logger.info(f"FSDP time: {r['performance']['time_fsdp_seconds']:.4f}s")
        logger.info(f"Reference time: {r['performance']['time_ref_seconds']:.4f}s")
        logger.info(f"FSDP throughput: {r['performance']['tokens_per_second_fsdp']:.0f} tokens/s")
        logger.info(f"Memory increase: {r['memory']['fsdp_memory_increase_gb']:.2f} GB")
        logger.info("="*80)


if __name__ == "__main__":
    test_fsdp_vlm_log_probs_cp2_with_perf()