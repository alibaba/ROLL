from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from functools import partial
from typing import Any, Dict

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from roll.datasets.collator import DataCollatorForDiffusion
from roll.datasets.dataset import get_dataset, update_dataset_domain
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.storage import SharedStorage
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.diffusion.rewards.dump_mixin import resolve_dump_step_dir
from roll.pipeline.diffusion.schema import require_batch_keys
from roll.pipeline.diffusion.utils import get_diffusion_encode_function, validate_diffusion_dataset
from roll.pipeline.rlvr.rlvr_pipeline import preprocess_dataset
from roll.utils.checkpoint_manager import download_model, file_lock_context
from roll.utils.constants import RAY_NAMESPACE, STORAGE_NAME
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.network_utils import get_node_ip


logger = get_logger()


def _remove_path(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _validate_transformer_checkpoint(transformer_ckpt_path: str) -> None:
    if not os.path.isdir(transformer_ckpt_path):
        raise FileNotFoundError(f"Transformer checkpoint directory not found: {transformer_ckpt_path}")

    config_path = os.path.join(transformer_ckpt_path, "config.json")
    weight_files = [
        name for name in os.listdir(transformer_ckpt_path) if name.endswith((".safetensors", ".bin"))
    ]
    if not os.path.isfile(config_path) or not weight_files:
        raise ValueError(
            "transformer_ckpt must be a Diffusers model directory containing config.json "
            f"and safetensors/bin weights: {transformer_ckpt_path}"
        )


def _prepared_qwen_image_model_path(base_model: str, transformer_ckpt: str) -> str:
    cache_key = hashlib.sha256(f"{base_model}\0{transformer_ckpt}".encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), "roll_prepared_models", cache_key)


def _copy_qwen_image_model_with_transformer(
    base_model_path: str,
    transformer_ckpt_path: str,
    prepared_model_path: str,
) -> str:
    """Create a Qwen-Image pipeline copy containing the requested transformer checkpoint."""
    if not os.path.isdir(base_model_path):
        raise FileNotFoundError(f"Qwen-Image base model directory not found: {base_model_path}")
    _validate_transformer_checkpoint(transformer_ckpt_path)

    if os.path.realpath(base_model_path) == os.path.realpath(prepared_model_path):
        raise ValueError("prepared_model_path must differ from base_model_path")

    os.makedirs(os.path.dirname(prepared_model_path), exist_ok=True)
    with file_lock_context(prepared_model_path):
        prepared_transformer_path = os.path.join(prepared_model_path, "transformer")
        if os.path.isdir(prepared_model_path):
            try:
                _validate_transformer_checkpoint(prepared_transformer_path)
                return prepared_model_path
            except (FileNotFoundError, ValueError):
                _remove_path(prepared_model_path)
        elif os.path.lexists(prepared_model_path):
            _remove_path(prepared_model_path)

        temp_model_path = f"{prepared_model_path}.tmp-{os.getpid()}"
        _remove_path(temp_model_path)

        base_model_root = os.path.realpath(base_model_path)

        def ignore_base_transformer(directory: str, names: list[str]) -> set[str]:
            if os.path.realpath(directory) == base_model_root and "transformer" in names:
                return {"transformer"}
            return set()

        try:
            # Copy all immutable pipeline components, but avoid copying the transformer
            # that will immediately be replaced by the requested checkpoint.
            shutil.copytree(
                base_model_path,
                temp_model_path,
                ignore=ignore_base_transformer,
            )
            # DCP contains resume and optimizer state that Diffusers inference does not need.
            shutil.copytree(
                transformer_ckpt_path,
                os.path.join(temp_model_path, "transformer"),
                ignore=shutil.ignore_patterns("dcp"),
            )
            os.replace(temp_model_path, prepared_model_path)
        except Exception as exc:
            _remove_path(temp_model_path)
            raise RuntimeError(
                "Failed to prepare Qwen-Image model copy: "
                f"base_model={base_model_path}, checkpoint={transformer_ckpt_path}, "
                f"output={prepared_model_path}"
            ) from exc

    logger.info(
        "Prepared Qwen-Image model copy at %s with transformer checkpoint %s",
        prepared_model_path,
        transformer_ckpt_path,
    )
    return prepared_model_path


@ray.remote
def _prepare_qwen_image_transformer(base_model: str, transformer_ckpt: str) -> str:
    local_base_model = download_model(base_model)
    local_transformer_ckpt = download_model(transformer_ckpt)
    prepared_model_path = _copy_qwen_image_model_with_transformer(
        base_model_path=local_base_model,
        transformer_ckpt_path=local_transformer_ckpt,
        prepared_model_path=_prepared_qwen_image_model_path(base_model, transformer_ckpt),
    )

    # Worker.initialize() resolves the original model URI through this node-local
    # cache. Point it at the prepared copy before actor_infer initialization.
    node_ip = get_node_ip()
    shared_storage = SharedStorage.options(
        name=STORAGE_NAME,
        get_if_exists=True,
        namespace=RAY_NAMESPACE,
    ).remote()
    ray.get(
        shared_storage.put.remote(
            key=f"{node_ip}:{base_model}",
            data=prepared_model_path,
        )
    )
    logger.info(
        "Updated model path cache on node %s: %s -> %s",
        node_ip,
        base_model,
        prepared_model_path,
    )
    return prepared_model_path


class DiffusionRolloutPipeline(BasePipeline):
    """Rollout-only Qwen-Image pipeline with OCR reward evaluation and dumping."""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        self.processor = default_tokenizer_provider(model_args=self.pipeline_config.actor_infer.model_args)
        self.dataset = self._build_dataset()

        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reward_clusters = {
            reward_name: Cluster(
                name=f"reward-{reward_name}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for reward_name, worker_config in self.pipeline_config.rewards.items()
        }
        self.download_models(self.actor_infer, *self.reward_clusters.values())

        self.generate_scheduler = ray.remote(DynamicSamplingScheduler).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )
        ).remote(
            pipeline_config=self.pipeline_config,
            actor_cluster=self.actor_infer,
            reward_clusters=self.reward_clusters,
            dataset=self.dataset,
            collect_fn_cls=DataCollatorForDiffusion,
            collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length),
            is_val=True,
        )

        ray.get(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        reward_init_refs = []
        for reward_cluster in self.reward_clusters.values():
            reward_init_refs.extend(
                reward_cluster.initialize(pipeline_config=self.pipeline_config, blocking=False)
            )
        ray.get(reward_init_refs)
        ray.get(self.generate_scheduler.initialize.remote())
        logger.info("DiffusionRolloutPipeline initialized with %d prompts", len(self.dataset))


    def download_models(self, *clusters: Cluster):
        """Download all models, then install the rollout transformer checkpoint on actor nodes."""
        super().download_models(*clusters)

        transformer_ckpt = self.pipeline_config.transformer_ckpt
        if not transformer_ckpt:
            return

        base_model = self.pipeline_config.actor_infer.model_args.model_name_or_path
        if not base_model:
            raise ValueError("actor_infer.model_args.model_name_or_path is required when transformer_ckpt is set")
        if self.actor_infer.placement_groups is None:
            raise RuntimeError("actor_infer placement groups must be allocated before downloading models")

        node2pg = {}
        for pg_list in self.actor_infer.placement_groups:
            for pg in pg_list:
                node2pg.setdefault(pg["node_rank"], pg["placement_group"])

        prepared_model_paths = ray.get(
            [
                _prepare_qwen_image_transformer.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group)
                ).remote(base_model=base_model, transformer_ckpt=transformer_ckpt)
                for placement_group in node2pg.values()
            ]
        )
        logger.info(
            "Prepared transformer checkpoint on %d actor-infer node(s): %s",
            len(node2pg),
            prepared_model_paths,
        )

    def _build_dataset(self):
        data_args = self.pipeline_config.actor_infer.data_args
        dataset = get_dataset(data_args)
        encode_function = get_diffusion_encode_function(
            template_name=getattr(data_args, "template", "native"),
            tokenizer=self.processor,
            data_args=data_args,
        )
        dataset = preprocess_dataset(
            dataset,
            self.pipeline_config.prompt_length,
            encode_function,
            data_args=data_args,
        )
        num_proc = data_args.preprocessing_num_workers
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=None if num_proc == 1 else num_proc,
            desc="update_dataset_domain",
            load_from_cache_file=False,
        )
        validate_diffusion_dataset(dataset)
        return dataset

    def _collect_rollout_batch(self, global_step: int) -> DataProto:
        request = DataProto(
            meta_info={
                "global_step": global_step,
                "is_val": True,
                "is_offload_states": False,
                "generation_config": self.pipeline_config.actor_infer.generating_args.to_dict(),
            }
        )
        output: DataProto = ray.get(
            self.generate_scheduler.get_batch.remote(
                data=request,
                global_step=global_step,
                batch_size=self.pipeline_config.rollout_batch_size,
            ),
            timeout=self.pipeline_config.rpc_timeout,
        )
        output.meta_info.pop("is_offload_states", None)
        return output

    def _get_reward_dump_step_dir(self, global_step: int) -> str:
        """Return the current reward dump directory, or an empty string when disabled.

        The rollout pipeline only produces validation-style outputs, so it asks
        for the ``val`` phase directory.
        """
        return resolve_dump_step_dir(self.pipeline_config, global_step, is_val=True)

    def _flush_reward_dumps(self) -> None:
        """Wait for asynchronous dump queues exposed by reward workers."""
        for reward_cluster in self.reward_clusters.values():
            flush_dump = getattr(reward_cluster, "flush_dump", None)
            if flush_dump is not None:
                flush_dump(blocking=True)

    @torch.no_grad()
    def run(self):
        summaries = []
        try:
            for global_step in range(self.pipeline_config.max_steps):
                logger.info("Diffusion rollout step %s start", global_step)
                started_at = time.perf_counter()
                batch = self._collect_rollout_batch(global_step)

                require_batch_keys(
                    batch,
                    ["scores", "token_level_rewards"],
                    "Diffusion rollout reward output validation",
                )
                metrics: Dict[str, float] = {
                    "time/step_rollout": time.perf_counter() - started_at,
                    "rollout/num_images": int(batch.batch.batch_size[0]),
                    "system/step": global_step,
                }
                metrics.update(reduce_metrics(batch.meta_info.get("metrics", {})))
                if "scores" in batch.batch.keys():
                    scores = batch.batch["scores"].float()
                    metrics.update(
                        {
                            "rollout/reward_mean": scores.mean().item(),
                            "rollout/reward_max": scores.max().item(),
                            "rollout/reward_min": scores.min().item(),
                        }
                    )
                dump_step_dir = self._get_reward_dump_step_dir(global_step)
                if dump_step_dir:
                    self._flush_reward_dumps()
                self.state.step = global_step
                self.state.log_history.append(metrics)
                self.tracker.log(values=metrics, step=global_step)
                summary = {"global_step": global_step, "output_dir": dump_step_dir or None, **metrics}
                summaries.append(summary)
                logger.info(
                    "Diffusion rollout step complete: %s",
                    json.dumps(summary, ensure_ascii=False),
                )
                DataProto.drop(batch)
        finally:
            ray.get(self.generate_scheduler.shutdown.remote())
            self.tracker.finish()

        return summaries
