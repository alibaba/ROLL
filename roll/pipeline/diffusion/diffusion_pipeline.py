from __future__ import annotations

import json
import os
import time
from functools import partial
from typing import Any, Dict, Optional

import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from roll.datasets.collator import DataCollatorForDiffusion
from roll.datasets.dataset import get_dataset
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.diffusion.rewards.dump_mixin import is_dump_enabled, resolve_dump_base_dir
from roll.pipeline.diffusion.schema import validate_diffusion_rollout_batch
from roll.pipeline.diffusion.utils import (
    attach_group_ids,
    compute_per_tag_score_metrics,
    get_diffusion_encode_function,
    prepare_training_batch,
    validate_diffusion_config,
    validate_diffusion_dataset,
)
from roll.datasets.dataset import update_dataset_domain
from roll.pipeline.rlvr.rlvr_pipeline import preprocess_dataset
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger


logger = get_logger()


class DiffusionPipeline(BasePipeline):
    """Generic diffusion training pipeline with algorithm/model/reward adapters.

    This first implementation intentionally keeps the verified FlowGRPO +
    Qwen-Image + OCR lifecycle order intact. Algorithm, model and reward
    differences are delegated to adapters, while trainer/inference engines keep
    using ROLL's existing ``WorkerConfig.strategy_args.strategy_name`` strategy
    abstraction, matching the RLVR pipeline.
    """

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.pipeline_config.set_max_steps(self.pipeline_config.max_steps)

        self.use_reference = self.pipeline_config.enable_reference
        self._validate_strategy_contracts()
        validate_diffusion_config(self.pipeline_config)
        if not self.pipeline_config.rewards:
            raise ValueError("rewards must be configured for diffusion training.")
        if self.pipeline_config.validation is not None and self.pipeline_config.validation.data_args is not None:
            if self.pipeline_config.validation.generating_args is None:
                raise ValueError("validation.generating_args must be configured when validation.data_args is configured.")

        self.processor = default_tokenizer_provider(model_args=self.pipeline_config.actor_infer.model_args)
        self.dataset = self._build_dataset(self.pipeline_config.actor_train.data_args)
        self.has_validation = (
            self.pipeline_config.validation is not None and self.pipeline_config.validation.data_args is not None
        )
        self.val_dataset = self._build_dataset(self.pipeline_config.validation.data_args) if self.has_validation else None
        
        logger.info(
            "DiffusionPipeline init: algorithm=%s model_variant=%s rewards=%s max_steps=%s enable_reference=%s",
            self.pipeline_config.algorithm,
            self.pipeline_config.diffusion_model_variant,
            list(self.pipeline_config.rewards.keys()),
            self.pipeline_config.max_steps,
            self.use_reference,
        )

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.reference: Optional[Any] = None
        if self.use_reference:
            self.reference = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
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

        download_clusters = [self.actor_train, self.actor_infer, *self.reward_clusters.values()]
        if self.reference is not None:
            download_clusters.append(self.reference)
        self.download_models(*download_clusters)

        self.generate_scheduler = self._build_scheduler(
            dataset=self.dataset,
            state=self.state.kv.get("scheduler_state_diffusion", None),
        )
        self.val_generate_scheduler: Optional[Any] = (
            self._build_scheduler(dataset=self.val_dataset, is_val=True) if self.has_validation else None
        )

        # Preserve the FlowGRPO ordering that avoids colocated vLLM backends
        # competing for peak GPU memory during initialization.
        ray.get(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        for reward_cluster in self.reward_clusters.values():
            ray.get(reward_cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))

        refs = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.reference is not None:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        if self.pipeline_config.algorithm == "diffnft":
            if self.resume_from_checkpoint:
                # Resuming: DCP checkpoint already restored both adapters + optimizer.
                # Skip preflight to preserve the trained EMA shadow state.
                logger.info("DiffNFT resume: skipping preflight_nft_ops (EMA state restored from checkpoint)")
            else:
                # Fresh start: verify strategy adapter ops and seed the EMA shadow adapter (shadow = current)
                self.actor_train.preflight_nft_ops()
        ray.get(self.generate_scheduler.initialize.remote())
        if self.val_generate_scheduler is not None:
            ray.get(self.val_generate_scheduler.initialize.remote())

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )
        self.set_checkpoint_clusters(self.actor_train)
        self._save_config_to_dump_dir()
        logger.info("DiffusionPipeline ready: clusters and scheduler initialized")

    def _get_dump_base_dir(self) -> str:
        """Return the dump base directory (without step subfolder)."""
        return resolve_dump_base_dir(self.pipeline_config)

    def _save_config_to_dump_dir(self):
        """Save the resolved pipeline config as YAML to the dump directory."""
        dump_base_dir = self._get_dump_base_dir()
        if not dump_base_dir:
            return
        os.makedirs(dump_base_dir, exist_ok=True)
        config_path = os.path.join(dump_base_dir, "config.yaml")
        config_dict = self.pipeline_config.to_dict()
        yaml_text = OmegaConf.to_yaml(OmegaConf.create(config_dict), resolve=True)
        with open(config_path, "w") as f:
            f.write(yaml_text)
        logger.info("Pipeline config saved to %s", config_path)

    def _strategy_name(self, worker_config, role: str) -> str:
        if worker_config.strategy_args is None:
            raise ValueError(f"{role}.strategy_args must be configured.")
        if worker_config.strategy_args.strategy_name is None:
            raise ValueError(f"{role}.strategy_args.strategy_name must be configured.")
        return worker_config.strategy_args.strategy_name

    def _validate_strategy_contracts(self) -> None:
        """Validate the existing ROLL strategy layer instead of adding engine adapters.

        RLVR already switches trainer and inference engines through
        ``WorkerConfig.strategy_args.strategy_name`` and ``create_strategy``. The
        diffusion pipeline follows the same boundary: strategy classes own
        engine-specific initialization, load/offload, log-prob replay and
        generation; adapters only own algorithm/model/reward semantics.
        """
        actor_train_strategy = self._strategy_name(self.pipeline_config.actor_train, "actor_train")
        actor_infer_strategy = self._strategy_name(self.pipeline_config.actor_infer, "actor_infer")

        if actor_train_strategy != "fsdp2_diffusion_train":
            raise ValueError(
                "DiffusionPipeline currently requires actor_train.strategy_args.strategy_name="
                f"'fsdp2_diffusion_train', got {actor_train_strategy!r}. Add or select a ROLL train strategy "
                "instead of introducing a trainer-engine adapter."
            )
        if actor_infer_strategy != "vllm_omni":
            raise ValueError(
                "DiffusionPipeline currently requires actor_infer.strategy_args.strategy_name='vllm_omni', "
                f"got {actor_infer_strategy!r}. Future SGLang support should be added through "
                "roll.distributed.strategy and scheduler/router output normalization."
            )
        if self.use_reference:
            reference_strategy = self._strategy_name(self.pipeline_config.reference, "reference")
            if reference_strategy not in {"fsdp2_diffusion_train", "fsdp2_diffusion_infer"}:
                raise ValueError(
                    "DiffusionPipeline reference strategy must be a diffusion FSDP2 replay strategy, "
                    f"got reference.strategy_args.strategy_name={reference_strategy!r}."
                )

    def _build_dataset(self, data_args):
        if data_args is None:
            raise ValueError("DiffusionPipeline requires data_args for every enabled dataset.")
        if not data_args.file_name:
            raise ValueError("DiffusionPipeline data_args.file_name must be configured.")

        dataset = get_dataset(data_args)
        template_name = getattr(data_args, "template", "native")
        encode_function = get_diffusion_encode_function(
            template_name=template_name,
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
        if num_proc == 1:
            num_proc = None
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=num_proc,
            desc="update_dataset_domain",
            load_from_cache_file=False,
        )
        validate_diffusion_dataset(dataset)
        return dataset

    def _build_scheduler(self, dataset, state=None, is_val: bool = False):
        if dataset is None:
            return None
        return ray.remote(DynamicSamplingScheduler).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )
        ).remote(
            pipeline_config=self.pipeline_config,
            actor_cluster=self.actor_infer,
            reward_clusters=self.reward_clusters,
            dataset=dataset,
            collect_fn_cls=DataCollatorForDiffusion,
            collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length),
            state = state,
            is_val = is_val,
        )

    def collect_rollout_batch(self, global_step: int) -> DataProto:
        """Collect rollout batch from generate scheduler.

        Caller is responsible for load_states/offload_states around this call.
        """
        batch = DataProto(meta_info={})
        batch.meta_info["global_step"] = global_step
        batch.meta_info["is_val"] = False
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["generation_config"] = self.pipeline_config.actor_infer.generating_args.to_dict()

        generate_output: DataProto = ray.get(
            self.generate_scheduler.get_batch.remote(
                data=batch,
                global_step=global_step,
                batch_size=self.pipeline_config.rollout_batch_size,
            ),
            timeout=self.pipeline_config.rpc_timeout,
        )

        generate_output = attach_group_ids(generate_output)
        generate_output.meta_info.pop("is_offload_states", None)
        return generate_output

    @torch.no_grad()
    def val(self, global_step: int) -> Dict[str, float]:
        """Run validation. Caller is responsible for load_states/offload_states."""
        if self.val_generate_scheduler is None or self.val_dataset is None:
            return {}

        batch = DataProto(meta_info={})
        batch.meta_info["global_step"] = global_step
        batch.meta_info["is_val"] = True
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["generation_config"] = self.pipeline_config.validation.generating_args.to_dict()

        step_generate_t0 = time.perf_counter()
        generate_output: DataProto = ray.get(
            self.val_generate_scheduler.get_batch.remote(
                data=batch,
                global_step=global_step,
                batch_size=len(self.val_dataset),
            ),
            timeout=self.pipeline_config.rpc_timeout,
        )
        generate_output.meta_info.pop("is_offload_states", None)

        metrics: Dict[str, float] = {"time/val_generate": time.perf_counter() - step_generate_t0}
        metrics.update({f"val/{k}": v for k, v in reduce_metrics(generate_output.meta_info.get("metrics", {})).items()})
        metrics.update(compute_per_tag_score_metrics(generate_output, prefix="val/reward"))

        if self._should_flush_reward_dumps(global_step, is_val=True):
            self._flush_reward_dumps()

        logger.info("DiffusionPipeline val metrics step=%s metrics=%s", global_step, json.dumps(metrics, ensure_ascii=False))
        DataProto.drop(generate_output)
        return metrics

    def extract_rollout_metrics(self, rollout_batch: DataProto) -> Dict[str, float]:
        return reduce_metrics(rollout_batch.meta_info.get("metrics", {}))

    def build_reward_summary_metrics(self, rollout_batch: DataProto) -> Dict[str, float]:
        metrics = compute_per_tag_score_metrics(rollout_batch, prefix="reward")
        if "token_level_rewards" in rollout_batch.batch.keys():
            token_rewards = rollout_batch.batch["token_level_rewards"].float()
            metrics["reward/overall/token_reward/mean"] = token_rewards.mean().item()
        return metrics

    def should_validate(self, global_step: int, is_last_step: bool) -> bool:
        return self.val_generate_scheduler is not None and (
            is_last_step
            or (self.pipeline_config.eval_steps > 0 and (global_step + 1) % self.pipeline_config.eval_steps == 0)
            or (global_step == 0 and self.pipeline_config.val_before_train)
        )

    def _flush_reward_dumps(self) -> None:
        """Wait for asynchronous dump queues exposed by reward workers."""
        for reward_cluster in self.reward_clusters.values():
            flush_dump = getattr(reward_cluster, "flush_dump", None)
            if flush_dump is not None:
                flush_dump(blocking=True)

    def _should_flush_reward_dumps(self, global_step: int, is_val: bool = False) -> bool:
        """Return whether reward dumping is enabled for the current step."""
        return is_dump_enabled(self.pipeline_config, global_step, is_val=is_val)

    @torch.no_grad()
    def run(self):
        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                continue
            logger.info("DiffusionPipeline step start: %s", global_step)
            step_t0 = time.perf_counter()
            metrics: Dict[str, float] = {}

            # Phase 1: Model update — sync trained weights to infer side
            # Tell infer workers the current step so LoRA adapter files get
            # unique per-step paths (e.g. /dev/shm/lora_adapter/step3)
            # instead of always overwriting step0's stale file.
            self.actor_infer.set_global_steps(global_step)
            self.actor_train.offload_states(blocking=True)
            model_update_t0 = time.perf_counter()
            metrics.update(self.model_update(global_step))
            metrics["time/step_model_update"] = time.perf_counter() - model_update_t0

            # Phase 2: Load infer + reward clusters (shared by val + rollout)
            self.actor_infer.load_states(blocking=True)
            for reward_cluster in self.reward_clusters.values():
                reward_cluster.load_states()

            # Phase 3: Validation (optional, uses freshly synced weights)
            is_last_step = global_step == self.pipeline_config.max_steps - 1
            if self.should_validate(global_step, is_last_step):
                metrics.update(self.val(global_step=global_step))

            # Phase 4: Rollout
            rollout_t0 = time.perf_counter()
            rollout_batch = self.collect_rollout_batch(global_step)
            metrics["time/step_generate"] = time.perf_counter() - rollout_t0
            metrics.update(self.extract_rollout_metrics(rollout_batch))

            # Phase 5: Offload infer + reward clusters (val + rollout both done)
            for reward_cluster in self.reward_clusters.values():
                reward_cluster.offload_states()
            self.actor_infer.offload_states(blocking=True)

            validate_diffusion_rollout_batch(rollout_batch, self.pipeline_config.algorithm, self.pipeline_config)

            batch, algorithm_metrics = prepare_training_batch(
                batch=rollout_batch,
                pipeline=self,
                global_step=global_step,
            )
            metrics.update(algorithm_metrics)

            # Phase 6: Train
            train_t0 = time.perf_counter()
            train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
            train_metrics = DataProto.materialize_concat(data_refs=train_metrics_refs)
            metrics.update(reduce_metrics(train_metrics.meta_info.pop("metrics", {})))
            metrics["time/step_train"] = time.perf_counter() - train_t0

            if self.pipeline_config.algorithm == "diffnft":
                # DiffNFT: update the shadow adapter every N global steps.
                # On off-steps the shadow stays frozen, creating a meaningful
                # gap between live and old policy for stronger training signal.
                interval = self.pipeline_config.old_policy_update_interval
                if interval <= 1 or global_step % interval == 0:
                    ema_decay = self.pipeline_config.get_ema_decay(global_step)
                    if ema_decay == 0.0:
                        self.actor_train.copy_adapter("default", "ema_lora")
                    else:
                        self.actor_train.ema_update_adapter(decay=ema_decay)
                    metrics["diffnft/ema_decay"] = ema_decay
                else:
                    metrics["diffnft/ema_decay"] = -1.0  # skipped update

            metrics.update(self.build_reward_summary_metrics(rollout_batch))
            DataProto.drop(rollout_batch)

            metrics["time/step_total"] = time.perf_counter() - step_t0
            metrics["system/step"] = global_step

            # Phase 7: Dump + Log + Checkpoint
            if self._should_flush_reward_dumps(global_step):
                self._flush_reward_dumps()
            self.state.kv["scheduler_state_diffusion"] = ray.get(
                self.generate_scheduler.get_scheduler_state.remote()
            )
            self.state.step = global_step
            self.state.log_history.append(metrics)
            logger.info("DiffusionPipeline train metrics step=%s metrics=%s", global_step, metrics)
            self.tracker.log(values=metrics, step=global_step, commit=True)
            self.do_checkpoint(global_step=global_step, is_last_step=is_last_step)
            logger.info("DiffusionPipeline step end: %s", global_step)

        self.shutdown()
        return self.state.log_history

    def shutdown(self) -> None:
        ray.get(self.generate_scheduler.shutdown.remote())
        if self.val_generate_scheduler is not None:
            ray.get(self.val_generate_scheduler.shutdown.remote())
