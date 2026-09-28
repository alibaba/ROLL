"""DMD pipeline orchestration for diffusion components."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterator
from concurrent import futures
from typing import Any

import numpy as np
import ray
import torch
from codetiming import Timer
from torch.utils.data import DataLoader

from roll.datasets.dataset import get_dataset
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.diffusion.dmd.dmd_algorithm import (
    DMD_KEY_GENERATED,
    DMD_KEY_GAN_INPUT_GRADIENT,
    DMD_KEY_NOISY_LATENT,
    DMD_KEY_NOISY_LATENT_CA,
    DMD_KEY_PRED_FAKE_IMAGE,
    DMD_KEY_PRED_REAL_IMAGE,
    DMD_KEY_PRED_REAL_COND_CA,
    DMD_KEY_PRED_REAL_UNCOND_CA,
    DMD_KEY_PRED_X0,
    DMD_KEY_PROMPTS,
    DMD_KEY_REAL_LATENT,
    DMD_KEY_REAL_PROMPTS,
)
from roll.pipeline.diffusion.dmd.dmd_config import DMDConfig
from roll.pipeline.diffusion.tensor_transfer import TensorTransferGroup
from roll.utils.checkpoint_manager import download_model
from roll.utils.functionals import clusters_have_disjoint_devices, reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager
from roll.utils.worker_state import WorkerState

logger = get_logger()

DMD_CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
DMD_DATA_STATE_KEY = "dmd_data"
DMD_GAN_DATA_STATE_KEY = "dmd_gan_data"
DMD_RESUME_ROLES = {"generator", "fake_score"}


class DMDPipeline(BasePipeline):
    """ROLL pipeline for the generator, fake-score, and real-score DMD roles."""

    def __init__(self, pipeline_config: DMDConfig) -> None:
        resume_manifest = self._prepare_resume(pipeline_config)
        super().__init__(pipeline_config)
        self._pending_checkpoint_finalize = None
        if self.resume_from_checkpoint:
            self.pipeline_config.resume_from_checkpoint = self.resume_from_checkpoint
            if self.state.step != int(resume_manifest["pipeline_step"]):
                raise ValueError(
                    f"DMD checkpoint pipeline step mismatch: manifest={resume_manifest['pipeline_step']}, "
                    f"state={self.state.step}"
                )

        if self.pipeline_config.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")
        generator_updates = (
            self.pipeline_config.max_steps + self.pipeline_config.dfake_gen_update_ratio - 1
        ) // self.pipeline_config.dfake_gen_update_ratio
        self.pipeline_config.generator.training_args.max_steps = generator_updates
        self.pipeline_config.fake_score.training_args.max_steps = self.pipeline_config.max_steps
        self.pipeline_config.real_score.training_args.max_steps = self.pipeline_config.max_steps
        dataset = get_dataset(self.pipeline_config.generator.data_args)
        gan_dataset = (
            get_dataset(self.pipeline_config.gan.real_data).with_format("numpy")
            if self.pipeline_config.gan.enabled
            else None
        )
        self._initialize_role_clusters()
        self._initialize_tensor_transfers()
        self._initialize_dataloader(dataset)
        if gan_dataset is not None:
            self._initialize_gan_dataloader(gan_dataset)
        role_clusters = {
            "generator": self.generator,
            "fake_score": self.fake_score,
        }
        unknown_checkpoint_roles = set(self.pipeline_config.checkpoint_roles) - set(role_clusters)
        if unknown_checkpoint_roles:
            raise ValueError(f"Unsupported DMD checkpoint roles: {sorted(unknown_checkpoint_roles)}")
        self.checkpoint_role_clusters = {
            role: role_clusters[role] for role in self.pipeline_config.checkpoint_roles
        }
        if self.resume_from_checkpoint:
            WorkerState.load_rng_state(
                load_dir=os.path.join(self.resume_from_checkpoint, "pipeline"),
                tag="pipeline",
            )

    @staticmethod
    def _prepare_resume(pipeline_config: DMDConfig) -> dict[str, Any] | None:
        """Resolve a complete DMD checkpoint before BasePipeline restores its state."""
        resume_checkpoint = pipeline_config.resume_from_checkpoint
        if resume_checkpoint is True:
            checkpoint_output_dir = pipeline_config.checkpoint_config.get("output_dir")
            if checkpoint_output_dir is None:
                checkpoint_output_dir = pipeline_config.output_dir
            search_root = checkpoint_output_dir
            if re.fullmatch(r"\d{8}-\d{6}", os.path.basename(search_root)):
                search_root = os.path.dirname(search_root)
            candidates: list[tuple[float, int, str]] = []
            if os.path.isdir(search_root):
                search_dirs = [search_root]
                search_dirs.extend(entry.path for entry in os.scandir(search_root) if entry.is_dir())
                checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
                for search_dir in search_dirs:
                    for entry in os.scandir(search_dir):
                        match = checkpoint_pattern.fullmatch(entry.name)
                        if (
                            match is not None
                            and entry.is_dir()
                            and os.path.isfile(os.path.join(entry.path, DMD_CHECKPOINT_MANIFEST))
                        ):
                            candidates.append((entry.stat().st_mtime, int(match.group(1)), entry.path))
            resume_checkpoint = max(candidates)[2] if candidates else False
        elif isinstance(resume_checkpoint, str):
            resume_checkpoint = download_model(resume_checkpoint)

        if not resume_checkpoint:
            pipeline_config.resume_from_checkpoint = False
            return None

        manifest_path = os.path.join(resume_checkpoint, DMD_CHECKPOINT_MANIFEST)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Incomplete DMD checkpoint: missing {manifest_path}")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            resume_manifest = json.load(manifest_file)
        if set(resume_manifest["roles"]) != DMD_RESUME_ROLES:
            raise ValueError(
                f"A resumable DMD checkpoint must contain {sorted(DMD_RESUME_ROLES)} roles, "
                f"got {resume_manifest['roles']}"
            )
        pipeline_config.resume_from_checkpoint = resume_checkpoint
        if pipeline_config.checkpoint_config.get("type") == "file_system":
            pipeline_config.checkpoint_config["output_dir"] = os.path.dirname(resume_checkpoint)
        return resume_manifest

    def _initialize_role_clusters(self) -> None:
        """Construct and initialize DMD role clusters."""

        self.generator = Cluster(
            name=self.pipeline_config.generator.name,
            worker_cls=self.pipeline_config.generator.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.generator,
        )
        self.fake_score = Cluster(
            name=self.pipeline_config.fake_score.name,
            worker_cls=self.pipeline_config.fake_score.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.fake_score,
        )
        self.real_score = Cluster(
            name=self.pipeline_config.real_score.name,
            worker_cls=self.pipeline_config.real_score.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.real_score,
        )
        role_clusters = [self.generator, self.fake_score, self.real_score]
        self.score_roles_colocated = not clusters_have_disjoint_devices([self.fake_score, self.real_score])
        if clusters_have_disjoint_devices(role_clusters):
            init_refs = []
            for cluster in role_clusters:
                init_refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
            ray.get(init_refs)
        else:
            for cluster in role_clusters:
                cluster.initialize(pipeline_config=self.pipeline_config, blocking=True)

    def _initialize_tensor_transfers(self) -> None:
        """Set up direct GPU transfers when every role pair supports them."""
        self.gpu_tensor_transfer_enabled = False
        self.tensor_transfer_groups: dict[str, TensorTransferGroup] = {}
        if not self.pipeline_config.gpu_tensor_transfer:
            return

        cluster_pairs = {
            "generator_to_fake_score": (self.generator, self.fake_score),
            "generator_to_real_score": (self.generator, self.real_score),
            "fake_score_to_generator": (self.fake_score, self.generator),
            "real_score_to_generator": (self.real_score, self.generator),
        }
        transfer_errors = [
            error
            for src_cluster, tgt_cluster in cluster_pairs.values()
            if (error := TensorTransferGroup.compatibility_error(src_cluster, tgt_cluster)) is not None
        ]
        if transfer_errors:
            logger.warning(
                "DMD direct GPU tensor transfer is unavailable; using CPU DataProto transport: "
                + "; ".join(transfer_errors)
            )
            role_clusters = [self.generator, self.fake_score, self.real_score]
            ray.get(
                [
                    worker.set_gpu_tensor_transfer_enabled.remote(False)
                    for cluster in role_clusters
                    for worker in cluster.workers
                ]
            )
            return

        self.tensor_transfer_groups = {
            name: TensorTransferGroup(src_cluster, tgt_cluster, name)
            for name, (src_cluster, tgt_cluster) in cluster_pairs.items()
        }
        self.gpu_tensor_transfer_enabled = True

    def _initialize_dataloader(self, dataset: Any) -> None:
        """Build the global prompt dataloader and restore its data cursor."""
        per_device_bs = self.pipeline_config.generator.training_args.per_device_train_batch_size
        global_train_batch_size = self.generator.dp_size * per_device_bs
        max_consumer_dp_size = max(self.fake_score.dp_size, self.real_score.dp_size)
        if global_train_batch_size < max_consumer_dp_size:
            raise ValueError(
                "DMD generator global batch size must be >= every consumer role's data-parallel size. "
                f"Got batch_size={global_train_batch_size}, fake_score.dp_size={self.fake_score.dp_size}, "
                f"real_score.dp_size={self.real_score.dp_size}."
            )
        if global_train_batch_size % self.fake_score.dp_size != 0:
            raise ValueError(
                f"DMD global batch size {global_train_batch_size} must be divisible by "
                f"fake_score.dp_size={self.fake_score.dp_size} so every rank optimizes an equally weighted "
                "local batch."
            )

        def collate_prompts(batch: list[dict[str, Any]]) -> dict[str, Any]:
            prompts = np.array([item["text"] for item in batch], dtype=object)
            return {DMD_KEY_PROMPTS: prompts, "batch_idx": torch.arange(len(batch), dtype=torch.int64)}

        data_seed = self.pipeline_config.generator.training_args.data_seed
        if data_seed is None:
            data_seed = self.pipeline_config.seed
        data_state = self.state.kv.get(DMD_DATA_STATE_KEY)
        legacy_resume = self.resume_from_checkpoint and data_state is None
        if data_state is None:
            data_state = {
                "seed": data_seed,
                "epoch": 0,
                "batch_offset": 0,
                "shuffle": not legacy_resume,
            }
            self.state.kv[DMD_DATA_STATE_KEY] = data_state

        self._data_generator = torch.Generator().manual_seed(
            int(data_state["seed"]) + int(data_state["epoch"])
        )
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=global_train_batch_size,
            shuffle=bool(data_state["shuffle"]),
            drop_last=True,
            num_workers=self.pipeline_config.generator.training_args.dataloader_num_workers,
            collate_fn=collate_prompts,
            generator=self._data_generator,
        )
        if len(self.dataloader) == 0:
            raise ValueError("DMD dataset must contain at least one global training batch")
        if legacy_resume:
            completed_steps = self.state.step + 1
            generator_batches = self.state.step // self.pipeline_config.dfake_gen_update_ratio + 1
            consumed_batches = completed_steps + generator_batches
            data_state["epoch"], data_state["batch_offset"] = divmod(consumed_batches, len(self.dataloader))
        self._dataloader_iter: Iterator[dict[str, Any]] | None = None

    def _initialize_gan_dataloader(self, dataset: Any) -> None:
        """Build the real-latent dataloader used only by the optional DMD2 GAN objective."""
        gan_config = self.pipeline_config.gan
        global_train_batch_size = int(self.dataloader.batch_size)
        num_batches = len(dataset) // global_train_batch_size
        if num_batches == 0:
            raise ValueError("DMD2 GAN real dataset must contain at least one global training batch")

        def collate_real_latents(batch: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                DMD_KEY_REAL_LATENT: torch.stack(
                    [torch.as_tensor(item[gan_config.real_latent_key]) for item in batch]
                ),
                DMD_KEY_REAL_PROMPTS: np.array(
                    [item[gan_config.real_prompt_key] for item in batch],
                    dtype=object,
                ),
            }

        data_seed = self.pipeline_config.generator.training_args.data_seed
        if data_seed is None:
            data_seed = self.pipeline_config.seed
        data_state = self.state.kv.get(DMD_GAN_DATA_STATE_KEY)
        if data_state is None:
            data_state = {
                "seed": int(data_seed) + 1,
                "epoch": 0,
                "batch_offset": 0,
                "shuffle": True,
            }
            if self.resume_from_checkpoint:
                data_state["epoch"], data_state["batch_offset"] = divmod(
                    self.state.step + 1,
                    num_batches,
                )
            self.state.kv[DMD_GAN_DATA_STATE_KEY] = data_state

        self._gan_data_generator = torch.Generator().manual_seed(
            int(data_state["seed"]) + int(data_state["epoch"])
        )
        self.gan_dataloader = DataLoader(
            dataset=dataset,
            batch_size=global_train_batch_size,
            shuffle=bool(data_state["shuffle"]),
            drop_last=True,
            num_workers=self.pipeline_config.generator.training_args.dataloader_num_workers,
            collate_fn=collate_real_latents,
            generator=self._gan_data_generator,
        )
        self._gan_dataloader_iter: Iterator[dict[str, Any]] | None = None

    def run(self) -> None:
        """Run the DMD role schedule and optional DMD2 classification objective."""
        global_step = self.state.step + 1
        metrics_mgr = MetricsManager()

        update_ratio = int(self.pipeline_config.dfake_gen_update_ratio)
        while global_step < self.pipeline_config.max_steps:
            should_update_generator = global_step % update_ratio == 0
            metrics_mgr.clear_metrics()

            with Timer(name="step_total", logger=None) as step_total_timer:
                if should_update_generator:
                    self._run_generator_phase(self._next_batch(), global_step, metrics_mgr)
                gan_batch = self._next_gan_batch() if self.pipeline_config.gan.enabled else None
                self._run_fake_score_phase(self._next_batch(), global_step, metrics_mgr, gan_batch)
            metrics_mgr.add_metric("time/step_total", step_total_timer.last)

            metrics = {key: float(value) for key, value in metrics_mgr.get_metrics().items()}
            display_step = global_step + 1
            metrics["system/step"] = display_step
            self.state.step = global_step
            self.state.log_history.append(metrics)
            self.tracker.log(values=metrics, step=display_step)
            self.do_checkpoint(global_step=display_step, is_last_step=display_step == self.pipeline_config.max_steps)

            if global_step % self.pipeline_config.logging_steps == 0:
                parts = [
                    f"{key} = {value:.4f}"
                    for key, value in metrics.items()
                    if key.endswith(
                        (
                            "/loss",
                            "/grad_norm",
                        )
                    )
                ]
                parts.append(f"time/step = {metrics.get('time/step_total', 0):.2f}s")
                logger.info(f"step {display_step}: {', '.join(parts)}")

            global_step += 1

        logger.info("DMD pipeline complete!")

    def _next_batch(self) -> dict[str, Any]:
        """Return the next global batch and advance the checkpointed data cursor."""
        dataloader_iter = self._dataloader_iter
        if dataloader_iter is None:
            data_state = self.state.kv[DMD_DATA_STATE_KEY]
            epoch = int(data_state["epoch"])
            sampler = self.dataloader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self._data_generator.manual_seed(int(data_state["seed"]) + epoch)
            dataloader_iter = iter(self.dataloader)
            for _ in range(int(data_state["batch_offset"])):
                next(dataloader_iter)
            self._dataloader_iter = dataloader_iter
        batch = next(dataloader_iter)
        data_state = self.state.kv[DMD_DATA_STATE_KEY]
        data_state["batch_offset"] = int(data_state["batch_offset"]) + 1
        if data_state["batch_offset"] == len(self.dataloader):
            data_state["epoch"] = int(data_state["epoch"]) + 1
            data_state["batch_offset"] = 0
            self._dataloader_iter = None
        return batch

    def _next_gan_batch(self) -> dict[str, Any]:
        """Return the next real-latent batch and advance its checkpointed cursor."""
        dataloader_iter = self._gan_dataloader_iter
        if dataloader_iter is None:
            data_state = self.state.kv[DMD_GAN_DATA_STATE_KEY]
            epoch = int(data_state["epoch"])
            sampler = self.gan_dataloader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self._gan_data_generator.manual_seed(int(data_state["seed"]) + epoch)
            dataloader_iter = iter(self.gan_dataloader)
            for _ in range(int(data_state["batch_offset"])):
                next(dataloader_iter)
            self._gan_dataloader_iter = dataloader_iter
        batch = next(dataloader_iter)
        data_state = self.state.kv[DMD_GAN_DATA_STATE_KEY]
        data_state["batch_offset"] = int(data_state["batch_offset"]) + 1
        if data_state["batch_offset"] == len(self.gan_dataloader):
            data_state["epoch"] = int(data_state["epoch"]) + 1
            data_state["batch_offset"] = 0
            self._gan_dataloader_iter = None
        return batch

    def _cleanup_old_checkpoints(self) -> None:
        """Clean checkpoints from the layout used by the active storage mode."""
        if self.checkpoint_manager.uploader is not None:
            super()._cleanup_old_checkpoints()
            return

        max_ckpt = getattr(self.pipeline_config, "max_ckpt_to_keep", 0)
        output_dir = self.pipeline_config.output_dir
        if max_ckpt <= 0 or not os.path.isdir(output_dir):
            return

        checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
        checkpoint_dirs: list[tuple[int, str]] = []
        for entry in os.scandir(output_dir):
            match = checkpoint_pattern.fullmatch(entry.name)
            if entry.is_dir() and match is not None:
                checkpoint_dirs.append((int(match.group(1)), entry.path))

        checkpoint_dirs.sort(key=lambda item: item[0], reverse=True)
        old_checkpoints = checkpoint_dirs[max_ckpt:]
        if not old_checkpoints:
            return

        steps_to_delete = [step for step, _ in old_checkpoints]
        logger.info(f"Cleaning up old DMD checkpoints. Keeping {max_ckpt}, deleting steps: {steps_to_delete}")
        for _, checkpoint_dir in old_checkpoints:
            try:
                shutil.rmtree(checkpoint_dir)
                logger.info(f"Deleted old DMD checkpoint: {checkpoint_dir}")
            except Exception as error:
                logger.warning(f"Failed to delete DMD checkpoint {checkpoint_dir}: {error}")

    def do_checkpoint(self, global_step: int, is_last_step: bool | None = None) -> None:
        """Save one coordinated DMD checkpoint across pipeline and trainable roles."""
        futures.wait(self.resume_futures)
        self.resume_futures.clear()
        pending_checkpoint_finalize = getattr(self, "_pending_checkpoint_finalize", None)
        if pending_checkpoint_finalize is not None and pending_checkpoint_finalize(wait=False):
            self._pending_checkpoint_finalize = None

        if is_last_step is None:
            is_last_step = global_step == self.pipeline_config.max_steps
        should_save_interval = (
            self.pipeline_config.save_steps > 0
            and global_step > 0
            and global_step % self.pipeline_config.save_steps == 0
        )
        should_save_extra_step = global_step in self.pipeline_config.checkpoint_steps
        if not (should_save_interval or should_save_extra_step or is_last_step):
            return
        pending_checkpoint_finalize = getattr(self, "_pending_checkpoint_finalize", None)
        if pending_checkpoint_finalize is not None:
            pending_checkpoint_finalize(wait=True)
            self._pending_checkpoint_finalize = None

        metrics = self.state.log_history[-1]
        metrics["system/step"] = global_step
        ckpt_id = f"checkpoint-{global_step}"
        checkpoint_config = self.pipeline_config.checkpoint_config
        keep_local_file = checkpoint_config.get("keep_local_file", False)
        async_upload = checkpoint_config.get("async_upload", True) and not is_last_step
        local_checkpoint = self.checkpoint_manager.uploader is None
        is_resumable_checkpoint = set(self.checkpoint_role_clusters) == DMD_RESUME_ROLES
        pipeline_step = self.state.step

        def collect_role_checkpoint(role_checkpoint_refs: list[Any]) -> None:
            role_checkpoint_output = DataProto.materialize_concat(data_refs=role_checkpoint_refs)
            metrics.update(reduce_metrics(role_checkpoint_output.meta_info.pop("metrics", {})))

        if getattr(self.pipeline_config, "parallel_checkpoint_roles", False):
            checkpoint_refs_by_role = [
                cluster.do_checkpoint(global_step=global_step, is_last_step=is_last_step, blocking=False)
                for cluster in self.checkpoint_role_clusters.values()
            ]
            for role_checkpoint_refs in checkpoint_refs_by_role:
                collect_role_checkpoint(role_checkpoint_refs)
        else:
            for cluster in self.checkpoint_role_clusters.values():
                role_checkpoint_refs = cluster.do_checkpoint(
                    global_step=global_step,
                    is_last_step=is_last_step,
                    blocking=False,
                )
                collect_role_checkpoint(role_checkpoint_refs)

        pipeline_upload_future = None
        if is_resumable_checkpoint:
            pipeline_upload_root = (
                os.path.join(self.pipeline_config.output_dir, ckpt_id)
                if local_checkpoint
                else os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id)
            )
            pipeline_dir = os.path.join(pipeline_upload_root, "pipeline")
            self.state.save_to_json(save_dir=pipeline_dir, tag="pipeline")
            self.state.save_rng_state(save_dir=pipeline_dir, tag="pipeline")
            upload_kwargs = {
                "ckpt_id": ckpt_id,
                "local_state_path": pipeline_upload_root,
                "keep_local_file": keep_local_file,
            }
            if not local_checkpoint:
                if async_upload:
                    pipeline_upload_future = self.executor.submit(
                        self.checkpoint_manager.upload,
                        **upload_kwargs,
                    )
                else:
                    self.checkpoint_manager.upload(**upload_kwargs)

        def finalize_checkpoint(wait: bool = True) -> bool:
            role_upload_refs = []
            for cluster in self.checkpoint_role_clusters.values():
                role_upload_refs.extend(
                    cluster.wait_for_checkpoint_upload(ckpt_id=ckpt_id, wait=wait, blocking=False)
                )
            role_uploads_complete = all(ray.get(role_upload_refs))

            pipeline_upload_complete = pipeline_upload_future is None
            if pipeline_upload_future is not None:
                if wait:
                    pipeline_upload_future.result()
                    pipeline_upload_complete = True
                else:
                    pipeline_upload_complete = pipeline_upload_future.done()
                    if pipeline_upload_complete:
                        pipeline_upload_future.result()
            if not role_uploads_complete or not pipeline_upload_complete:
                return False

            if is_resumable_checkpoint:
                manifest_upload_root = (
                    pipeline_upload_root
                    if local_checkpoint
                    else os.path.join(self.pipeline_config.output_dir, "manifest", ckpt_id)
                )
                os.makedirs(manifest_upload_root, exist_ok=True)
                manifest_path = os.path.join(manifest_upload_root, DMD_CHECKPOINT_MANIFEST)
                with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                    json.dump(
                        {
                            "format_version": 1,
                            "checkpoint_id": ckpt_id,
                            "global_step": global_step,
                            "pipeline_step": pipeline_step,
                            "roles": sorted(self.checkpoint_role_clusters),
                        },
                        manifest_file,
                        indent=2,
                        sort_keys=True,
                    )
                    manifest_file.write("\n")
                if not local_checkpoint:
                    self.checkpoint_manager.upload(
                        ckpt_id=ckpt_id,
                        local_state_path=manifest_upload_root,
                        keep_local_file=keep_local_file,
                    )

            self._cleanup_old_checkpoints()
            return True

        if async_upload and not local_checkpoint:
            self._pending_checkpoint_finalize = finalize_checkpoint
        else:
            finalize_checkpoint()

    def _run_generator_phase(
        self,
        prompt_batch: dict[str, Any],
        global_step: int,
        metrics_mgr: MetricsManager,
    ) -> None:
        """Run DMD generator update through separate score workers."""
        batch = DataProto.from_single_dict(prompt_batch)
        batch.meta_info = {
            "global_step": global_step,
        }

        with Timer(name="generator_sample", logger=None) as timer:
            score_request_refs = self.generator.begin_generator_step(batch, blocking=False)
            score_request = DataProto.materialize_concat(data_refs=score_request_refs)
            metrics_mgr.add_reduced_metrics(score_request.meta_info.pop("metrics", {}))
            score_request.meta_info = {
                "global_step": global_step,
            }
            if self.gpu_tensor_transfer_enabled:
                self.tensor_transfer_groups["generator_to_fake_score"].transfer(
                    DMD_KEY_NOISY_LATENT,
                    DMD_KEY_NOISY_LATENT,
                    clear_source=False,
                )
                self.tensor_transfer_groups["generator_to_real_score"].transfer(
                    DMD_KEY_NOISY_LATENT,
                    DMD_KEY_NOISY_LATENT,
                )
                if self.pipeline_config.ddmd.enabled:
                    self.tensor_transfer_groups["generator_to_real_score"].transfer(
                        DMD_KEY_NOISY_LATENT_CA,
                        DMD_KEY_NOISY_LATENT_CA,
                    )
                if self.pipeline_config.gan.enabled:
                    self.tensor_transfer_groups["generator_to_fake_score"].transfer(
                        DMD_KEY_GENERATED,
                        DMD_KEY_GENERATED,
                    )
        metrics_mgr.add_metric("time/generator_sample", timer.last)

        with Timer(name="score_forward", logger=None) as timer:
            fake_score_refs = self.fake_score.forward_score(score_request, blocking=False)
            if not self.score_roles_colocated:
                real_score_refs = self.real_score.forward_score(score_request, blocking=False)
                fake_score_output = DataProto.materialize_concat(data_refs=fake_score_refs)
            else:
                fake_score_output = DataProto.materialize_concat(data_refs=fake_score_refs)
                real_score_refs = self.real_score.forward_score(score_request, blocking=False)
            real_score_output = DataProto.materialize_concat(data_refs=real_score_refs)
            metrics_mgr.add_reduced_metrics(fake_score_output.meta_info.pop("metrics", {}))
            metrics_mgr.add_reduced_metrics(real_score_output.meta_info.pop("metrics", {}))
            if self.gpu_tensor_transfer_enabled:
                self.tensor_transfer_groups["fake_score_to_generator"].transfer(
                    DMD_KEY_PRED_X0,
                    DMD_KEY_PRED_FAKE_IMAGE,
                )
                if self.pipeline_config.gan.enabled:
                    self.tensor_transfer_groups["fake_score_to_generator"].transfer(
                        DMD_KEY_GAN_INPUT_GRADIENT,
                        DMD_KEY_GAN_INPUT_GRADIENT,
                    )
                self.tensor_transfer_groups["real_score_to_generator"].transfer(
                    DMD_KEY_PRED_X0,
                    DMD_KEY_PRED_REAL_IMAGE,
                )
                if self.pipeline_config.ddmd.enabled:
                    self.tensor_transfer_groups["real_score_to_generator"].transfer(
                        DMD_KEY_PRED_REAL_COND_CA,
                        DMD_KEY_PRED_REAL_COND_CA,
                    )
                    self.tensor_transfer_groups["real_score_to_generator"].transfer(
                        DMD_KEY_PRED_REAL_UNCOND_CA,
                        DMD_KEY_PRED_REAL_UNCOND_CA,
                    )
                score_batch = score_request
            else:
                fake_score_output.rename(old_keys=DMD_KEY_PRED_X0, new_keys=DMD_KEY_PRED_FAKE_IMAGE)
                real_score_output.rename(old_keys=DMD_KEY_PRED_X0, new_keys=DMD_KEY_PRED_REAL_IMAGE)
                score_batch = fake_score_output.union(real_score_output)
            score_batch.meta_info = {"global_step": global_step}
        metrics_mgr.add_metric("time/score_forward", timer.last)

        with Timer(name="generator_update", logger=None) as timer:
            generator_update_refs = self.generator.finish_generator_step(score_batch, blocking=False)
            generator_update_output = DataProto.materialize_concat(data_refs=generator_update_refs)
            metrics_mgr.add_reduced_metrics(generator_update_output.meta_info.pop("metrics", {}))
        metrics_mgr.add_metric("time/generator_update", timer.last)

    def _run_fake_score_phase(
        self,
        prompt_batch: dict[str, Any],
        global_step: int,
        metrics_mgr: MetricsManager,
        real_batch: dict[str, Any] | None = None,
    ) -> None:
        """Update fake-score and its optional DMD2 classification head."""
        generation_input = dict(prompt_batch)
        if real_batch is not None and self.pipeline_config.share_prompt_embeddings:
            generation_input[DMD_KEY_REAL_PROMPTS] = real_batch[DMD_KEY_REAL_PROMPTS]
        batch = DataProto.from_single_dict(generation_input)
        batch.meta_info = {
            "global_step": global_step,
        }

        with Timer(name="fake_score_sample", logger=None) as timer:
            generation_refs = self.generator.generate_no_grad(batch, blocking=False)
            generated_batch = DataProto.materialize_concat(data_refs=generation_refs)
            metrics_mgr.add_reduced_metrics(generated_batch.meta_info.pop("metrics", {}))
            generated_batch.meta_info = {
                "global_step": global_step,
            }
            if self.gpu_tensor_transfer_enabled:
                self.tensor_transfer_groups["generator_to_fake_score"].transfer(
                    DMD_KEY_GENERATED,
                    DMD_KEY_GENERATED,
                    clear_source=True,
                )
            if real_batch is not None:
                generated_batch = generated_batch.union(DataProto.from_single_dict(real_batch))
        metrics_mgr.add_metric("time/fake_score_sample", timer.last)

        with Timer(name="fake_score_update", logger=None) as timer:
            fake_score_update_refs = self.fake_score.train_score_step(generated_batch, blocking=False)
            fake_score_update_output = DataProto.materialize_concat(data_refs=fake_score_update_refs)
            metrics_mgr.add_reduced_metrics(fake_score_update_output.meta_info.pop("metrics", {}))
        metrics_mgr.add_metric("time/fake_score_update", timer.last)
