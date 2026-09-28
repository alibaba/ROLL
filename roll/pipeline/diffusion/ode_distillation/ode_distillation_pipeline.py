"""Pipeline orchestration for ODE trajectory distillation."""

from __future__ import annotations

import json
import os
import re
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
from roll.pipeline.diffusion.ode_distillation.ode_distillation_config import ODEDistillationConfig
from roll.pipeline.diffusion.ode_distillation.ode_distillation_worker import (
    ODE_INPUT,
    ODE_PAIR_IDS,
    ODE_PROMPTS,
    ODE_STUDENT_ROLE,
    ODE_TARGET,
)
from roll.pipeline.diffusion.tensor_transfer import TensorTransferGroup
from roll.utils.checkpoint_manager import download_model
from roll.utils.functionals import clusters_have_disjoint_devices, reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.worker_state import WorkerState

logger = get_logger()
ODE_DISTILLATION_CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
ODE_DISTILLATION_DATA_STATE_KEY = "ode_distillation_data_state"
ODE_DISTILLATION_RESUME_ROLES = {ODE_STUDENT_ROLE}


class ODEDistillationPipeline(BasePipeline):
    """Distill teacher ODE trajectories into a causal student model."""

    def __init__(self, pipeline_config: ODEDistillationConfig) -> None:
        resume_manifest = self._prepare_resume(pipeline_config)
        super().__init__(pipeline_config)
        if self.resume_from_checkpoint:
            self.pipeline_config.resume_from_checkpoint = self.resume_from_checkpoint
            if resume_manifest is None or self.state.step != int(resume_manifest["pipeline_step"]):
                manifest_step = None if resume_manifest is None else resume_manifest.get("pipeline_step")
                raise ValueError(
                    f"ODE distillation checkpoint pipeline step mismatch: manifest={manifest_step}, "
                    f"state={self.state.step}"
                )
        self.pipeline_config.teacher.training_args.max_steps = self.pipeline_config.max_steps
        self.pipeline_config.student.training_args.max_steps = self.pipeline_config.max_steps
        self.teacher = Cluster(
            name=self.pipeline_config.teacher.name,
            worker_cls=self.pipeline_config.teacher.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.teacher,
        )
        self.student = Cluster(
            name=self.pipeline_config.student.name,
            worker_cls=self.pipeline_config.student.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.student,
        )
        if clusters_have_disjoint_devices([self.teacher, self.student]):
            ray.get(
                self.teacher.initialize(pipeline_config=self.pipeline_config, blocking=False)
                + self.student.initialize(pipeline_config=self.pipeline_config, blocking=False)
            )
        else:
            self.teacher.initialize(pipeline_config=self.pipeline_config, blocking=True)
            self.student.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self._initialize_tensor_transfer()

        dataset = (
            get_dataset(self.pipeline_config.teacher.data_args)
            .shuffle(seed=self.pipeline_config.seed)
            .select(range(self.pipeline_config.num_ode_pairs))
            .add_column(ODE_PAIR_IDS, list(range(self.pipeline_config.num_ode_pairs)))
        )
        self._initialize_dataloader(dataset)
        if self.resume_from_checkpoint:
            WorkerState.load_rng_state(
                load_dir=os.path.join(self.resume_from_checkpoint, "pipeline"),
                tag="pipeline",
            )
        self.checkpoint_clusters = [self.student]

    def _initialize_tensor_transfer(self) -> None:
        """Initialize the direct teacher-to-student tensor data plane."""
        self.tensor_transfer_group: TensorTransferGroup | None = None
        if not self.pipeline_config.gpu_tensor_transfer:
            return

        error = TensorTransferGroup.compatibility_error(self.teacher, self.student)
        if error is not None:
            raise ValueError(f"ODE distillation GPU tensor transfer requires matching role topologies: {error}")

        self.tensor_transfer_group = TensorTransferGroup(
            src_cluster=self.teacher,
            tgt_cluster=self.student,
            name="ode_teacher_to_student",
        )
        ray.get(
            [
                worker.set_gpu_tensor_transfer_enabled.remote(True)
                for cluster in (self.teacher, self.student)
                for worker in cluster.workers
            ]
        )

    @staticmethod
    def _load_resume_manifest(checkpoint_dir: str) -> dict[str, Any]:
        """Load and validate one committed ODE distillation checkpoint."""
        manifest_path = os.path.join(checkpoint_dir, ODE_DISTILLATION_CHECKPOINT_MANIFEST)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Incomplete ODE distillation checkpoint: missing {manifest_path}")
        try:
            with open(manifest_path, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid ODE distillation checkpoint manifest at {manifest_path}") from error

        if not isinstance(manifest, dict):
            raise ValueError(f"ODE distillation checkpoint manifest at {manifest_path} must be a JSON object")
        checkpoint_id = os.path.basename(os.path.normpath(checkpoint_dir))
        checkpoint_match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_id)
        expected_roles = sorted(ODE_DISTILLATION_RESUME_ROLES)
        if manifest.get("format_version") != 1:
            raise ValueError(f"Unsupported ODE distillation checkpoint manifest version at {manifest_path}")
        if manifest.get("checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"ODE distillation checkpoint manifest id mismatch: "
                f"expected {checkpoint_id}, got {manifest.get('checkpoint_id')}"
            )
        if checkpoint_match is None or manifest.get("global_step") != int(checkpoint_match.group(1)):
            raise ValueError(f"ODE distillation checkpoint global step mismatch at {manifest_path}")
        if not isinstance(manifest.get("pipeline_step"), int):
            raise ValueError(f"ODE distillation checkpoint manifest is missing an integer pipeline_step")
        if manifest.get("roles") != expected_roles:
            raise ValueError(
                f"A resumable ODE distillation checkpoint must contain {expected_roles} roles, "
                f"got {manifest.get('roles')}"
            )

        required_files = (
            os.path.join("pipeline", "worker_state_pipeline.json"),
            os.path.join("pipeline", "rng_state_pipeline.pth"),
            os.path.join(ODE_STUDENT_ROLE, "dcp", ".metadata"),
        )
        missing_files = [
            relative_path
            for relative_path in required_files
            if not os.path.isfile(os.path.join(checkpoint_dir, relative_path))
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Incomplete ODE distillation checkpoint at {checkpoint_dir}; missing {missing_files}"
            )
        dcp_dir = os.path.join(checkpoint_dir, ODE_STUDENT_ROLE, "dcp")
        if not any(entry.is_file() and entry.name.endswith(".distcp") for entry in os.scandir(dcp_dir)):
            raise FileNotFoundError(
                f"Incomplete ODE distillation checkpoint at {checkpoint_dir}; missing DCP shard files"
            )
        return manifest

    @classmethod
    def _prepare_resume(cls, pipeline_config: ODEDistillationConfig) -> dict[str, Any] | None:
        """Resolve the latest complete checkpoint before BasePipeline restores state."""
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
                        if match is None or not entry.is_dir():
                            continue
                        try:
                            cls._load_resume_manifest(entry.path)
                        except (FileNotFoundError, ValueError) as error:
                            logger.warning(f"Skipping incomplete ODE distillation checkpoint {entry.path}: {error}")
                            continue
                        candidates.append((entry.stat().st_mtime, int(match.group(1)), entry.path))
            resume_checkpoint = max(candidates)[2] if candidates else False
        elif isinstance(resume_checkpoint, str):
            resume_checkpoint = download_model(resume_checkpoint)

        if not resume_checkpoint:
            pipeline_config.resume_from_checkpoint = False
            return None

        resume_manifest = cls._load_resume_manifest(resume_checkpoint)
        pipeline_config.resume_from_checkpoint = resume_checkpoint
        if pipeline_config.checkpoint_config.get("type") == "file_system":
            pipeline_config.checkpoint_config["output_dir"] = os.path.dirname(resume_checkpoint)
        return resume_manifest

    def _initialize_dataloader(self, dataset: Any) -> None:
        """Build the prompt dataloader and restore its checkpointed cursor."""
        global_batch_size = (
            self.teacher.dp_size * self.pipeline_config.teacher.training_args.per_device_train_batch_size
        )
        student_global_batch_size = (
            self.student.dp_size * self.pipeline_config.student.training_args.per_device_train_batch_size
        )
        if global_batch_size != student_global_batch_size:
            raise ValueError(
                "ODE distillation teacher and student global batch sizes must match: "
                f"teacher={global_batch_size}, student={student_global_batch_size}"
            )
        data_seed = self.pipeline_config.teacher.training_args.data_seed
        if data_seed is None:
            data_seed = self.pipeline_config.seed
        data_state = self.state.kv.get(ODE_DISTILLATION_DATA_STATE_KEY)
        legacy_resume = self.resume_from_checkpoint and data_state is None
        if data_state is None:
            data_state = {
                "seed": int(data_seed),
                "epoch": 0,
                "batch_offset": 0,
            }
            self.state.kv[ODE_DISTILLATION_DATA_STATE_KEY] = data_state
        self._data_generator = torch.Generator()
        self.dataloader = DataLoader(
            dataset,
            batch_size=global_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.pipeline_config.teacher.training_args.dataloader_num_workers,
            collate_fn=self._collate_prompts,
            generator=self._data_generator,
        )
        if len(self.dataloader) == 0:
            raise ValueError("ODE distillation dataset must contain at least one global training batch")
        if legacy_resume:
            consumed_batches = (
                (self.state.step + 1)
                * self.pipeline_config.student.training_args.gradient_accumulation_steps
            )
            data_state["epoch"], data_state["batch_offset"] = divmod(consumed_batches, len(self.dataloader))
        self._dataloader_iter: Iterator[dict[str, Any]] | None = None

    @staticmethod
    def _collate_prompts(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            ODE_PROMPTS: np.array([item["text"] for item in batch], dtype=object),
            ODE_PAIR_IDS: torch.tensor([item[ODE_PAIR_IDS] for item in batch], dtype=torch.int64),
        }

    def _next_batch(self) -> dict[str, Any]:
        data_state = self.state.kv[ODE_DISTILLATION_DATA_STATE_KEY]
        if self._dataloader_iter is None:
            self._data_generator.manual_seed(int(data_state["seed"]) + int(data_state["epoch"]))
            self._dataloader_iter = iter(self.dataloader)
            for _ in range(int(data_state["batch_offset"])):
                next(self._dataloader_iter)
        batch = next(self._dataloader_iter)
        data_state["batch_offset"] = int(data_state["batch_offset"]) + 1
        if data_state["batch_offset"] == len(self.dataloader):
            data_state["epoch"] = int(data_state["epoch"]) + 1
            data_state["batch_offset"] = 0
            self._dataloader_iter = None
        return batch

    def do_checkpoint(self, global_step: int, is_last_step: bool | None = None) -> None:
        """Commit a checkpoint only after student and pipeline uploads complete."""
        futures.wait(self.resume_futures)
        self.resume_futures.clear()

        if is_last_step is None:
            is_last_step = global_step == self.pipeline_config.max_steps
        should_save_interval = (
            self.pipeline_config.save_steps > 0
            and global_step > 0
            and global_step % self.pipeline_config.save_steps == 0
        )
        if not (should_save_interval or is_last_step):
            return

        metrics = self.state.log_history[-1]
        metrics["system/step"] = global_step

        checkpoint_refs = [
            cluster.do_checkpoint(global_step=global_step, is_last_step=is_last_step, blocking=False)
            for cluster in self.checkpoint_clusters
        ]
        for refs in checkpoint_refs:
            checkpoint_metrics = DataProto.materialize_concat(data_refs=refs)
            metrics.update(reduce_metrics(checkpoint_metrics.meta_info.pop("metrics", {})))

        ckpt_id = f"checkpoint-{global_step}"
        local_checkpoint = self.checkpoint_manager.uploader is None
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
            "keep_local_file": self.pipeline_config.checkpoint_config.get("keep_local_file", False),
        }
        pipeline_upload: futures.Future[Any] | None = None
        if not local_checkpoint:
            if self.pipeline_config.checkpoint_config.get("async_upload", True) and not is_last_step:
                pipeline_upload = self.executor.submit(
                    self.checkpoint_manager.upload,
                    **upload_kwargs,
                )
            else:
                self.checkpoint_manager.upload(**upload_kwargs)

        upload_refs: list[Any] = []
        for cluster in self.checkpoint_clusters:
            upload_refs.extend(cluster.wait_for_checkpoint_upload(ckpt_id=ckpt_id, blocking=False))
        ray.get(upload_refs)
        if pipeline_upload is not None:
            pipeline_upload.result()

        manifest_upload_root = (
            pipeline_upload_root
            if local_checkpoint
            else os.path.join(self.pipeline_config.output_dir, "manifest", ckpt_id)
        )
        os.makedirs(manifest_upload_root, exist_ok=True)
        manifest_path = os.path.join(manifest_upload_root, ODE_DISTILLATION_CHECKPOINT_MANIFEST)
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(
                {
                    "format_version": 1,
                    "checkpoint_id": ckpt_id,
                    "global_step": global_step,
                    "pipeline_step": self.state.step,
                    "roles": sorted(ODE_DISTILLATION_RESUME_ROLES),
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
                keep_local_file=self.pipeline_config.checkpoint_config.get("keep_local_file", False),
            )
        self._cleanup_old_checkpoints()

    def run(self) -> None:
        """Run online teacher trajectory generation followed by one student update."""
        global_step = self.state.step + 1
        gradient_accumulation_steps = self.pipeline_config.student.training_args.gradient_accumulation_steps
        while global_step < self.pipeline_config.max_steps:
            with Timer("ode_distillation_step", logger=None) as timer:
                metrics: dict[str, float] = {}
                student_loss = 0.0
                for accumulation_step in range(gradient_accumulation_steps):
                    batch = DataProto.from_single_dict(self._next_batch())
                    batch.meta_info = {
                        "global_step": global_step,
                        "accumulation_step": accumulation_step,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                    }
                    teacher_batch = DataProto.materialize_concat(
                        data_refs=self.teacher.make_regression_batch(batch, blocking=False)
                    )
                    metrics.update(reduce_metrics(teacher_batch.meta_info.pop("metrics", {})))
                    teacher_batch.meta_info = dict(batch.meta_info)
                    if self.tensor_transfer_group is not None:
                        self.tensor_transfer_group.transfer(src_slot=ODE_INPUT, tgt_slot=ODE_INPUT)
                        self.tensor_transfer_group.transfer(src_slot=ODE_TARGET, tgt_slot=ODE_TARGET)
                    student_output = DataProto.materialize_concat(
                        data_refs=self.student.train_step(teacher_batch, blocking=False)
                    )
                    student_metrics = reduce_metrics(student_output.meta_info.pop("metrics", {}))
                    student_loss += student_metrics.pop("student/loss")
                    metrics.update(student_metrics)

                metrics["student/loss"] = student_loss / gradient_accumulation_steps

            display_step = global_step + 1
            metrics["time/step"] = timer.last
            metrics["system/step"] = display_step
            self.state.step = global_step
            self.state.log_history.append(metrics)
            self.tracker.log(values=metrics, step=display_step)
            self.do_checkpoint(display_step, is_last_step=display_step == self.pipeline_config.max_steps)
            if global_step % self.pipeline_config.logging_steps == 0:
                logger.info(
                    f"step {display_step}: student/loss = {metrics['student/loss']:.4f}, "
                    f"student/grad_norm = {metrics['student/grad_norm']:.4f}, time/step = {timer.last:.2f}s"
                )
            global_step += 1

        logger.info("ODE distillation pipeline complete!")
