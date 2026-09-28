"""Teacher and causal-student workers for ODE trajectory distillation."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from collections.abc import Mapping
from concurrent import futures
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from codetiming import Timer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.pipeline.diffusion.models.base import DiffusionAdapter, EncodedPrompt
from roll.pipeline.diffusion.models.registry import get_diffusion_model_adapter, get_self_forcing_module
from roll.pipeline.diffusion.models.scheduling_flow_match_sde_discrete import FlowMatchSDEDiscreteScheduler
from roll.pipeline.diffusion.ode_distillation.ode_distillation_config import ODEDistillationConfig
from roll.pipeline.diffusion.tensor_transfer import TensorTransferWorker
from roll.platforms import current_platform
from roll.utils.checkpoint_manager import CheckpointManager
from roll.utils.context_managers import state_offload_manger
from roll.utils.offload_states import OffloadStateType

ODE_INPUT = "ode_input"
ODE_TARGET = "ode_target"
ODE_TIMESTEP = "ode_timestep"
ODE_PROMPTS = "prompts"
ODE_PAIR_IDS = "ode_pair_ids"
ODE_STUDENT_ROLE = "student"
_FORWARD_STATES = (OffloadStateType.model_params, OffloadStateType.other_params)
_ODE_PAIR_CACHE_VERSION = 1
_ODE_PAIR_CACHE_TRAJECTORY_KEY = "trajectory"


def _canonicalize_cache_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _canonicalize_cache_value(value.tolist())
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_cache_value(item) for item in value]
    raise TypeError(f"Unsupported ODE cache fingerprint value: {type(value).__module__}.{type(value).__qualname__}")


class BaseODEDistillationWorker(TensorTransferWorker):
    """Shared model and adapter wiring for the two ODE distillation roles."""

    def __init__(self, worker_config: WorkerConfig) -> None:
        super().__init__(worker_config=worker_config)
        self.strategy: Any = None
        self.diffusion_adapter: DiffusionAdapter | None = None
        self.prompt_encoder: Any = None

    def _initialize_runtime(
        self,
        pipeline_config: ODEDistillationConfig,
        model_provider: Callable[..., torch.nn.Module] | None,
    ) -> None:
        super().initialize(pipeline_config)
        self.device = (
            torch.device("cpu")
            if current_platform.device_type == "cpu"
            else torch.device(f"{current_platform.device_type}:{current_platform.current_device()}")
        )
        self.strategy = create_strategy(worker=self)
        if self.worker_config.strategy_args.strategy_name == "fsdp2_train":
            # FSDP2 converts global max_steps to local DP updates during initialization.
            data_parallel_size = self.world_size // int(self.worker_config.model_args.ulysses_size or 1)
            self.worker_config.training_args.max_steps *= data_parallel_size
        adapter_type = get_diffusion_model_adapter(pipeline_config.diffusion_model_variant)
        self.strategy.initialize(model_provider)

        self.model = self.strategy.model
        self.diffusion_adapter = adapter_type(transformer=self.model)
        self.prompt_encoder = getattr(self.strategy, "prompt_encoder", None) or getattr(
            self.model, "prompt_encoder", None
        )
        text_encoder = self.prompt_encoder.text_encoder
        if not pipeline_config.is_offload_states:
            text_encoder.to(self.device)
        self.dtype = self.strategy.param_dtype

    def _encode_prompt(self, prompts: list[str]) -> EncodedPrompt:
        prompt_encoder_context = getattr(self.prompt_encoder, "device_context", None)
        device_context = (
            prompt_encoder_context(
                self.device,
                offload_after=self.pipeline_config.is_offload_states,
            )
            if callable(prompt_encoder_context)
            else nullcontext()
        )
        with device_context:
            return self.diffusion_adapter.encode_prompt(
                prompt_encoder=self.prompt_encoder,
                prompt_inputs={
                    "prompts": prompts,
                    "device": self.device,
                    "dtype": self.dtype,
                },
            )

    @staticmethod
    def _extract_prompts(data: DataProto) -> list[str]:
        prompts = data.non_tensor_batch[ODE_PROMPTS]
        return prompts.tolist() if isinstance(prompts, np.ndarray) else list(prompts)


class ODETrajectoryTeacherWorker(BaseODEDistillationWorker):
    """Frozen teacher role that produces one sampled ODE-regression batch."""

    def _build_cache_config_fingerprint(self) -> str:
        model_args = self.worker_config.model_args
        strategy_args = self.worker_config.strategy_args
        fingerprint_payload = {
            "cache_version": _ODE_PAIR_CACHE_VERSION,
            "pipeline": {
                "denoising_step_list": self.pipeline_config.denoising_step_list,
                "diffusion_model_variant": self.pipeline_config.diffusion_model_variant,
                "guidance_scale": self.pipeline_config.guidance_scale,
                "latent_shape": self.pipeline_config.latent_shape,
                "negative_prompt": self.pipeline_config.negative_prompt,
                "num_train_timesteps": self.pipeline_config.num_train_timesteps,
                "seed": self.pipeline_config.seed,
                "teacher_num_steps": self.pipeline_config.teacher_num_steps,
                "timestep_shift": self.pipeline_config.timestep_shift,
            },
            "scheduler_class": f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}",
            "teacher": {
                "adapter_class": (
                    f"{type(self.diffusion_adapter).__module__}.{type(self.diffusion_adapter).__qualname__}"
                ),
                "model_args": model_args.to_dict(),
                "runtime_dtype": str(self.dtype),
                "strategy_config": strategy_args.strategy_config,
                "strategy_name": strategy_args.strategy_name,
            },
        }
        serialized_payload = json.dumps(
            _canonicalize_cache_value(fingerprint_payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()

    def _load_cached_trajectory(self, cache_path: str, pair_id: int, prompt: str) -> torch.Tensor:
        cached_record = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(cached_record, Mapping):
            raise ValueError(
                f"Unsupported ODE pair cache format at {cache_path}; "
                "remove the legacy cache file before restarting"
            )

        expected_metadata = {
            "cache_version": _ODE_PAIR_CACHE_VERSION,
            "config_fingerprint": self._cache_config_fingerprint,
            "pair_id": pair_id,
            "prompt": prompt,
        }
        for key, expected_value in expected_metadata.items():
            if cached_record.get(key) != expected_value:
                raise ValueError(
                    f"ODE pair cache metadata mismatch for {key!r} at {cache_path}; "
                    "use a separate cache directory or remove the stale cache file"
                )

        trajectory = cached_record.get(_ODE_PAIR_CACHE_TRAJECTORY_KEY)
        if not isinstance(trajectory, torch.Tensor):
            raise ValueError(f"ODE pair cache at {cache_path} does not contain a trajectory tensor")
        expected_shape = (len(self.pipeline_config.denoising_step_list) + 1, *self.pipeline_config.latent_shape)
        if tuple(trajectory.shape) != expected_shape:
            raise ValueError(
                f"ODE pair cache trajectory at {cache_path} has shape {tuple(trajectory.shape)}, "
                f"expected {expected_shape}"
            )
        if trajectory.dtype != torch.float16:
            raise ValueError(
                f"ODE pair cache trajectory at {cache_path} has dtype {trajectory.dtype}, expected torch.float16"
            )
        return trajectory.to(device=self.device, dtype=self.dtype)

    def _save_cached_trajectory(
        self,
        cache_path: str,
        trajectory: torch.Tensor,
        pair_id: int,
        prompt: str,
    ) -> None:
        cached_record: dict[str, Any] = {
            "cache_version": _ODE_PAIR_CACHE_VERSION,
            "config_fingerprint": self._cache_config_fingerprint,
            "pair_id": pair_id,
            "prompt": prompt,
            _ODE_PAIR_CACHE_TRAJECTORY_KEY: trajectory.cpu(),
        }
        temporary_path = f"{cache_path}.{self.rank}.tmp"
        try:
            torch.save(cached_record, temporary_path)
            os.replace(temporary_path, cache_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config: ODEDistillationConfig) -> None:
        """Initialize the non-causal teacher and its flow-matching scheduler."""
        self._initialize_runtime(pipeline_config, None)
        self.model.requires_grad_(False)
        self.model.eval()
        self.scheduler = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=pipeline_config.num_train_timesteps,
            shift=pipeline_config.timestep_shift,
        )
        if pipeline_config.ode_pair_cache_dir is not None:
            self._cache_config_fingerprint = self._build_cache_config_fingerprint()
            os.makedirs(pipeline_config.ode_pair_cache_dir, exist_ok=True)
        if pipeline_config.is_offload_states:
            self.strategy.offload_states()
        self.logger.info(f"{self.worker_name} initialized as ODETrajectoryTeacherWorker")

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def make_regression_batch(self, data: DataProto) -> DataProto:
        """Generate a teacher ODE and retain the randomly selected student inputs."""
        metrics: dict[str, float] = {}
        data = data.to(self.device)
        prompts = self._extract_prompts(data)
        pair_ids = [int(pair_id) for pair_id in data.batch[ODE_PAIR_IDS].tolist()]
        cache_paths = [
            (
                os.path.join(self.pipeline_config.ode_pair_cache_dir, f"{pair_id:05d}.pt")
                if self.pipeline_config.ode_pair_cache_dir is not None
                else None
            )
            for pair_id in pair_ids
        ]
        cache_hits = [cache_path is not None and os.path.exists(cache_path) for cache_path in cache_paths]
        cache_miss_indices = [index for index, cache_hit in enumerate(cache_hits) if not cache_hit]
        offload_context = (
            state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/make_regression_batch",
                is_offload_states=True,
                load_kwargs={"include": _FORWARD_STATES},
            )
            if self.pipeline_config.is_offload_states and cache_miss_indices
            else nullcontext()
        )
        with (
            offload_context,
            torch.no_grad(),
            getattr(self.strategy, "model_fwd_context", nullcontext()),
        ):
            # Match the original extra-one-step ODE grid: run N model evaluations,
            # then take the final Euler step from the last non-zero sigma to zero.
            sigmas = np.linspace(1.0, 0.0, self.pipeline_config.teacher_num_steps + 1)[:-1]
            self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
            timesteps = self.scheduler.timesteps
            node_positions = torch.tensor(
                [
                    (self.pipeline_config.num_train_timesteps - step)
                    * self.pipeline_config.teacher_num_steps
                    // self.pipeline_config.num_train_timesteps
                    for step in [*self.pipeline_config.denoising_step_list, 0]
                ],
                device=self.device,
            )
            node_timesteps = torch.cat((timesteps, timesteps.new_zeros(1)))[node_positions]
            trajectories: dict[int, torch.Tensor] = {}
            for index, (cache_path, pair_id, prompt_text) in enumerate(zip(cache_paths, pair_ids, prompts)):
                if cache_hits[index]:
                    trajectories[index] = self._load_cached_trajectory(
                        cache_path=cache_path,
                        pair_id=pair_id,
                        prompt=prompt_text,
                    )

            if cache_miss_indices:
                missing_prompts = [prompts[index] for index in cache_miss_indices]
                prompt = self._encode_prompt(missing_prompts)
                negative_prompt = self._encode_prompt(
                    [self.pipeline_config.negative_prompt] * len(cache_miss_indices)
                )
                latents = torch.stack(
                    [
                        torch.randn(
                            self.pipeline_config.latent_shape,
                            generator=torch.Generator(device=self.device).manual_seed(
                                int(self.pipeline_config.seed) + pair_ids[index]
                            ),
                            device=self.device,
                            dtype=self.dtype,
                        )
                        for index in cache_miss_indices
                    ]
                )
                trajectory_nodes = [latents.clone()]
                next_node = 1
                for step_index, timestep in enumerate(timesteps):
                    model_timestep = timestep.expand(latents.shape[0], latents.shape[1])
                    prediction = self.diffusion_adapter.forward_step(
                        latents=latents,
                        timestep=model_timestep,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        # The adapter uses cond + scale * (cond - uncond), while
                        # standard CFG is uncond + cfg * (cond - uncond).
                        guidance_scale=self.pipeline_config.guidance_scale - 1.0,
                    )
                    flow_pred = prediction.flow_pred
                    latents = self.scheduler.step(
                        model_output=flow_pred,
                        timestep=timestep,
                        sample=latents,
                        sde_type="ode",
                        logprobs=False,
                        return_dict=False,
                    )[0]
                    if next_node < len(node_positions) and step_index + 1 == int(node_positions[next_node]):
                        trajectory_nodes.append(latents.clone())
                        next_node += 1
                # The original pipeline writes ODE trajectories to LMDB in fp16.
                # Quantize here as well so cached and freshly generated pairs have
                # identical student inputs.
                generated_trajectories = torch.stack(trajectory_nodes, dim=1).half()
                for generated_index, batch_index in enumerate(cache_miss_indices):
                    cache_path = cache_paths[batch_index]
                    trajectory = generated_trajectories[generated_index]
                    if cache_path is not None:
                        self._save_cached_trajectory(
                            cache_path=cache_path,
                            trajectory=trajectory,
                            pair_id=pair_ids[batch_index],
                            prompt=prompts[batch_index],
                        )
                    trajectories[batch_index] = trajectory.to(dtype=self.dtype)

            trajectory = torch.stack([trajectories[index] for index in range(len(prompts))])
            num_frames = trajectory.shape[2]
            num_blocks = math.ceil(num_frames / self.pipeline_config.num_frame_per_block)
            node_generator = torch.Generator(device=self.device).manual_seed(
                int(self.pipeline_config.seed)
                + self.pipeline_config.num_ode_pairs
                + (
                    int(data.meta_info["global_step"])
                    * int(data.meta_info["gradient_accumulation_steps"])
                    + int(data.meta_info["accumulation_step"])
                )
                * self.world_size
                + int(self.rank_info.dp_rank)
            )
            node_indices = torch.randint(
                0,
                len(self.pipeline_config.denoising_step_list),
                (len(prompts), num_blocks),
                generator=node_generator,
                device=self.device,
            ).repeat_interleave(self.pipeline_config.num_frame_per_block, dim=1)[:, :num_frames]
            ode_input = torch.gather(
                trajectory,
                dim=1,
                index=node_indices.reshape(len(prompts), 1, num_frames, 1, 1, 1).expand(
                    -1, -1, -1, *trajectory.shape[-3:]
                ),
            ).squeeze(1)
            selected_timesteps = node_timesteps[node_indices]
        tensors = {ODE_TIMESTEP: selected_timesteps.cpu()}
        if self._gpu_tensor_transfer_enabled:
            self._tensor_transfer_slots[ODE_INPUT] = ode_input.detach().contiguous()
            self._tensor_transfer_slots[ODE_TARGET] = trajectory[:, -1].detach().contiguous()
        else:
            tensors[ODE_INPUT] = ode_input.cpu()
            tensors[ODE_TARGET] = trajectory[:, -1].cpu()
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors={ODE_PROMPTS: prompts},
            meta_info={"metrics": metrics},
        )


class ODEDistillationStudentWorker(BaseODEDistillationWorker):
    """Causal student role trained by teacher trajectory regression."""

    def __init__(self, worker_config: WorkerConfig) -> None:
        super().__init__(worker_config=worker_config)
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.worker_config.checkpoint_config)
        self._checkpoint_uploads: dict[str, futures.Future[Any]] = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config: ODEDistillationConfig) -> None:
        """Initialize the provider-built causal student and optimizer."""
        self.worker_config.strategy_args.strategy_config["init_tokenizer_processor"] = False
        model_provider = importlib.import_module(
            get_self_forcing_module(pipeline_config.diffusion_model_variant)
        ).build_model
        self._initialize_runtime(pipeline_config, model_provider)
        self.optimizer = self.strategy.optimizer
        if self.pipeline_config.resume_from_checkpoint:
            self.strategy.load_checkpoint(
                load_dir=os.path.join(self.pipeline_config.resume_from_checkpoint, ODE_STUDENT_ROLE)
            )
        self.strategy.tokenizer = self.prompt_encoder.tokenizer
        if self.pipeline_config.is_offload_states:
            self.strategy.offload_states()
        self.logger.info(f"{self.worker_name} initialized as ODEDistillationStudentWorker")

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def train_step(self, data: DataProto) -> DataProto:
        """Regress the causal student's x0 prediction to the teacher ODE endpoint."""
        metrics: dict[str, float] = {}
        offload_context = (
            state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/train_step",
                is_offload_states=True,
            )
            if self.pipeline_config.is_offload_states
            else nullcontext()
        )
        with offload_context:
            data = data.to(self.device)
            if self._gpu_tensor_transfer_enabled:
                ode_input = self._tensor_transfer_slots.pop(ODE_INPUT).to(dtype=self.dtype)
                ode_target = self._tensor_transfer_slots.pop(ODE_TARGET).to(dtype=self.dtype)
            else:
                ode_input = data.batch[ODE_INPUT].to(dtype=self.dtype)
                ode_target = data.batch[ODE_TARGET].to(dtype=self.dtype)
            timestep = data.batch[ODE_TIMESTEP]
            prompt = self._encode_prompt(self._extract_prompts(data))

            # ODE regression is deterministic teacher forcing; eval mode still
            # records gradients while disabling any model-side stochastic layers.
            self.model.eval()
            accumulation_step = int(data.meta_info["accumulation_step"])
            gradient_accumulation_steps = int(data.meta_info["gradient_accumulation_steps"])
            if accumulation_step == 0:
                self.optimizer.zero_grad(set_to_none=True)
            with getattr(self.strategy, "model_fwd_context", nullcontext()):
                prediction = self.diffusion_adapter.forward_step(
                    latents=ode_input,
                    timestep=timestep,
                    prompt=prompt,
                )
                flow_pred = prediction.flow_pred
                sigma = (timestep.float() / self.pipeline_config.num_train_timesteps).view(
                    *timestep.shape, 1, 1, 1
                )
                pred_x0 = ode_input.float() - sigma * flow_pred.float()
                mask = timestep != 0
                loss = F.mse_loss(pred_x0[mask], ode_target.float()[mask])

            scaler = getattr(self.strategy, "scaler", None)
            with getattr(self.strategy, "model_bwd_context", nullcontext()):
                if scaler is None:
                    (loss / gradient_accumulation_steps).backward()
                else:
                    scaler.scale(loss / gradient_accumulation_steps).backward()

            metrics["student/loss"] = float(loss.detach())
            if accumulation_step + 1 == gradient_accumulation_steps:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                grad_norm = self.strategy.clip_grad_norm(self.pipeline_config.max_grad_norm)
                optimizer_stepped = False
                if scaler is not None:
                    scale_before_step = scaler.get_scale()
                    scaler.step(self.optimizer)
                    scaler.update()
                    optimizer_stepped = scaler.get_scale() >= scale_before_step
                elif torch.isfinite(grad_norm).all():
                    self.optimizer.step()
                    optimizer_stepped = True
                if optimizer_stepped and self.strategy.scheduler is not None:
                    self.strategy.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                metrics["student/grad_norm"] = float(grad_norm.detach().float().mean())

        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step: int, is_last_step: bool = False) -> DataProto:
        """Save a resumable FSDP checkpoint and a reusable Diffusers transformer."""
        ckpt_id = f"checkpoint-{global_step}"
        upload_root = (
            os.path.join(self.pipeline_config.output_dir, ckpt_id)
            if self.checkpoint_manager.uploader is None
            else os.path.join(self.pipeline_config.output_dir, ODE_STUDENT_ROLE, ckpt_id)
        )
        save_dir = os.path.join(upload_root, ODE_STUDENT_ROLE)
        with Timer("do_checkpoint", logger=None) as total_timer:
            checkpoint_metrics: dict[str, float] = {}
            strategy_checkpoint_uploader = self.strategy.checkpoint_manager.uploader
            strategy_async_save = self.strategy.async_save_strategy
            try:
                # The pipeline coordinates component uploads and commits the manifest.
                # Reuse strategy serialization synchronously without starting its upload.
                self.strategy.checkpoint_manager.uploader = None
                self.strategy.async_save_strategy = False
                checkpoint_metrics.update(
                    self.strategy.save_checkpoint(
                        save_dir,
                        global_step,
                        ckpt_id,
                        is_last_step=is_last_step,
                    )
                )
            finally:
                self.strategy.checkpoint_manager.uploader = strategy_checkpoint_uploader
                self.strategy.async_save_strategy = strategy_async_save
                if self.pipeline_config.is_offload_states:
                    with Timer("offload_states", logger=None) as offload_timer:
                        self.strategy.offload_states()
                    checkpoint_metrics["offload_states"] = offload_timer.last
            with Timer("dcp_barrier", logger=None) as dcp_barrier_timer:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            checkpoint_metrics["dcp_barrier"] = dcp_barrier_timer.last

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0 and self.checkpoint_manager.uploader is not None:
                checkpoint_config = self.worker_config.checkpoint_config or {}
                upload_kwargs = {
                    "ckpt_id": ckpt_id,
                    "local_state_path": upload_root,
                    "keep_local_file": checkpoint_config.get("keep_local_file", False),
                }
                with Timer("upload", logger=None) as upload_timer:
                    if checkpoint_config.get("async_upload", True) and not is_last_step:
                        self._checkpoint_uploads[ckpt_id] = self.strategy.thread_executor.submit(
                            self.checkpoint_manager.upload,
                            **upload_kwargs,
                        )
                    else:
                        self.checkpoint_manager.upload(**upload_kwargs)
                checkpoint_metrics["upload_submit"] = upload_timer.last

        metrics = {f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last}
        metrics.update(
            {f"time/{self.cluster_name}/do_checkpoint/{key}": value for key, value in checkpoint_metrics.items()}
        )
        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wait_for_checkpoint_upload(self, ckpt_id: str) -> None:
        """Wait until the shared student checkpoint upload has completed."""
        upload = self._checkpoint_uploads.pop(ckpt_id, None)
        if upload is not None:
            upload.result()
