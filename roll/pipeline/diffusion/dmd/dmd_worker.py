"""DMD worker roles for generator and score models."""

from __future__ import annotations

import importlib
import os
import random
import types
from contextlib import nullcontext
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from codetiming import Timer
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler

from roll.configs.worker_config import WorkerConfig
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.models.model_providers import default_diffusion_model_provider
from roll.pipeline.diffusion.dmd.dmd_algorithm import (
    DMDFlowMatchScheduler,
    DMD_KEY_DENOISED_TIMESTEP_FROM,
    DMD_KEY_DENOISED_TIMESTEP_TO,
    DMD_KEY_GAN_INPUT_GRADIENT,
    DMD_KEY_GENERATED,
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
    DMD_KEY_TIMESTEP,
    DMD_KEY_TIMESTEP_CA,
    DMD_NEGATIVE_PROMPT_TENSOR_PREFIX,
    DMD_PROMPT_TENSOR_PREFIX,
    DMD_REAL_PROMPT_TENSOR_PREFIX,
    DMDResidualConv3DHead,
    compute_ddmd_generator_loss,
    compute_dmd_fake_score_loss,
    compute_dmd_gan_classification_loss,
    compute_dmd_gan_generator_loss,
    compute_dmd_gan_generator_surrogate,
    compute_dmd_generator_loss,
    generate_self_forcing_sample,
    get_score_timestep_window,
    latent_shape_from_vae_config,
    sample_denoising_indices,
    sample_dmd_timesteps,
    shift_dmd_timesteps,
)
from roll.pipeline.diffusion.dmd.dmd_config import DMDConfig, FSDP2_TRAIN_STRATEGY
from roll.pipeline.diffusion.models.base import (
    DiffusionAdapter,
    EncodedPrompt,
)
from roll.pipeline.diffusion.models.registry import (
    get_diffusion_model_adapter,
    get_self_forcing_module,
)
from roll.pipeline.diffusion.tensor_transfer import TensorTransferWorker
from roll.platforms import current_platform
from roll.utils.checkpoint_manager import CheckpointManager
from roll.utils.context_managers import state_offload_manger
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

logger = get_logger()

_RNG_GENERATOR_TRAIN_SAMPLE = 0
_RNG_GENERATOR_TRAIN_EXIT_STEP = 1
_RNG_GENERATOR_QUERY_TIMESTEP = 2
_RNG_GENERATOR_QUERY_NOISE = 3
_RNG_FAKE_SAMPLE = 4
_RNG_FAKE_SAMPLE_EXIT_STEP = 5
_RNG_FAKE_CRITIC_TIMESTEP = 6
_RNG_FAKE_CRITIC_NOISE = 7
_RNG_GAN_GENERATOR_TIMESTEP = 8
_RNG_GAN_GENERATOR_NOISE = 9
_RNG_GAN_CLASSIFICATION_TIMESTEP = 10
_RNG_GAN_CLASSIFICATION_NOISE = 11
_RNG_DDMD_CA_TIMESTEP = 12
_RNG_DDMD_CA_NOISE = 13
_FORWARD_STATE_TYPES = (OffloadStateType.model_params, OffloadStateType.other_params)


class ModelEMA:
    """Exponential moving average shadow for one trainable model."""

    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: self._local_parameter(param).detach().float().cpu().clone()
            for name, param in module.named_parameters()
        }

    @staticmethod
    def _local_parameter(param: nn.Parameter) -> torch.Tensor:
        """Return the local FSDP2 shard or the ordinary parameter tensor."""
        to_local = getattr(param, "to_local", None)
        return to_local() if callable(to_local) else param

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        """Blend current module parameters into the CPU shadow."""
        for name, param in module.named_parameters():
            current = self._local_parameter(param).detach().float().cpu()
            self.shadow[name].mul_(self.decay).add_(current, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return EMA shadow state."""
        return self.shadow

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load EMA shadow state."""
        self.shadow = {key: value.detach().float().cpu().clone() for key, value in state_dict.items()}


class BaseDMDWorker(TensorTransferWorker):
    """Base class for DMD Ray worker roles."""

    role_name: str = ""

    def __init__(self, worker_config: WorkerConfig) -> None:
        super().__init__(worker_config=worker_config)
        self.strategy = None
        self.models = nn.ModuleDict()
        self.prompt_encoder: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.classification_head: nn.Module | None = None
        self.classification_optimizer: torch.optim.Optimizer | None = None
        self.classification_scheduler: Any | None = None
        self.model_ema: ModelEMA | None = None
        self.diffusion_adapter: DiffusionAdapter | None = None
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.worker_config.checkpoint_config)
        self._checkpoint_uploads: dict[str, Any] = {}
        self._last_optimizer_stepped = False
        self.step = 0

    def _make_algorithm_generator(self, data: DataProto, stream: int) -> torch.Generator:
        """Create a deterministic RNG stream independent from backend model execution."""
        seed = int(
            np.random.SeedSequence(
                [
                    int(self.pipeline_config.seed),
                    int(data.meta_info["global_step"]),
                    int(self.rank_info.dp_rank),
                    stream,
                ]
            ).generate_state(1, dtype=np.uint64)[0]
        )
        return torch.Generator(device=self.device).manual_seed(seed)

    def _initialize_components(
        self,
        pipeline_config: DMDConfig,
        *,
        requires_optimizer: bool,
    ) -> None:
        """Initialize the backend-owned model, prompt encoder, and DMD scheduler."""
        super().initialize(pipeline_config)
        self._gpu_tensor_transfer_enabled = pipeline_config.gpu_tensor_transfer
        seed = int(pipeline_config.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = pipeline_config.full_determinism
        torch.backends.cudnn.benchmark = not pipeline_config.full_determinism
        torch.use_deterministic_algorithms(pipeline_config.full_determinism, warn_only=False)

        self.strategy = create_strategy(worker=self)
        if self.worker_config.strategy_args.strategy_name == FSDP2_TRAIN_STRATEGY:
            # FSDP2 divides max_steps by DP during initialize(). DMD already stores
            # max_steps as local optimizer updates, so provide the pre-DP value it expects.
            data_parallel_size = self.world_size // int(self.worker_config.model_args.ulysses_size or 1)
            self.worker_config.training_args.max_steps *= data_parallel_size
        if current_platform.device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"{current_platform.device_type}:{current_platform.current_device()}")

        role_name = self.role_name or self.worker_config.name
        model_variant = pipeline_config.diffusion_model_variant
        adapter_cls = get_diffusion_model_adapter(model_variant)

        # VeOmni ignores the provider and builds its native model. FSDP2 calls the
        # provider, so a Self-Forcing generator swaps in the model-specific AR builder.
        uses_fsdp2_self_forcing_provider = (
            role_name == "generator"
            and pipeline_config.self_forcing.enabled
            and self.worker_config.strategy_args.strategy_name == FSDP2_TRAIN_STRATEGY
        )
        model_provider = partial(
            default_diffusion_model_provider,
            diffusion_model_variant=model_variant,
        )
        if uses_fsdp2_self_forcing_provider:
            # Keep AR construction beside its adapter. DMDConfig disables the
            # strategy-owned tokenizer/processor because this provider carries
            # its own prompt encoder.
            model_provider = importlib.import_module(get_self_forcing_module(model_variant)).build_model
        self.strategy.initialize(model_provider)
        self.model = self.strategy.model
        # Native backends expose diffusion auxiliaries from the strategy runtime;
        # provider-built diffusion modules carry their own prompt encoder.
        self.prompt_encoder = getattr(self.strategy, "prompt_encoder", None) or getattr(
            self.model, "prompt_encoder", None
        )
        needs_prompt_encoder = role_name == "generator" or not pipeline_config.share_prompt_embeddings
        if needs_prompt_encoder and self.prompt_encoder is None:
            raise RuntimeError(
                f"DMD {role_name} role expects strategy '{self.worker_config.strategy_args.strategy_name}' to expose "
                "a prompt_encoder, or the provider-built model to carry one."
            )
        text_encoder = getattr(self.prompt_encoder, "text_encoder", None) if self.prompt_encoder is not None else None
        if text_encoder is not None and not pipeline_config.is_offload_states:
            text_encoder.to(self.device)
        self.vae_config = None
        if self.prompt_encoder is not None:
            vae = getattr(self.prompt_encoder, "vae", None)
            self.vae_config = vae.config if vae is not None else getattr(self.prompt_encoder, "vae_config", None)
        if role_name == "generator" and self.pipeline_config.self_forcing.enabled and not all(
            callable(getattr(self.model, method, None)) for method in ("create_ar_state", "forward_ar")
        ):
            raise RuntimeError(
                "DMD self_forcing.enabled=True requires a generator model with "
                "create_ar_state(...) and forward_ar(...)."
            )
        self.models = nn.ModuleDict({"model": self.model})
        self.dtype = self.strategy.param_dtype
        self.optimizer = getattr(self.strategy, "optimizer", None)
        if requires_optimizer and self.optimizer is None:
            raise RuntimeError(
                f"DMD {role_name} role requires strategy '{self.worker_config.strategy_args.strategy_name}' "
                "to create an optimizer during initialize(...)."
            )

        num_train_timesteps = int(pipeline_config.score_timestep.num_train_timestep)
        self.scheduler = DMDFlowMatchScheduler(
            num_inference_steps=num_train_timesteps,
            num_train_timesteps=num_train_timesteps,
            shift=float(pipeline_config.timestep_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(num_train_timesteps, device=self.device)
        self.diffusion_adapter = adapter_cls(
            transformer=self.model,
            scheduler=self.scheduler,
        )

    def _backward_and_step(
        self,
        loss: torch.Tensor,
        metrics: dict[str, float],
    ) -> None:
        """Backpropagate one DMD loss and update the strategy-built optimizer."""
        self.optimizer.zero_grad()

        scaler = getattr(self.strategy, "scaler", None)
        with getattr(self.strategy, "model_bwd_context", nullcontext()):
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            scaler.unscale_(self.optimizer)

        grad_norm = self.strategy.clip_grad_norm(float(self.pipeline_config.max_grad_norm))
        metrics[f"{self.worker_config.name}/grad_norm"] = float(grad_norm.detach().float().mean().item())

        if (
            self.pipeline_config.is_offload_states
            or self.pipeline_config.is_offload_optimizer_states_in_train_step
        ):
            self.strategy.load_states(include=(OffloadStateType.optimizer_states,))

        optimizer_stepped = False
        if scaler is not None:
            scale_before_step = scaler.get_scale()
            scaler.step(self.optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before_step
        elif torch.isfinite(grad_norm).all():
            self.optimizer.step()
            optimizer_stepped = True
        else:
            logger.warning(f"WARN: DMD role {self.worker_config.name} grad_norm is not finite: {grad_norm}")

        lr_scheduler = getattr(self.strategy, "scheduler", None)
        if optimizer_stepped and lr_scheduler is not None:
            lr_scheduler.step()
        self.optimizer.zero_grad()
        if self.pipeline_config.is_offload_optimizer_states_in_train_step:
            self.strategy.offload_states(
                include=(OffloadStateType.optimizer_states,),
                non_blocking=True,
            )
        self._last_optimizer_stepped = optimizer_stepped

    @staticmethod
    def _extract_prompts(data: DataProto) -> list[str]:
        prompts = data.non_tensor_batch[DMD_KEY_PROMPTS]
        return prompts.tolist() if isinstance(prompts, np.ndarray) else list(prompts)

    def _encode_prompt_batches(self, prompt_batches: list[list[str]]) -> list[EncodedPrompt]:
        """Encode prompt batches while loading a provider-owned encoder only once."""
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
            return [
                self.diffusion_adapter.encode_prompt(
                    prompt_encoder=self.prompt_encoder,
                    prompt_inputs={
                        "prompts": prompts,
                        "device": self.device,
                        "dtype": self.dtype,
                    },
                )
                for prompts in prompt_batches
            ]

    def _encode_prompt(self, prompts: list[str]) -> EncodedPrompt:
        """Encode prompts into opaque model-family prompt tensors."""
        return self._encode_prompt_batches([prompts])[0]

    def _get_prompt(self, data: DataProto) -> EncodedPrompt:
        """Return generator-shared prompt tensors or encode local prompt strings."""
        if not self.pipeline_config.share_prompt_embeddings:
            return self._encode_prompt(self._extract_prompts(data))
        return {
            key.removeprefix(DMD_PROMPT_TENSOR_PREFIX): (
                value.to(dtype=self.dtype) if value.is_floating_point() else value
            )
            for key, value in data.batch.items()
            if key.startswith(DMD_PROMPT_TENSOR_PREFIX)
        }

    def _get_real_prompt(self, data: DataProto) -> EncodedPrompt:
        """Return generator-shared GAN real prompt tensors or encode local strings."""
        if not self.pipeline_config.share_prompt_embeddings:
            real_prompts = data.non_tensor_batch[DMD_KEY_REAL_PROMPTS]
            prompts = real_prompts.tolist() if isinstance(real_prompts, np.ndarray) else list(real_prompts)
            return self._encode_prompt(prompts)
        real_prompt = {
            key.removeprefix(DMD_REAL_PROMPT_TENSOR_PREFIX): (
                value.to(dtype=self.dtype) if value.is_floating_point() else value
            )
            for key, value in data.batch.items()
            if key.startswith(DMD_REAL_PROMPT_TENSOR_PREFIX)
        }
        if not real_prompt:
            raise KeyError("GAN real prompt tensors are missing from the generator output")
        return real_prompt

    def _dcp_process_group(self) -> dist.ProcessGroup | None:
        """Return the backend process group used by DCP, if the strategy provides one."""
        if not dist.is_available() or not dist.is_initialized():
            return None
        get_process_group = getattr(self.strategy, "get_dcp_process_group", None)
        return get_process_group() if get_process_group is not None else None

    def _save_pretrained_model(self, checkpoint_dir: str) -> None:
        """Gather the DMD model and write a reusable artifact on rank 0."""
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_model_state = get_model_state_dict(self.model, options=options)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return

        had_instance_state_dict = "state_dict" in self.model.__dict__
        original_state_dict = self.model.__dict__.get("state_dict")
        self.model.state_dict = types.MethodType(lambda _model, *args, **kwargs: full_model_state, self.model)
        try:
            self.model.save_pretrained(
                checkpoint_dir,
                state_dict=full_model_state,
                safe_serialization=True,
            )
        finally:
            if had_instance_state_dict:
                self.model.state_dict = original_state_dict
            else:
                delattr(self.model, "state_dict")

    def _save_checkpoint_to_dir(self, checkpoint_dir: str, save_hf_model: bool = False) -> dict[str, float]:
        """Save model, optimizer, EMA, step, and RNG state."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics: dict[str, float] = {}
        dcp_dir = os.path.join(checkpoint_dir, "dcp")
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)

        if self.pipeline_config.is_offload_states:
            with Timer("load_states", logger=None) as timer:
                self.strategy.load_states()
            metrics["load_states"] = timer.last

        try:
            rng_state = {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            }
            if torch.cuda.is_available():
                rng_state["cuda"] = torch.cuda.get_rng_state_all()
            worker_state: dict[str, Any] = {
                "worker_step": int(self.step),
                "rng_state": rng_state,
            }
            lr_scheduler = getattr(self.strategy, "scheduler", None)
            if lr_scheduler is not None:
                worker_state["lr_scheduler"] = lr_scheduler.state_dict()
            scaler = getattr(self.strategy, "scaler", None)
            if scaler is not None:
                worker_state["scaler"] = scaler.state_dict()
            if self.model_ema is not None:
                worker_state["ema"] = self.model_ema.state_dict()
            if self.classification_head is not None:
                worker_state["classification_head"] = self.classification_head.state_dict()
                worker_state["classification_optimizer"] = self.classification_optimizer.state_dict()
                worker_state["classification_scheduler"] = self.classification_scheduler.state_dict()
            model_key = f"model__{self.role_name}"
            optimizer_key = f"optimizer__{self.role_name}"
            # TODO: Add VeOmni EP-aware state-dict handling for checkpoint save.
            state_dict: dict[str, Any] = {
                model_key: get_model_state_dict(self.model, options=options),
                optimizer_key: get_optimizer_state_dict(
                    self.models,
                    self.optimizer,
                    options=options,
                ),
            }
            with Timer("dcp_save", logger=None) as timer:
                dcp.save(
                    state_dict=state_dict,
                    checkpoint_id=dcp_dir,
                    process_group=self._dcp_process_group(),
                )
            metrics["dcp_save"] = timer.last

            if save_hf_model:
                with Timer("hf_save", logger=None) as timer:
                    self._save_pretrained_model(checkpoint_dir)
                metrics["hf_save"] = timer.last

            with Timer("worker_state_save", logger=None) as timer:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                torch.save(worker_state, os.path.join(checkpoint_dir, f"dmd_worker_state_rank_{rank}.pt"))
            metrics["worker_state_save"] = timer.last
        finally:
            if self.pipeline_config.is_offload_states:
                with Timer("offload_states", logger=None) as timer:
                    self.strategy.offload_states()
                metrics["offload_states"] = timer.last

        return metrics

    def _load_checkpoint_from_dir(self, checkpoint_dir: str) -> None:
        """Load a DMD role checkpoint."""
        dcp_dir = os.path.join(checkpoint_dir, "dcp")
        if not os.path.isdir(dcp_dir):
            raise FileNotFoundError(f"missing DMD DCP checkpoint directory: {dcp_dir}")

        model_key = f"model__{self.role_name}"
        optimizer_key = f"optimizer__{self.role_name}"
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        # TODO: Add VeOmni EP-aware state-dict handling for checkpoint load.
        state_dict: dict[str, Any] = {
            model_key: get_model_state_dict(self.model, options=options),
        }
        state_dict[optimizer_key] = get_optimizer_state_dict(
            self.models,
            self.optimizer,
            options=options,
        )
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=dcp_dir,
            process_group=self._dcp_process_group(),
        )
        set_model_state_dict(self.model, state_dict[model_key], options=options)
        set_optimizer_state_dict(self.models, self.optimizer, state_dict[optimizer_key], options=options)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        worker_state_path = os.path.join(checkpoint_dir, f"dmd_worker_state_rank_{rank}.pt")
        if not os.path.exists(worker_state_path):
            raise FileNotFoundError(f"missing DMD worker state at {worker_state_path}")

        worker_state = torch.load(worker_state_path, map_location="cpu", weights_only=False)
        rng_state = worker_state["rng_state"]
        torch.set_rng_state(rng_state["torch"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        self.step = int(worker_state["worker_step"])
        lr_scheduler_state = worker_state.get("lr_scheduler")
        if lr_scheduler_state is not None:
            self.strategy.scheduler.load_state_dict(lr_scheduler_state)
        scaler_state = worker_state.get("scaler")
        if scaler_state is not None:
            self.strategy.scaler.load_state_dict(scaler_state)
        ema_state = worker_state.get("ema")
        if ema_state is not None and self.pipeline_config.ema.enabled:
            self.model_ema = ModelEMA(self.model, decay=float(self.pipeline_config.ema.weight))
            self.model_ema.load_state_dict(ema_state)
        if self.classification_head is not None:
            self.classification_head.load_state_dict(worker_state["classification_head"])
            self.classification_optimizer.load_state_dict(worker_state["classification_optimizer"])
            self.classification_scheduler.load_state_dict(worker_state["classification_scheduler"])
        logger.info(f"loaded DMD checkpoint from {checkpoint_dir}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step: int, is_last_step: bool = False) -> DataProto:
        """Save this role into local staging and upload every rank's checkpoint files."""
        ckpt_id = f"checkpoint-{global_step}"
        local_checkpoint = self.checkpoint_manager.uploader is None
        upload_root = (
            os.path.join(self.pipeline_config.output_dir, ckpt_id)
            if local_checkpoint
            else os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
        )
        save_dir = os.path.join(upload_root, self.role_name)
        logger.info(f"save DMD {self.cluster_name} checkpoint-{global_step} to {save_dir}")

        with Timer("do_checkpoint", logger=None) as total_timer:
            exec_metrics = self._save_checkpoint_to_dir(save_dir, self.role_name == "generator")
            if dist.is_available() and dist.is_initialized():
                dist.barrier(group=self._dcp_process_group())

            if not local_checkpoint:
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
                exec_metrics["upload_submit"] = upload_timer.last

        metrics = {f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last}
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{key}": value for key, value in exec_metrics.items()})
        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def wait_for_checkpoint_upload(self, ckpt_id: str, wait: bool = True) -> bool:
        """Wait for an upload, or report its completion without blocking."""
        upload = self._checkpoint_uploads.get(ckpt_id)
        if upload is None:
            return True
        if not wait and not upload.done():
            return False
        upload.result()
        self._checkpoint_uploads.pop(ckpt_id, None)
        return True


class DMDGeneratorWorker(BaseDMDWorker):
    """Trainable DMD generator worker role."""

    role_name = "generator"

    def _maybe_initialize_ema(self, global_step: int) -> None:
        """Create generator EMA once the configured start step is reached."""
        ema_config = self.pipeline_config.ema
        if not ema_config.enabled or self.model_ema is not None:
            return
        if global_step < int(ema_config.start_step):
            return
        self.model_ema = ModelEMA(self.model, decay=float(ema_config.weight))
        logger.info(f"EMA for generator created at global step {global_step} with decay={float(ema_config.weight)}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config: DMDConfig) -> None:
        """Initialize generator, prompt encoder, optimizer, and generation state."""
        self._initialize_components(
            pipeline_config,
            requires_optimizer=True,
        )
        denoising_config = self.pipeline_config.denoising
        self.denoising_step_list = torch.tensor(denoising_config.step_list, dtype=torch.long)
        if denoising_config.warp_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[self.scheduler.num_train_timesteps - self.denoising_step_list]
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self._pending_generator_step: dict[str, Any] | None = None
        self._shared_negative_prompt: EncodedPrompt = {}
        if self.pipeline_config.share_prompt_embeddings:
            self._shared_negative_prompt = {
                key: value.detach().cpu()
                for key, value in self._encode_prompt([self.pipeline_config.guidance.negative_prompt]).items()
            }
        checkpoint_root = self.pipeline_config.resume_from_checkpoint
        if checkpoint_root:
            self._load_checkpoint_from_dir(os.path.join(checkpoint_root, self.role_name))
        self._maybe_initialize_ema(0)
        if self.pipeline_config.is_offload_states:
            self.strategy.offload_states()
        elif self.pipeline_config.is_offload_optimizer_states_in_train_step:
            self.strategy.offload_states(include=(OffloadStateType.optimizer_states,))
        logger.info(f"{self.worker_name} initialized as DMDGeneratorWorker")

    def _prompt_tensors_for_score_roles(
        self,
        prompt: EncodedPrompt,
        batch_size: int,
        real_prompt: EncodedPrompt | None = None,
    ) -> dict[str, torch.Tensor]:
        """Package positive, negative, and optional GAN real prompt tensors for score workers."""
        if not self.pipeline_config.share_prompt_embeddings:
            return {}
        tensors = {
            f"{DMD_PROMPT_TENSOR_PREFIX}{key}": value.detach().cpu()
            for key, value in prompt.items()
        }
        tensors.update(
            {
                f"{DMD_NEGATIVE_PROMPT_TENSOR_PREFIX}{key}": value.expand(
                    batch_size,
                    *value.shape[1:],
                ).contiguous()
                for key, value in self._shared_negative_prompt.items()
            }
        )
        if real_prompt is not None:
            tensors.update(
                {
                    f"{DMD_REAL_PROMPT_TENSOR_PREFIX}{key}": value.detach().cpu()
                    for key, value in real_prompt.items()
                }
            )
        return tensors

    def _generate_native_sample(
        self,
        noise: torch.Tensor,
        prompt: EncodedPrompt,
        noise_generator: torch.Generator,
        exit_step_generator: torch.Generator,
    ) -> tuple[torch.Tensor, None, int, int, int]:
        """Run the configurable non-causal DMD generator rollout."""
        latent_shape = list(noise.shape)
        denoising_index = sample_denoising_indices(
            1,
            len(self.denoising_step_list),
            noise.device,
            generator=exit_step_generator,
        )[0]
        noisy_latents = noise
        for step_index in range(denoising_index):
            current_timestep = self.denoising_step_list[step_index]
            timestep = torch.full(
                [latent_shape[0], latent_shape[1]],
                current_timestep.item(),
                device=noise.device,
                dtype=self.denoising_step_list.dtype,
            )
            with torch.no_grad():
                prediction = self.diffusion_adapter.forward_step(
                    latents=noisy_latents,
                    prompt=prompt,
                    timestep=timestep,
                )
                next_timestep = self.denoising_step_list[step_index + 1]
                step_noise = torch.randn(
                    prediction.pred_x0.shape,
                    device=prediction.pred_x0.device,
                    dtype=prediction.pred_x0.dtype,
                    generator=noise_generator,
                )
                noisy_latents = self.scheduler.add_noise(
                    prediction.pred_x0.flatten(0, 1),
                    step_noise.flatten(0, 1),
                    torch.full(
                        [latent_shape[0] * latent_shape[1]],
                        next_timestep.item(),
                        device=noise.device,
                        dtype=self.denoising_step_list.dtype,
                    ),
                ).unflatten(0, prediction.pred_x0.shape[:2])

        current_timestep = self.denoising_step_list[denoising_index]
        timestep = torch.full(
            [latent_shape[0], latent_shape[1]],
            current_timestep.item(),
            device=noise.device,
            dtype=self.denoising_step_list.dtype,
        )
        prediction = self.diffusion_adapter.forward_step(
            latents=noisy_latents,
            prompt=prompt,
            timestep=timestep,
        )
        timestep_from, timestep_to = get_score_timestep_window(
            self.scheduler,
            self.denoising_step_list,
            denoising_index,
            noise.device,
        )
        return prediction.pred_x0.to(self.dtype), None, timestep_from, timestep_to, denoising_index

    def _generate_generator_sample(
        self,
        batch_size: int,
        prompt: EncodedPrompt,
        noise_generator: torch.Generator,
        exit_step_generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int | None, int | None, int | None]:
        """Run the generator path chosen by the DMD training config."""
        latent_shape = latent_shape_from_vae_config(
            batch_size=batch_size,
            num_frames=self.pipeline_config.sample.num_frames,
            height=self.pipeline_config.sample.height,
            width=self.pipeline_config.sample.width,
            vae_config=self.vae_config,
        )
        noise = torch.randn(
            latent_shape,
            device=self.device,
            dtype=self.dtype,
            generator=noise_generator,
        )
        if self.pipeline_config.self_forcing.enabled:
            # The model plugin owns AR forward/cache mechanics; DMD still owns the
            # denoising schedule, gradient window, and generated sample semantics.
            return (
                *generate_self_forcing_sample(
                    model=self.model,
                    scheduler=self.scheduler,
                    config=self.pipeline_config.self_forcing,
                    denoising_step_list=self.denoising_step_list,
                    noise=noise,
                    prompt=prompt,
                    noise_generator=noise_generator,
                    exit_step_generator=exit_step_generator,
                ),
                None,
            )
        return self._generate_native_sample(
            noise,
            prompt,
            noise_generator,
            exit_step_generator,
        )

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def begin_generator_step(self, data: DataProto) -> DataProto:
        """Generate with grad and return a detached score query."""
        if self.pipeline_config.is_offload_states:
            self.strategy.load_states(include=_FORWARD_STATE_TYPES)
        data = data.to(self.device)
        prompts = self._extract_prompts(data)
        batch_size = len(prompts)
        self.model.eval()
        with getattr(self.strategy, "model_fwd_context", nullcontext()):
            prompt = self._encode_prompt(prompts)
            generated_latents, gradient_mask, timestep_from, timestep_to, denoising_index = (
                self._generate_generator_sample(
                    batch_size=batch_size,
                    prompt=prompt,
                    noise_generator=self._make_algorithm_generator(data, _RNG_GENERATOR_TRAIN_SAMPLE),
                    exit_step_generator=self._make_algorithm_generator(data, _RNG_GENERATOR_TRAIN_EXIT_STEP),
                )
            )
            _, num_frames = generated_latents.shape[:2]

            with torch.no_grad():
                timestep = sample_dmd_timesteps(
                    self.pipeline_config.score_timestep,
                    self.device,
                    timestep_from,
                    timestep_to,
                    batch_size,
                    num_frames,
                    generator=self._make_algorithm_generator(data, _RNG_GENERATOR_QUERY_TIMESTEP),
                )
                noise = torch.randn(
                    generated_latents.shape,
                    device=generated_latents.device,
                    dtype=generated_latents.dtype,
                    generator=self._make_algorithm_generator(data, _RNG_GENERATOR_QUERY_NOISE),
                )
                noisy_latent = self.scheduler.add_noise(
                    generated_latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                ).detach().unflatten(0, (batch_size, num_frames))
                if self.pipeline_config.ddmd.enabled:
                    ca_index = torch.randint(
                        min(denoising_index + 1, len(self.denoising_step_list) - 1),
                        len(self.denoising_step_list),
                        (1,),
                        device=self.device,
                        generator=self._make_algorithm_generator(data, _RNG_DDMD_CA_TIMESTEP),
                    ).item()
                    timestep_ca = torch.full(
                        (batch_size, num_frames),
                        self.denoising_step_list[ca_index].item(),
                        device=self.device,
                        dtype=self.denoising_step_list.dtype,
                    )
                    noise_ca = torch.randn(
                        generated_latents.shape,
                        device=generated_latents.device,
                        dtype=generated_latents.dtype,
                        generator=self._make_algorithm_generator(data, _RNG_DDMD_CA_NOISE),
                    )
                    noisy_latent_ca = self.scheduler.add_noise(
                        generated_latents.flatten(0, 1),
                        noise_ca.flatten(0, 1),
                        timestep_ca.flatten(0, 1),
                    ).detach().unflatten(0, (batch_size, num_frames))

        self._pending_generator_step = {
            "generated_latents": generated_latents,
            "gradient_mask": gradient_mask,
        }
        tensors = {
            DMD_KEY_TIMESTEP: timestep.detach().cpu(),
            **self._prompt_tensors_for_score_roles(prompt, batch_size),
        }
        if self.pipeline_config.ddmd.enabled:
            tensors[DMD_KEY_TIMESTEP_CA] = timestep_ca.detach().cpu()
        if self.pipeline_config.gan.enabled:
            if self._gpu_tensor_transfer_enabled:
                self._tensor_transfer_slots[DMD_KEY_GENERATED] = generated_latents.detach().contiguous()
            else:
                tensors[DMD_KEY_GENERATED] = generated_latents.detach().cpu()
        if self._gpu_tensor_transfer_enabled:
            self._tensor_transfer_slots[DMD_KEY_NOISY_LATENT] = noisy_latent.detach().contiguous()
            if self.pipeline_config.ddmd.enabled:
                self._tensor_transfer_slots[DMD_KEY_NOISY_LATENT_CA] = noisy_latent_ca.detach().contiguous()
        else:
            tensors[DMD_KEY_NOISY_LATENT] = noisy_latent.detach().cpu()
            if self.pipeline_config.ddmd.enabled:
                tensors[DMD_KEY_NOISY_LATENT_CA] = noisy_latent_ca.detach().cpu()
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors={DMD_KEY_PROMPTS: prompts},
            meta_info={"metrics": {}},
        )

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def finish_generator_step(self, data: DataProto) -> DataProto:
        """Apply generator loss using score tensors returned by score workers."""
        pending_step = self._pending_generator_step
        generated_latents = pending_step["generated_latents"]
        if self._gpu_tensor_transfer_enabled:
            fake_pred_x0 = self._tensor_transfer_slots.pop(DMD_KEY_PRED_FAKE_IMAGE).to(dtype=generated_latents.dtype)
            real_pred_x0 = self._tensor_transfer_slots.pop(DMD_KEY_PRED_REAL_IMAGE).to(dtype=generated_latents.dtype)
        else:
            data = data.to(self.device)
            fake_pred_x0 = data.batch[DMD_KEY_PRED_FAKE_IMAGE].to(self.device, dtype=generated_latents.dtype)
            real_pred_x0 = data.batch[DMD_KEY_PRED_REAL_IMAGE].to(self.device, dtype=generated_latents.dtype)
        if self.pipeline_config.ddmd.enabled:
            if self._gpu_tensor_transfer_enabled:
                real_cond_pred_x0 = self._tensor_transfer_slots.pop(DMD_KEY_PRED_REAL_COND_CA).to(
                    dtype=generated_latents.dtype
                )
                real_uncond_pred_x0 = self._tensor_transfer_slots.pop(DMD_KEY_PRED_REAL_UNCOND_CA).to(
                    dtype=generated_latents.dtype
                )
            else:
                real_cond_pred_x0 = data.batch[DMD_KEY_PRED_REAL_COND_CA].to(
                    device=self.device,
                    dtype=generated_latents.dtype,
                )
                real_uncond_pred_x0 = data.batch[DMD_KEY_PRED_REAL_UNCOND_CA].to(
                    device=self.device,
                    dtype=generated_latents.dtype,
                )
            guidance_config = self.pipeline_config.guidance
            guidance_scale = (
                guidance_config.real_scale if guidance_config.real_scale is not None else guidance_config.scale
            )
            dmd_loss, gradient_dm_mean_abs, gradient_ca_mean_abs = compute_ddmd_generator_loss(
                generated_latents=generated_latents,
                fake_pred_x0=fake_pred_x0,
                real_pred_x0=real_pred_x0,
                real_cond_pred_x0=real_cond_pred_x0,
                real_uncond_pred_x0=real_uncond_pred_x0,
                guidance_scale=float(guidance_scale),
                gradient_scale=float(self.pipeline_config.ddmd.gradient_scale),
                gradient_mask=pending_step["gradient_mask"],
            )
            metrics = {
                "generator/loss": float(dmd_loss.detach().item()),
                "generator/ddmd_gradient_dm_mean_abs": float(gradient_dm_mean_abs.detach().item()),
                "generator/ddmd_gradient_ca_mean_abs": float(gradient_ca_mean_abs.detach().item()),
            }
        else:
            dmd_loss, gradient_mean_abs = compute_dmd_generator_loss(
                generated_latents=generated_latents,
                fake_pred_x0=fake_pred_x0,
                real_pred_x0=real_pred_x0,
                gradient_mask=pending_step["gradient_mask"],
            )
            metrics = {
                "generator/loss": float(dmd_loss.detach().item()),
                "generator/dmd_gradient_mean_abs": float(gradient_mean_abs.detach().item()),
            }
        loss = dmd_loss
        if self.pipeline_config.gan.enabled:
            if self._gpu_tensor_transfer_enabled:
                gan_input_gradient = self._tensor_transfer_slots.pop(DMD_KEY_GAN_INPUT_GRADIENT)
            else:
                gan_input_gradient = data.batch[DMD_KEY_GAN_INPUT_GRADIENT].to(self.device)
            gan_surrogate, gan_gradient_mean_abs = compute_dmd_gan_generator_surrogate(
                generated_latents=generated_latents,
                input_gradient=gan_input_gradient,
                gradient_mask=pending_step["gradient_mask"],
            )
            loss = loss + float(self.pipeline_config.gan.generator_loss_weight) * gan_surrogate
            metrics["generator/gan_gradient_mean_abs"] = float(gan_gradient_mean_abs.detach().item())
        self._maybe_initialize_ema(int(data.meta_info["global_step"]))
        self._backward_and_step(loss, metrics)
        self._pending_generator_step = None
        ema_config = self.pipeline_config.ema
        next_step = self.step + 1
        if (
            ema_config.enabled
            and self.model_ema is not None
            and self._last_optimizer_stepped
            and next_step % int(ema_config.update_interval) == 0
        ):
            self.model_ema.update(self.model)
        self.step += 1
        if self.pipeline_config.is_offload_states:
            self.strategy.offload_states()
        return DataProto.from_dict(
            tensors={"step": torch.tensor([self.step], dtype=torch.long)},
            meta_info={"metrics": metrics},
        )

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def generate_no_grad(self, data: DataProto) -> DataProto:
        """Generate detached samples used by DMDFakeScoreWorker.train_score_step."""
        if self.pipeline_config.is_offload_states:
            self.strategy.load_states(include=_FORWARD_STATE_TYPES)
        data = data.to(self.device)
        prompts = self._extract_prompts(data)
        real_prompts: list[str] | None = None
        if self.pipeline_config.gan.enabled and self.pipeline_config.share_prompt_embeddings:
            if DMD_KEY_REAL_PROMPTS not in data.non_tensor_batch:
                raise KeyError("GAN real prompts are required when share_prompt_embeddings=True")
            raw_real_prompts = data.non_tensor_batch[DMD_KEY_REAL_PROMPTS]
            real_prompts = (
                raw_real_prompts.tolist() if isinstance(raw_real_prompts, np.ndarray) else list(raw_real_prompts)
            )
        self.model.eval()
        with torch.no_grad(), getattr(self.strategy, "model_fwd_context", nullcontext()):
            prompt_batches = [prompts]
            if real_prompts is not None:
                prompt_batches.append(real_prompts)
            encoded_prompts = self._encode_prompt_batches(prompt_batches)
            prompt = encoded_prompts[0]
            real_prompt = encoded_prompts[1] if real_prompts is not None else None
            generated_latents, _, timestep_from, timestep_to, _ = self._generate_generator_sample(
                batch_size=len(prompts),
                prompt=prompt,
                noise_generator=self._make_algorithm_generator(data, _RNG_FAKE_SAMPLE),
                exit_step_generator=self._make_algorithm_generator(data, _RNG_FAKE_SAMPLE_EXIT_STEP),
            )
        if self.pipeline_config.is_offload_states:
            self.strategy.offload_states(include=_FORWARD_STATE_TYPES)
        batch_size = generated_latents.shape[0]
        tensors = {
            DMD_KEY_DENOISED_TIMESTEP_FROM: torch.full(
                (batch_size,),
                -1 if timestep_from is None else int(timestep_from),
                dtype=torch.long,
                device=generated_latents.device,
            ).cpu(),
            DMD_KEY_DENOISED_TIMESTEP_TO: torch.full(
                (batch_size,),
                -1 if timestep_to is None else int(timestep_to),
                dtype=torch.long,
                device=generated_latents.device,
            ).cpu(),
            **self._prompt_tensors_for_score_roles(prompt, batch_size, real_prompt=real_prompt),
        }
        if self._gpu_tensor_transfer_enabled:
            self._tensor_transfer_slots[DMD_KEY_GENERATED] = generated_latents.detach().contiguous()
        else:
            tensors[DMD_KEY_GENERATED] = generated_latents.detach().cpu()
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors={DMD_KEY_PROMPTS: prompts},
            meta_info={"metrics": {}},
        )


class DMDRealScoreWorker(BaseDMDWorker):
    """Frozen DMD teacher score worker role."""

    role_name = "real_score"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config: DMDConfig) -> None:
        """Initialize frozen real-score model and prompt encoder."""
        self._initialize_components(
            pipeline_config,
            requires_optimizer=False,
        )
        guidance_config = pipeline_config.guidance
        self.score_guidance_scale = float(
            guidance_config.real_scale if guidance_config.real_scale is not None else guidance_config.scale
        )
        self._cached_negative_prompt: EncodedPrompt | None = None
        if self.pipeline_config.is_offload_states:
            self.strategy.offload_states()
        logger.info(f"{self.worker_name} initialized as DMDRealScoreWorker")

    def _negative_prompt_for_batch(self, data: DataProto, batch_size: int) -> EncodedPrompt:
        """Return the cached negative prompt expanded to one score batch."""
        if self.pipeline_config.share_prompt_embeddings:
            return {
                key.removeprefix(DMD_NEGATIVE_PROMPT_TENSOR_PREFIX): (
                    value.to(dtype=self.dtype) if value.is_floating_point() else value
                )
                for key, value in data.batch.items()
                if key.startswith(DMD_NEGATIVE_PROMPT_TENSOR_PREFIX)
            }
        if self._cached_negative_prompt is None:
            negative_prompt = self.pipeline_config.guidance.negative_prompt
            self._cached_negative_prompt = {
                key: value.detach().cpu()
                for key, value in self._encode_prompt([negative_prompt]).items()
            }
        return {
            key: value.to(self.device).expand(batch_size, *value.shape[1:])
            for key, value in self._cached_negative_prompt.items()
        }

    def _score_pred_x0(self, data: DataProto, guidance_scale: float) -> torch.Tensor:
        """Run score model and return guided x0 prediction."""
        prompt = self._get_prompt(data)
        if self._gpu_tensor_transfer_enabled:
            noisy_latent = self._tensor_transfer_slots.pop(DMD_KEY_NOISY_LATENT).to(dtype=self.dtype)
        else:
            noisy_latent = data.batch[DMD_KEY_NOISY_LATENT].to(device=self.device, dtype=self.dtype)
        timestep = data.batch[DMD_KEY_TIMESTEP].to(self.device)

        if guidance_scale == 0.0:
            return self.diffusion_adapter.forward_step(
                latents=noisy_latent,
                prompt=prompt,
                timestep=timestep,
            ).pred_x0

        negative_prompt = self._negative_prompt_for_batch(data, noisy_latent.shape[0])
        return self.diffusion_adapter.forward_step(
            latents=noisy_latent,
            prompt=prompt,
            timestep=timestep,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
        ).pred_x0

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def forward_score(self, data: DataProto) -> DataProto:
        """Return the real-score predictions required by DMD or DDMD."""
        metrics: dict[str, float] = {}
        metric_role = self.cluster_name
        offload_context = (
            state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{metric_role}/forward_score",
                is_offload_states=True,
                load_kwargs={"include": _FORWARD_STATE_TYPES},
            )
            if self.pipeline_config.is_offload_states
            else nullcontext()
        )
        with Timer(f"{metric_role}_forward", logger=None) as timer:
            with offload_context:
                with torch.no_grad():
                    data = data.to(self.device)
                    self.model.eval()
                    with getattr(self.strategy, "model_fwd_context", nullcontext()):
                        if self.pipeline_config.ddmd.enabled and self.role_name == "real_score":
                            prompt = self._get_prompt(data)
                            if self._gpu_tensor_transfer_enabled:
                                noisy_latent = self._tensor_transfer_slots.pop(DMD_KEY_NOISY_LATENT).to(
                                    dtype=self.dtype
                                )
                                noisy_latent_ca = self._tensor_transfer_slots.pop(DMD_KEY_NOISY_LATENT_CA).to(
                                    dtype=self.dtype
                                )
                            else:
                                noisy_latent = data.batch[DMD_KEY_NOISY_LATENT].to(
                                    device=self.device,
                                    dtype=self.dtype,
                                )
                                noisy_latent_ca = data.batch[DMD_KEY_NOISY_LATENT_CA].to(
                                    device=self.device,
                                    dtype=self.dtype,
                                )
                            timestep = data.batch[DMD_KEY_TIMESTEP].to(self.device)
                            timestep_ca = data.batch[DMD_KEY_TIMESTEP_CA].to(self.device)
                            negative_prompt = self._negative_prompt_for_batch(data, noisy_latent.shape[0])
                            predictions = {
                                DMD_KEY_PRED_X0: self.diffusion_adapter.forward_step(
                                    latents=noisy_latent,
                                    prompt=prompt,
                                    timestep=timestep,
                                ).pred_x0.detach(),
                                DMD_KEY_PRED_REAL_COND_CA: self.diffusion_adapter.forward_step(
                                    latents=noisy_latent_ca,
                                    prompt=prompt,
                                    timestep=timestep_ca,
                                ).pred_x0.detach(),
                                DMD_KEY_PRED_REAL_UNCOND_CA: self.diffusion_adapter.forward_step(
                                    latents=noisy_latent_ca,
                                    prompt=negative_prompt,
                                    timestep=timestep_ca,
                                ).pred_x0.detach(),
                            }
                        else:
                            predictions = {
                                DMD_KEY_PRED_X0: self._score_pred_x0(
                                    data,
                                    guidance_scale=self.score_guidance_scale,
                                ).detach()
                            }
        metrics[f"{metric_role}/forward_time"] = timer.last
        if self._gpu_tensor_transfer_enabled:
            self._tensor_transfer_slots.update(
                {key: prediction.contiguous() for key, prediction in predictions.items()}
            )
            return DataProto(meta_info={"metrics": metrics})
        return DataProto.from_dict(
            tensors={key: prediction.cpu() for key, prediction in predictions.items()},
            meta_info={"metrics": metrics},
        )


class DMDFakeScoreWorker(DMDRealScoreWorker):
    """Trainable DMD fake-score role with an optional DMD2 classification head."""

    role_name = "fake_score"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config: DMDConfig) -> None:
        """Initialize trainable fake-score model, prompt encoder, and optimizer."""
        self._initialize_components(
            pipeline_config,
            requires_optimizer=True,
        )
        guidance_config = pipeline_config.guidance
        self.score_guidance_scale = float(guidance_config.fake_scale or 0.0)
        self._cached_negative_prompt: EncodedPrompt | None = None
        if pipeline_config.gan.enabled:
            if not callable(getattr(self.diffusion_adapter, "extract_features", None)):
                raise ValueError(
                    f"DMD GAN requires feature extraction, but {type(self.diffusion_adapter).__name__} "
                    "does not provide it"
                )
            self.classification_head = DMDResidualConv3DHead(
                feature_dim=self.diffusion_adapter.feature_dim(),
                head_config=pipeline_config.gan.head,
            ).to(self.device)
            if dist.is_available() and dist.is_initialized():
                for parameter in self.classification_head.parameters():
                    dist.broadcast(parameter.data, src=0)
            training_args = self.worker_config.training_args
            self.classification_optimizer = torch.optim.AdamW(
                self.classification_head.parameters(),
                lr=training_args.learning_rate,
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                weight_decay=training_args.weight_decay,
            )
            max_steps = int(training_args.max_steps)
            self.classification_scheduler = get_scheduler(
                name=training_args.lr_scheduler_type,
                optimizer=self.classification_optimizer,
                num_warmup_steps=training_args.get_warmup_steps(max_steps),
                num_training_steps=max_steps,
            )
        checkpoint_root = self.pipeline_config.resume_from_checkpoint
        if checkpoint_root:
            self._load_checkpoint_from_dir(os.path.join(checkpoint_root, self.role_name))
        if self.pipeline_config.is_offload_states:
            self.strategy.offload_states()
        elif self.pipeline_config.is_offload_optimizer_states_in_train_step:
            self.strategy.offload_states(include=(OffloadStateType.optimizer_states,))
        logger.info(f"{self.worker_name} initialized as DMDFakeScoreWorker")

    def _prepare_classification_latents(
        self,
        data: DataProto,
        latents: torch.Tensor,
        timestep_rng_stream: int,
        noise_rng_stream: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare clean or diffusion-noised inputs for the DMD2 classification head."""
        batch_size, num_frames = latents.shape[:2]
        if not self.pipeline_config.gan.noisy_input:
            return latents, torch.zeros((batch_size, num_frames), dtype=torch.long, device=self.device)

        timestep = torch.randint(
            0,
            int(self.pipeline_config.gan.max_timestep),
            (batch_size, 1),
            device=self.device,
            generator=self._make_algorithm_generator(data, timestep_rng_stream),
        ).expand(batch_size, num_frames)
        timestep = shift_dmd_timesteps(
            timestep,
            int(self.pipeline_config.score_timestep.num_train_timestep),
            float(self.pipeline_config.timestep_shift),
        )
        noise = torch.randn(
            latents.shape,
            device=self.device,
            dtype=latents.dtype,
            generator=self._make_algorithm_generator(data, noise_rng_stream),
        )
        noisy_latents = self.scheduler.add_noise(
            latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, latents.shape[:2])
        return noisy_latents, timestep

    def _classification_logits(
        self,
        latents: torch.Tensor,
        prompt: EncodedPrompt,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Classify latents through the conditioned fake-score backbone and its head."""
        extract_features = getattr(self.diffusion_adapter, "extract_features")
        features = extract_features(
            latents=latents,
            prompt=prompt,
            timestep=timestep,
        )
        return self.classification_head(features)

    def _backward_classification_head(
        self,
        loss: torch.Tensor,
        metrics: dict[str, float],
    ) -> None:
        """Update the replicated classification head with frozen fake-score features."""
        self.classification_optimizer.zero_grad()
        scaler = getattr(self.strategy, "scaler", None)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for parameter in self.classification_head.parameters():
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world_size)
        if scaler is not None:
            scaler.unscale_(self.classification_optimizer)

        head_grad_norm = clip_grad_norm_(
            self.classification_head.parameters(),
            float(self.pipeline_config.max_grad_norm),
        )
        gradients_finite = bool(torch.isfinite(head_grad_norm).item())
        metrics[f"{self.worker_config.name}/classification_grad_norm"] = float(head_grad_norm.detach().item())

        optimizer_stepped = False
        if scaler is not None:
            scale_before_step = scaler.get_scale()
            if gradients_finite:
                scaler.step(self.classification_optimizer)
            scaler.update()
            optimizer_stepped = gradients_finite and scaler.get_scale() >= scale_before_step
        elif gradients_finite:
            self.classification_optimizer.step()
            optimizer_stepped = True
        else:
            logger.warning(f"WARN: DMD classification head grad_norm is not finite: {head_grad_norm}")

        if optimizer_stepped:
            self.classification_scheduler.step()
        self.classification_optimizer.zero_grad()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def forward_score(self, data: DataProto) -> DataProto:
        """Return fake-score x0 and, when enabled, DMD2 generator classification feedback."""
        if not self.pipeline_config.gan.enabled:
            return super().forward_score(data)

        metrics: dict[str, float] = {}
        offload_context = (
            state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/forward_score",
                is_offload_states=True,
                load_kwargs={"include": _FORWARD_STATE_TYPES},
            )
            if self.pipeline_config.is_offload_states
            else nullcontext()
        )
        with Timer(f"{self.cluster_name}_forward", logger=None) as timer:
            with offload_context:
                data = data.to(self.device)
                prompt = self._get_prompt(data)
                if self._gpu_tensor_transfer_enabled:
                    noisy_latent = self._tensor_transfer_slots.pop(DMD_KEY_NOISY_LATENT).to(dtype=self.dtype)
                    generated_latents = self._tensor_transfer_slots.pop(DMD_KEY_GENERATED).to(dtype=self.dtype)
                else:
                    noisy_latent = data.batch[DMD_KEY_NOISY_LATENT].to(device=self.device, dtype=self.dtype)
                    generated_latents = data.batch[DMD_KEY_GENERATED].to(device=self.device, dtype=self.dtype)
                score_timestep = data.batch[DMD_KEY_TIMESTEP].to(self.device)

                self.model.eval()
                self.classification_head.eval()
                with getattr(self.strategy, "model_fwd_context", nullcontext()):
                    with torch.no_grad():
                        fake_pred_x0 = self.diffusion_adapter.forward_step(
                            latents=noisy_latent,
                            prompt=prompt,
                            timestep=score_timestep,
                        ).pred_x0.detach()
                    generated_latents = generated_latents.detach().requires_grad_(True)
                    classifier_latents, classifier_timestep = self._prepare_classification_latents(
                        data,
                        generated_latents,
                        _RNG_GAN_GENERATOR_TIMESTEP,
                        _RNG_GAN_GENERATOR_NOISE,
                    )
                    fake_logits = self._classification_logits(classifier_latents, prompt, classifier_timestep)
                    gan_loss = compute_dmd_gan_generator_loss(fake_logits)
                    gan_input_gradient = torch.autograd.grad(
                        gan_loss * generated_latents.shape[0],
                        generated_latents,
                    )[0].detach()
        metrics[f"{self.cluster_name}/forward_time"] = timer.last
        metrics["generator/gan_loss"] = float(gan_loss.detach().item())
        if self._gpu_tensor_transfer_enabled:
            self._tensor_transfer_slots[DMD_KEY_PRED_X0] = fake_pred_x0.contiguous()
            self._tensor_transfer_slots[DMD_KEY_GAN_INPUT_GRADIENT] = gan_input_gradient.contiguous()
            return DataProto(meta_info={"metrics": metrics})
        return DataProto.from_dict(
            tensors={
                DMD_KEY_PRED_X0: fake_pred_x0.cpu(),
                DMD_KEY_GAN_INPUT_GRADIENT: gan_input_gradient.cpu(),
            },
            meta_info={"metrics": metrics},
        )

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def train_score_step(self, data: DataProto) -> DataProto:
        """Update fake-score diffusion and optional classification objectives separately."""
        metrics: dict[str, float] = {}
        metric_role = self.cluster_name
        offload_context = (
            state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{metric_role}/train_score_step",
                is_offload_states=True,
                load_kwargs=(
                    {"include": _FORWARD_STATE_TYPES}
                    if self.pipeline_config.is_offload_optimizer_states_in_train_step
                    else {}
                ),
            )
            if self.pipeline_config.is_offload_states
            else nullcontext()
        )
        with offload_context:
            data = data.to(self.device)
            self.model.train()
            with getattr(self.strategy, "model_fwd_context", nullcontext()):
                prompt = self._get_prompt(data)
                if self._gpu_tensor_transfer_enabled:
                    generated_latents = self._tensor_transfer_slots.pop(DMD_KEY_GENERATED).to(dtype=self.dtype)
                else:
                    generated_latents = data.batch[DMD_KEY_GENERATED].to(device=self.device, dtype=self.dtype)
                batch_size, num_frames = generated_latents.shape[:2]
                timestep_from = data.batch[DMD_KEY_DENOISED_TIMESTEP_FROM].flatten().to(self.device)
                timestep_to = data.batch[DMD_KEY_DENOISED_TIMESTEP_TO].flatten().to(self.device)
                critic_timestep = sample_dmd_timesteps(
                    self.pipeline_config.score_timestep,
                    self.device,
                    None if torch.all(timestep_from < 0) else timestep_from,
                    None if torch.all(timestep_to < 0) else timestep_to,
                    batch_size,
                    num_frames,
                    generator=self._make_algorithm_generator(data, _RNG_FAKE_CRITIC_TIMESTEP),
                )
                critic_noise = torch.randn(
                    generated_latents.shape,
                    device=generated_latents.device,
                    dtype=generated_latents.dtype,
                    generator=self._make_algorithm_generator(data, _RNG_FAKE_CRITIC_NOISE),
                )
                noisy_latents = self.scheduler.add_noise(
                    generated_latents.flatten(0, 1),
                    critic_noise.flatten(0, 1),
                    critic_timestep.flatten(0, 1),
                ).unflatten(0, generated_latents.shape[:2])
                fake_prediction = self.diffusion_adapter.forward_step(
                    latents=noisy_latents,
                    prompt=prompt,
                    timestep=critic_timestep,
                )
                fake_score_loss = compute_dmd_fake_score_loss(
                    flow_pred=fake_prediction.flow_pred,
                    clean_latents=generated_latents,
                    noise=critic_noise,
                )
                metrics["fake_score/loss"] = float(fake_score_loss.detach().item())
            self._backward_and_step(fake_score_loss, metrics)

            if self.classification_head is not None:
                self.model.eval()
                self.classification_head.train()
                with getattr(self.strategy, "model_fwd_context", nullcontext()):
                    real_latents = data.batch[DMD_KEY_REAL_LATENT].to(device=self.device, dtype=self.dtype)
                    real_prompt = self._get_real_prompt(data)
                    classification_prompt = {
                        key: torch.cat((prompt[key], real_prompt[key]), dim=0)
                        for key in prompt
                    }
                    classification_latents, classification_timestep = self._prepare_classification_latents(
                        data,
                        torch.cat((generated_latents.detach(), real_latents), dim=0),
                        _RNG_GAN_CLASSIFICATION_TIMESTEP,
                        _RNG_GAN_CLASSIFICATION_NOISE,
                    )
                    with torch.no_grad():
                        features = self.diffusion_adapter.extract_features(
                            latents=classification_latents,
                            prompt=classification_prompt,
                            timestep=classification_timestep,
                        )
                    fake_logits, real_logits = self.classification_head(features).chunk(2, dim=0)
                    classification_loss = compute_dmd_gan_classification_loss(real_logits, fake_logits)
                    metrics.update(
                        {
                            "fake_score/classification_loss": float(classification_loss.detach().item()),
                            "fake_score/real_probability": float(real_logits.detach().float().sigmoid().mean().item()),
                            "fake_score/fake_probability": float(fake_logits.detach().float().sigmoid().mean().item()),
                        }
                    )
                self._backward_classification_head(
                    float(self.pipeline_config.gan.classification_loss_weight) * classification_loss,
                    metrics,
                )
        self.step += 1
        return DataProto.from_dict(
            tensors={"step": torch.tensor([self.step], dtype=torch.long)},
            meta_info={"metrics": metrics},
        )
