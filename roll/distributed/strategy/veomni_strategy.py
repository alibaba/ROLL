"""VeOmni-native diffusion strategies for ROLL."""

from __future__ import annotations

import math
import os
from collections.abc import Collection
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

try:
    from veomni.distributed.offloading import (
        load_model_to_gpu,
        load_optimizer,
        offload_model_to_cpu,
        offload_optimizer,
    )

    VEOMNI_AVAILABLE = True
except ImportError:
    VEOMNI_AVAILABLE = False

from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.fsdp2_strategy import FSDP2InferStrategy, FSDP2TrainStrategy
from roll.platforms import current_platform
from roll.utils.checkpoint_manager import download_model
from roll.utils.functionals import append_to_dict, parse_dtype
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

logger = get_logger()

_DIFFUSION_OPS_DEFAULTS = {
    "moe_implementation": "eager",
    "cross_entropy_loss_implementation": "eager",
    "rms_norm_implementation": "eager",
    "swiglu_mlp_implementation": "eager",
    "rotary_pos_emb_implementation": "eager",
    "load_balancing_loss_implementation": "eager",
    "rms_norm_gated_implementation": "eager",
    "causal_conv1d_implementation": "eager",
    "chunk_gated_delta_rule_implementation": "eager",
}


class DiffusionStepStats(NamedTuple):
    """Scheduler-derived statistics for transitions along a diffusion trajectory."""

    log_probs: torch.Tensor
    prev_sample_mean: torch.Tensor
    std_dev_t: torch.Tensor


def _initialize_veomni_parallel_state(strategy_config: dict[str, Any]) -> SimpleNamespace:
    from veomni.distributed import parallel_state

    if not dist.is_initialized():
        if current_platform.device_type != "cpu":
            backend = f"cpu:gloo,{current_platform.device_type}:{current_platform.communication_backend}"
        else:
            backend = current_platform.communication_backend
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    fsdp_size = int(strategy_config.get("fsdp_size", -1))
    expert_parallel_size = int(strategy_config.get("expert_parallel_size", 1))
    tp_size = int(strategy_config.get("tp_size", 1))
    pp_size = int(strategy_config.get("pp_size", 1))
    cp_size = int(strategy_config.get("cp_size", 1))
    ulysses_parallel_size = int(strategy_config.get("ulysses_parallel_size", 1))

    if tp_size != 1:
        raise NotImplementedError("VeOmni strategy does not support tensor parallelism yet")
    if pp_size != 1:
        raise NotImplementedError("VeOmni strategy does not support pipeline parallelism yet")
    if cp_size != 1:
        raise NotImplementedError("VeOmni strategy does not support ring-attention context parallelism yet")
    if expert_parallel_size < 1 or ulysses_parallel_size < 1:
        raise ValueError("expert_parallel_size and ulysses_parallel_size must be positive")

    if world_size % (tp_size * pp_size * cp_size * ulysses_parallel_size) != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by "
            f"tp_size * pp_size * cp_size * ulysses_parallel_size "
            f"({tp_size} * {pp_size} * {cp_size} * {ulysses_parallel_size})"
        )
    if world_size % expert_parallel_size != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by expert_parallel_size ({expert_parallel_size})"
        )

    dp_size = world_size // (tp_size * pp_size * cp_size * ulysses_parallel_size)
    if fsdp_size == -1 or fsdp_size == dp_size:
        dp_replicate_size = 1
        dp_shard_size = dp_size
    else:
        if fsdp_size < 1 or fsdp_size > dp_size:
            raise ValueError(f"fsdp_size must be -1 or between 1 and dp_size ({dp_size}), got {fsdp_size}")
        if dp_size % fsdp_size != 0:
            raise ValueError(
                f"Data parallel size ({dp_size}) must be divisible by fsdp_size ({fsdp_size}). "
                "Please adjust your configuration."
            )
        dp_replicate_size = dp_size // fsdp_size
        dp_shard_size = fsdp_size

    dp_mode = "fsdp2" if world_size // (pp_size * tp_size) > 1 else "ddp"
    init_kwargs = {
        "dp_size": dp_size,
        "dp_replicate_size": dp_replicate_size,
        "dp_shard_size": dp_shard_size,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "cp_size": cp_size,
        "ulysses_size": ulysses_parallel_size,
        "dp_mode": dp_mode,
        "device_type": current_platform.device_type,
        "include_sp_in_fsdp": bool(strategy_config.get("include_sp_in_fsdp", True)),
        "async_enabled": bool(strategy_config.get("enable_async_ulysses", False)),
        "extra_parallel_sizes": (expert_parallel_size,),
        "extra_parallel_placement_innermost": (
            bool(strategy_config.get("extra_parallel_placement_innermost", False)),
        ),
        "extra_parallel_names": ("ep",),
    }
    parallel_state.init_parallel_state(**init_kwargs)

    ps = parallel_state.get_parallel_state()
    return SimpleNamespace(
        world_size=world_size,
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        expert_parallel_size=expert_parallel_size,
        ulysses_size=int(ps.ulysses_size),
        ulysses_rank=int(ps.ulysses_rank) if bool(ps.sp_enabled) else 0,
        ulysses_group=ps.ulysses_group if bool(ps.sp_enabled) else None,
        sp_enabled=bool(ps.sp_enabled),
        async_ulysses_enabled=bool(ps.async_enabled),
        dp_mode=ps.dp_mode,
        dp_rank=int(ps.dp_rank),
    )


def _apply_veomni_parallel_state(strategy: Any) -> None:
    if strategy.parallel_state_initialized:
        raise RuntimeError("VeOmni parallel state has already been initialized for this strategy")

    state = _initialize_veomni_parallel_state(strategy.worker_config.strategy_args.strategy_config)
    logger.info(
        f"Initializing VeOmni parallel state: "
        f"world_size={state.world_size}, dp_size={state.dp_size}, "
        f"dp_replicate_size={state.dp_replicate_size}, "
        f"dp_shard_size={state.dp_shard_size}, "
        f"ep_size={state.expert_parallel_size}, ulysses_size={state.ulysses_size}, "
        f"async_ulysses={state.async_ulysses_enabled}"
    )
    strategy.dp_size = state.dp_size
    strategy.ep_enabled = state.expert_parallel_size > 1
    strategy.dp_mode = state.dp_mode
    strategy.parallel_state_initialized = True

    from roll.utils.context_parallel.globals import get_upg_manager

    upg_manager = get_upg_manager()
    if state.sp_enabled:
        upg_manager.ulysses_group = state.ulysses_group
        upg_manager.ulysses_size = state.ulysses_size
        logger.info(f"Ulysses Sequence Parallelism enabled: size={state.ulysses_size}, rank={state.ulysses_rank}")
    else:
        upg_manager.ulysses_group = None
        upg_manager.ulysses_size = 1

    strategy.worker.rank_info.dp_rank = state.dp_rank
    strategy.worker.rank_info.dp_size = state.dp_size
    strategy.worker.rank_info.cp_size = state.ulysses_size
    strategy.worker.rank_info.cp_rank = state.ulysses_rank if state.sp_enabled else 0
    logger.info("VeOmni parallel state initialized successfully")


class VeOmniInferStrategy(FSDP2InferStrategy):
    """VeOmni inference strategy using native VeOmni builders."""

    strategy_name = "veomni_infer"
    is_trainable = False

    def __init__(self, worker: Worker) -> None:
        if not VEOMNI_AVAILABLE:
            raise ImportError(
                "VeOmni is not available. Please install VeOmni to use VeOmniStrategy. "
                "See: https://github.com/ByteDance-Seed/VeOmni"
            )

        super().__init__(worker)

        self.parallel_state_initialized = False
        self.model_fwd_context = nullcontext()
        self.model_bwd_context = nullcontext()
        self.prompt_encoder = None
        self.adapter = None
        self.optimizer = None
        self.scheduler = None
        self.diffusion_scheduler = None

        self.param_dtype = torch.bfloat16
        self.reduce_dtype = torch.float32
        self.scaler = None
        self.dp_size = 1
        self.ep_enabled = False
        self.dp_mode = "ddp"

    def get_dcp_process_group(self) -> Optional[dist.ProcessGroup]:
        """Return the process group used by distributed checkpointing."""
        return self._get_dcp_process_group()

    def _initialize_diffusion_scheduler(self) -> None:
        """Load the model's reverse-transition scheduler for diffusion rollout training."""
        from roll.pipeline.diffusion.models.scheduling_flow_match_sde_discrete import (
            FlowMatchSDEDiscreteScheduler,
        )

        model_path = download_model(self.worker_config.model_args.model_name_or_path)
        self.diffusion_scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=os.path.exists(model_path),
        )

    def _configure_diffusion_generation_params(self) -> None:
        """Configure transition sampling from the diffusion actor's generation arguments."""
        generation_config = self.worker.pipeline_config.actor_infer.generating_args.to_dict()
        extra_args = generation_config["extra_args"]
        self._noise_level = float(extra_args["noise_level"])
        self._sde_type = extra_args["sde_type"]
        self._guidance_scale = float(generation_config["guidance_scale"])
        self.diffusion_scheduler.set_timesteps(
            int(generation_config["num_inference_steps"]),
            device=current_platform.device_type,
        )

    def compute_diffusion_step_stats(
        self,
        *,
        model_output: torch.Tensor,
        all_latents: torch.Tensor,
        all_timesteps: torch.Tensor,
    ) -> DiffusionStepStats:
        """Evaluate transition statistics for model predictions along an existing trajectory."""
        current_latents = all_latents[:, :-1]
        next_latents = all_latents[:, 1:]

        if model_output.shape != current_latents.shape:
            raise ValueError(
                f"Diffusion replay output shape {tuple(model_output.shape)} does not match "
                f"trajectory shape {tuple(current_latents.shape)}"
            )

        log_probs = []
        prev_sample_means = []
        std_dev_ts = []
        for step in range(all_timesteps.shape[1]):
            _, step_log_prob, step_prev_sample_mean, step_std_dev_t = self.diffusion_scheduler.sample_previous_step(
                sample=current_latents[:, step].float(),
                model_output=model_output[:, step],
                timestep=all_timesteps[:, step],
                noise_level=self._noise_level,
                prev_sample=next_latents[:, step].float(),
                sde_type=self._sde_type,
                logprobs=True,
            )
            log_probs.append(step_log_prob)
            prev_sample_means.append(step_prev_sample_mean)
            std_dev_ts.append(step_std_dev_t)
        return DiffusionStepStats(
            log_probs=torch.stack(log_probs, dim=1),
            prev_sample_mean=torch.stack(prev_sample_means, dim=1),
            std_dev_t=torch.stack(std_dev_ts, dim=1),
        )

    def _prepare_diffusion_forward_context(self, data: DataProto) -> dict[str, Any]:
        """Collect prompt conditioning shared by trajectory replay and training callbacks."""
        negative_prompt = None
        if "negative_prompt_embeds" in data.batch:
            negative_prompt = {"prompt_embeds": data.batch["negative_prompt_embeds"]}
        return {
            "prompt": {"prompt_embeds": data.batch["prompt_embeds"]},
            "negative_prompt": negative_prompt,
            "guidance_scale": self._guidance_scale,
            "model_kwargs": {},
        }

    def forward_diffusion_model_step(
        self,
        step_latents: torch.Tensor,
        step_timesteps: torch.Tensor,
        model_context: dict[str, Any],
    ) -> torch.Tensor:
        """Run one normalized diffusion prediction for a recorded transition."""
        return self.adapter.forward_step(
            latents=step_latents,
            timestep=step_timesteps,
            prompt=model_context["prompt"],
            negative_prompt=model_context["negative_prompt"],
            guidance_scale=model_context["guidance_scale"],
            **model_context["model_kwargs"],
        ).flow_pred

    def _forward_diffusion_model(self, data: DataProto) -> torch.Tensor:
        """Replay every transition while preserving the trajectory's step dimension."""
        all_latents = data.batch["all_latents"]
        all_timesteps = data.batch["all_timesteps"]
        model_context = self._prepare_diffusion_forward_context(data)
        return torch.stack(
            [
                self.forward_diffusion_model_step(
                    step_latents=all_latents[:, step],
                    step_timesteps=all_timesteps[:, step],
                    model_context=model_context,
                )
                for step in range(all_timesteps.shape[1])
            ],
            dim=1,
        )

    def _build_diffusion_model(
        self,
        mixed_precision: Any,
        model_init_dtype: str,
    ) -> None:
        """Build and parallelize a VeOmni model without constructing a Trainer."""
        from veomni.arguments import OpsImplementationConfig
        from veomni.distributed.offloading import build_activation_offloading_context
        from veomni.distributed.torch_parallelize import build_parallelize_model
        from veomni.models.auto import build_foundation_model

        model_args = self.worker_config.model_args
        strategy_config = self.worker_config.strategy_args.strategy_config
        init_device = strategy_config.get(
            "init_device",
            "meta" if self.dp_mode == "fsdp2" else current_platform.device_type,
        )
        model_root = download_model(model_args.model_name_or_path)
        model_args.model_name_or_path = model_root
        model_path = model_root
        transformer_path = os.path.join(model_root, "transformer")
        if os.path.isfile(os.path.join(transformer_path, "config.json")):
            model_path = transformer_path

        attention_implementation = model_args.attn_implementation
        if attention_implementation in (None, "auto", "fa2"):
            attention_implementation = "flash_attention_2"
        # Diffusion models use a dependency-free backend profile by default.
        # Optimized kernels remain opt-in through ops_implementation.
        ops_kwargs = _DIFFUSION_OPS_DEFAULTS.copy()
        ops_kwargs.update(strategy_config.get("ops_implementation", {}))
        ops_kwargs.setdefault("attn_implementation", attention_implementation)
        self.model = build_foundation_model(
            config_path=model_path,
            weights_path=model_path,
            torch_dtype=model_init_dtype,
            init_device=init_device,
            ops_implementation=OpsImplementationConfig(**ops_kwargs),
            config_kwargs=strategy_config.get("model_config", {}),
        )
        # Match FSDP2's component-initialization switch when prompt tensors are supplied externally.
        if strategy_config.get("init_tokenizer_processor", True):
            from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

            condition_model_type = self.model.config.condition_model_type
            condition_config = MODEL_CONFIG_REGISTRY[condition_model_type]().from_pretrained(
                strategy_config.get("condition_model_path", model_root),
                seed=int(getattr(self.worker.pipeline_config, "seed", 42)),
                **strategy_config.get("condition_model_cfg", {}),
            )
            self.prompt_encoder = MODELING_REGISTRY[condition_model_type]()._from_config(condition_config)
            device = (
                torch.device("cpu")
                if current_platform.device_type == "cpu"
                else torch.device(f"{current_platform.device_type}:{current_platform.current_device()}")
            )
            self.prompt_encoder.to(device)
            self.prompt_encoder.requires_grad_(False)
            self.prompt_encoder.eval()

        parallel_plan = self.model.get_parallel_plan() if hasattr(self.model, "get_parallel_plan") else None

        self.model = build_parallelize_model(
            self.model,
            init_device=init_device,
            weights_path=model_path,
            enable_reshard_after_forward=strategy_config.get("reshard_after_forward", True),
            mixed_precision=mixed_precision,
            enable_gradient_checkpointing=not bool(model_args.disable_gradient_checkpointing),
            basic_modules=list(strategy_config.get("basic_modules", [])),
            enable_reentrant=bool(getattr(model_args, "gradient_checkpointing_use_reentrant", False)),
            enable_forward_prefetch=strategy_config.get("forward_prefetch", True),
            enable_fsdp_offload=strategy_config.get("offload_policy", False),
            broadcast_model_weights_from_rank0=strategy_config.get(
                "broadcast_model_weights_from_rank0",
                self.dp_mode == "fsdp2",
            ),
            max_load_broadcast_size=float(strategy_config.get("max_load_broadcast_size", 20.0)),
            muon_expert_zero_comm=bool(strategy_config.get("muon_expert_zero_comm", False)),
            cpu_load_param_name=getattr(parallel_plan, "cpu_load_param_name", None),
            fqn_to_index_mapping=strategy_config.get("fqn_to_index_mapping"),
        )
        self.model.uses_veomni_native_forward = True
        if hasattr(self.worker.pipeline_config, "actor_infer"):
            from roll.pipeline.diffusion.models.registry import get_diffusion_model_adapter

            adapter_type = get_diffusion_model_adapter(self.worker.pipeline_config.diffusion_model_variant)
            self._initialize_diffusion_scheduler()
            self._configure_diffusion_generation_params()
            self.adapter = adapter_type(
                transformer=self.model,
                scheduler=self.diffusion_scheduler,
            )

        if self.is_trainable:
            self.model.train()
        else:
            self.model.requires_grad_(False)
            self.model.eval()

        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            enable_activation=bool(strategy_config.get("enable_activation_offload", False)),
            enable_gradient_checkpointing=not bool(model_args.disable_gradient_checkpointing),
            activation_gpu_limit=float(strategy_config.get("activation_gpu_limit_gb", 0.0)),
        )

    def initialize(self, model_provider: Callable[..., torch.nn.Module] | None = None) -> None:
        """Initialize through the VeOmni native model builder.

        ``model_provider`` is accepted only for the common ROLL strategy interface.
        VeOmni strategies always build models through VeOmni so sharding, optimizer,
        scheduler, contexts, and checkpoint/offload state stay on one backend path.
        """
        from veomni.arguments import MixedPrecisionConfig

        strategy_config = self.worker_config.strategy_args.strategy_config
        mixed_precision_enabled = strategy_config.get("mixed_precision", True)
        if not isinstance(mixed_precision_enabled, bool):
            raise TypeError("VeOmni strategy_config.mixed_precision must be a bool")
        mixed_precision = MixedPrecisionConfig(
            enable=mixed_precision_enabled,
            param_dtype=strategy_config.get("param_dtype", "bfloat16"),
            reduce_dtype=strategy_config.get("reduce_dtype", "float32"),
            output_dtype=strategy_config.get("output_dtype"),
            cast_forward_inputs=strategy_config.get("cast_forward_inputs", True),
        )

        _apply_veomni_parallel_state(self)
        configured_dtype = parse_dtype(self.worker_config.model_args.dtype)
        self.param_dtype = parse_dtype(mixed_precision.param_dtype) if mixed_precision_enabled else configured_dtype
        self.reduce_dtype = parse_dtype(mixed_precision.reduce_dtype)
        if mixed_precision_enabled and self.param_dtype == torch.float16:
            from torch.distributed._composable.fsdp import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        logger.info(
            f"Mixed precision configured: enable={mixed_precision_enabled}, "
            f"param_dtype={self.param_dtype}, reduce_dtype={self.reduce_dtype}"
        )

        logger.info("Using VeOmni native model builder")
        self._build_diffusion_model(
            mixed_precision=mixed_precision,
            model_init_dtype={
                torch.bfloat16: "bfloat16",
                torch.float16: "float16",
                torch.float32: "float32",
            }[torch.float32 if mixed_precision_enabled else configured_dtype],
        )
        logger.info("VeOmni native diffusion path built model and forward/backward contexts")

        self.cpu_offload_enabled = bool(strategy_config.get("offload_policy", False))
        logger.info(f"FSDP CPU offload policy enabled: {self.cpu_offload_enabled}")

    def load_states(
        self,
        include: Collection[OffloadStateType] | None = None,
        non_blocking: bool = False,
    ) -> None:
        """Load VeOmni model and optimizer states through ROLL's standard strategy interface."""
        device = (
            torch.device("cpu")
            if current_platform.device_type == "cpu"
            else torch.device(f"{current_platform.device_type}:{current_platform.current_device()}")
        )
        if not self.cpu_offload_enabled and (include is None or OffloadStateType.model_params in include):
            load_model_to_gpu(self.model, device)
        if self.prompt_encoder is not None and (include is None or OffloadStateType.other_params in include):
            self.prompt_encoder.to(device, non_blocking=non_blocking)
        if (
            not self.cpu_offload_enabled
            and self.optimizer is not None
            and (include is None or OffloadStateType.optimizer_states in include)
        ):
            load_optimizer(self.optimizer, device)

    def offload_states(
        self,
        include: Collection[OffloadStateType] | None = None,
        non_blocking: bool = False,
    ) -> None:
        """Offload VeOmni model and optimizer states through ROLL's standard strategy interface."""
        should_empty_cache = False
        if not self.cpu_offload_enabled and (include is None or OffloadStateType.model_params in include):
            offload_model_to_cpu(self.model, empty_device_cache=False)
            should_empty_cache = True
        if self.prompt_encoder is not None and (include is None or OffloadStateType.other_params in include):
            self.prompt_encoder.to("cpu", non_blocking=non_blocking)
            should_empty_cache = True
        if (
            not self.cpu_offload_enabled
            and self.optimizer is not None
            and (include is None or OffloadStateType.optimizer_states in include)
        ):
            offload_optimizer(self.optimizer)
            should_empty_cache = True
        if should_empty_cache and current_platform.device_type != "cpu":
            current_platform.empty_cache()

    @torch.no_grad()
    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Replay trajectory micro-batches and let the algorithm consume normalized predictions."""
        self.model.eval()
        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        micro_batches = batch.chunk(chunks=max(math.ceil(batch_size / micro_batch_size), 1))

        outputs = []
        for data in micro_batches:
            with (
                torch.autocast(device_type=current_platform.device_type, dtype=self.param_dtype),
                self.model_fwd_context,
            ):
                model_output = self._forward_diffusion_model(data)
                _, output = forward_func(data, model_output)
            outputs.append(output)
        return collate_fn_to_dict_list(outputs)


class VeOmniTrainStrategy(VeOmniInferStrategy, FSDP2TrainStrategy):
    """VeOmni training strategy using native optimizer and scheduler builders."""

    strategy_name = "veomni_train"
    is_trainable = True

    def initialize(self, model_provider: Callable[..., torch.nn.Module] | None = None) -> None:
        """Initialize VeOmni training through the native VeOmni build path."""
        super().initialize(None)
        from veomni.optim import build_lr_scheduler, build_optimizer

        strategy_config = self.worker_config.strategy_args.strategy_config
        self.async_save_strategy = bool(strategy_config.get("async_save_ckpt", True))
        self.save_only_model = bool(strategy_config.get("save_only_model", False))
        self.checkpoint_future = None

        training_args = self.worker_config.training_args
        max_steps = int(training_args.max_steps or -1)
        if max_steps <= 0:
            max_steps = int(self.worker.pipeline_config.max_steps)
        self.optimizer = build_optimizer(
            self.model,
            lr=training_args.learning_rate,
            betas=(float(training_args.adam_beta1), float(training_args.adam_beta2)),
            weight_decay=training_args.weight_decay,
            fused=True,
            optimizer_type=strategy_config.get("optimizer_type", "adamw"),
            no_decay_modules=list(strategy_config.get("no_decay_modules", [])),
            no_decay_params=list(strategy_config.get("no_decay_params", [])),
            muon_kwargs=strategy_config.get("muon_kwargs"),
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=max_steps,
            lr=training_args.learning_rate,
            lr_min=0.0,
            lr_decay_style=training_args.lr_scheduler_type,
            lr_warmup_ratio=training_args.get_warmup_steps(max_steps) / max_steps,
        )
        logger.info("VeOmni native train path built optimizer and lr scheduler")

        dist.barrier()

    def train_step(
        self,
        batch: DataProto,
        loss_func: Callable[
            [DataProto, dict[str, Any], "VeOmniTrainStrategy"],
            tuple[torch.Tensor, dict[str, Any]],
        ],
        no_sync: bool = False,
    ) -> dict[str, Any]:
        """Execute the common diffusion micro-batch/backward/update shell.

        ``loss_func`` owns algorithm-specific replay and loss construction. DMD
        currently performs its split generator/score updates directly in its workers;
        trajectory algorithms such as FlowGRPO can use this entry point.
        """
        self.model.train()
        mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
        data_iter = batch.make_iterator(mini_batch_size=mini_batch_size, epochs=1)
        mini_steps = batch.batch.batch_size[0] // mini_batch_size
        gradient_accumulation_steps = self.worker_config.training_args.gradient_accumulation_steps
        batch.meta_info["micro_batch_size"] = mini_batch_size
        loss_scale = mini_steps * self.worker.rank_info.dp_size
        metrics: dict[str, Any] = {}

        for step in range(mini_steps):
            data = next(data_iter)
            model_context = self._prepare_diffusion_forward_context(data)
            sync_boundary = (
                (step + 1) % gradient_accumulation_steps == 0 or step + 1 == mini_steps
            ) and not no_sync
            sync_context = (
                self._grad_accumulation_context()
                if not sync_boundary and not no_sync
                else nullcontext()
            )

            with (
                sync_context,
                torch.autocast(device_type=current_platform.device_type, dtype=self.param_dtype),
            ):
                with self.model_fwd_context:
                    loss, reduced_metrics = loss_func(data, model_context, self)
                    append_to_dict(metrics, reduced_metrics)
                    if self.worker_config.apply_loss_scale:
                        loss *= loss_scale
                    loss = loss / gradient_accumulation_steps

                with self.model_bwd_context:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if not sync_boundary:
                continue

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = self.clip_grad_norm(self.worker.pipeline_config.max_grad_norm)
            metrics[f"{self.worker_config.name}/grad_norm"] = grad_norm.item()

            should_offload_optimizer = data.meta_info.get(
                "is_offload_optimizer_states_in_train_step",
                self.worker.pipeline_config.is_offload_optimizer_states_in_train_step,
            )
            if not self.cpu_offload_enabled:
                self.load_states(include=[OffloadStateType.optimizer_states])
            optimizer_stepped = False
            if self.scaler is not None:
                scale_before_step = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = self.scaler.get_scale() >= scale_before_step
            elif torch.isfinite(grad_norm).all():
                self.optimizer.step()
                optimizer_stepped = True
            else:
                logger.warning(f"rank {dist.get_rank()} grad_norm is not finite: {grad_norm}")
            if optimizer_stepped:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if not self.cpu_offload_enabled and should_offload_optimizer:
                self.offload_states(
                    include=[OffloadStateType.optimizer_states],
                    non_blocking=True,
                )

        return metrics

    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        """Clip gradients through VeOmni's parallelized model root."""
        model_clip_grad_norm = getattr(self.model, "clip_grad_norm_", None)
        if callable(model_clip_grad_norm):
            grad_norm = model_clip_grad_norm(max_norm)
        else:
            grad_norm = super().clip_grad_norm(max_norm)
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        return grad_norm

    def unwrap_model(self) -> torch.nn.Module:
        """Return the VeOmni FSDP2 model root.

        VeOmni attaches extra-parallel metadata such as EP-aware grad clipping
        to the parallelized root module, so model update and helper paths should
        not peel it down to ``.module``.
        """
        return self.model
