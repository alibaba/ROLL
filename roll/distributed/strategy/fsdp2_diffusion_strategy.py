import os
from types import SimpleNamespace
import contextlib
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from transformers import get_scheduler, set_seed

from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.strategy.fsdp2_strategy import (
    FSDP2InferStrategy,
    FSDP2TrainStrategy,
    create_device_mesh_with_ep,
)
from roll.utils.functionals import parse_dtype
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import (
    clear_fsdp2_init_context,
    default_processor_provider,
    default_tokenizer_provider,
    set_fsdp2_init_context,
)
from roll.platforms import current_platform
from roll.pipeline.diffusion.models.registry import get_diffusion_model_adapter
from roll.pipeline.diffusion.models.scheduling_flow_match_sde_discrete import FlowMatchSDEDiscreteScheduler
from roll.utils.checkpoint_manager import download_model
from roll.utils.context_parallel import set_upg_manager
from roll.utils.fsdp_utils import get_init_weight_context_manager
from roll.utils.functionals import append_to_dict
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType


logger = get_logger()


# TODO: Merge this diffusion-specific strategy with VeOmniStrategy.
class FSDP2DiffusionInferStrategy(FSDP2InferStrategy):
    strategy_name = "fsdp2_diffusion_infer"

    def _initialize_diffusion_scheduler(self):
        model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        self.diffusion_scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model_name_or_path,
            subfolder="scheduler",
            local_files_only=os.path.exists(model_name_or_path),
        )

    def _configure_diffusion_generation_params(self):
        """Configure scheduler timesteps and store sampling params from pipeline_config.

        Must be called after _initialize_diffusion_scheduler(), during initialize().
        """
        assert hasattr(self.worker, "pipeline_config") and self.worker.pipeline_config is not None
        generating_args = getattr(
            getattr(self.worker.pipeline_config, "actor_infer", None), "generating_args", None
        )
        generation_config = generating_args.to_dict() if generating_args is not None else {}

        # Store sampling parameters as instance attributes.
        self._noise_level = generation_config["extra_args"]["noise_level"]
        self._sde_type = generation_config["extra_args"]["sde_type"]
        self._guidance_scale = float(generation_config["guidance_scale"])

        # Configure scheduler timesteps.
        num_inference_steps = int(generation_config["num_inference_steps"])
        height = int(generation_config["height"])
        width = int(generation_config["width"])

        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

        vae_scale_factor = 8
        latent_height = height // vae_scale_factor // 2
        latent_width = width // vae_scale_factor // 2
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            latent_height * latent_width,
            self.diffusion_scheduler.config.get("base_image_seq_len", 256),
            self.diffusion_scheduler.config.get("max_image_seq_len", 4096),
            self.diffusion_scheduler.config.get("base_shift", 0.5),
            self.diffusion_scheduler.config.get("max_shift", 1.15),
        )
        self.diffusion_scheduler.set_timesteps(
            num_inference_steps,
            device=current_platform.device_type,
            sigmas=sigmas,
            mu=mu,
        )

    def compute_diffusion_step_log_probs(
        self,
        *,
        model_output: torch.Tensor,
        all_latents: torch.Tensor,
        all_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        current_latents = all_latents[:, :-1]
        next_latents = all_latents[:, 1:]

        assert model_output.ndim == 4, f"model_output.ndim={model_output.ndim}, expected 4"
        assert current_latents.ndim == 4 and next_latents.ndim == 4 and all_timesteps.ndim == 2
        assert model_output.shape[:2] == current_latents.shape[:2], \
            f"horizon mismatch: output={model_output.shape[:2]} vs latents={current_latents.shape[:2]}"

        if model_output.shape[-1] != current_latents.shape[-1]:
            model_output = model_output[..., : current_latents.shape[-1]]

        log_probs = []
        for step in range(all_timesteps.shape[1]):
            step_model_output = model_output[:, step]
            _, step_log_prob, _, _ = self.diffusion_scheduler.sample_previous_step(
                sample=current_latents[:, step].float(),
                model_output=step_model_output,
                timestep=all_timesteps[:, step],
                noise_level=self._noise_level,
                prev_sample=next_latents[:, step].float(),
                sde_type=self._sde_type,
            )
            log_probs.append(step_log_prob)

        return torch.stack(log_probs, dim=1)

    def _prepare_fsdp2_diffusion_model(
        self,
        model_provider,
        *,
        is_trainable: bool,
        default_model_dtype: torch.dtype,
        warmup_collective: bool = False,
    ):
        set_seed(seed=self.worker.pipeline_config.seed)

        if not torch.distributed.is_initialized():
            if current_platform.device_type != "cpu":
                backends_str = f"cpu:gloo,{current_platform.device_type}:{current_platform.communication_backend}"
            else:
                backends_str = current_platform.communication_backend
            torch.distributed.init_process_group(backend=backends_str)

        if warmup_collective:
            dist.all_reduce(torch.zeros(1).to(current_platform.device_type))

        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()

        cp_size = self.worker_config.model_args.ulysses_size
        if cp_size > 1:
            if current_platform.apply_ulysses_patch() is not None:
                set_upg_manager(
                    ulysses_size=cp_size,
                    rank=global_rank,
                    world_size=world_size,
                )
            else:
                cp_size = 1

        if self.worker_config.model_args.ulysses_size != cp_size:
            logger.warning(
                "ulysses_size in config (%s) is not equal to cp_size (%s), using cp_size instead",
                self.worker_config.model_args.ulysses_size,
                cp_size,
            )
            self.worker_config.strategy_args.strategy_config["fsdp_size"] = (
                self.worker_config.strategy_args.strategy_config["fsdp_size"]
                * self.worker_config.model_args.ulysses_size
            )
            self.worker_config.model_args.ulysses_size = cp_size

        self.worker.rank_info.dp_rank = global_rank // cp_size
        self.worker.rank_info.dp_size = world_size // cp_size
        self.worker.rank_info.cp_rank = global_rank % cp_size
        self.worker.rank_info.cp_size = cp_size

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)

        torch_dtype = self.worker_config.strategy_args.strategy_config.get("param_dtype", default_model_dtype)
        torch_dtype = parse_dtype(torch_dtype)
        self.worker_config.model_args.compute_dtype = torch_dtype

        fsdp_size = self.worker_config.strategy_args.strategy_config.get("fsdp_size", 1)

        self.worker_config.strategy_args.strategy_config["fsdp_size"] = fsdp_size
        self.device_mesh, _ = create_device_mesh_with_ep(world_size=world_size, fsdp_size=fsdp_size, efsdp_size=1, ep_size=1)
        self.ep_enabled = False

        model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        config = SimpleNamespace(
            model_type="diffusion_model",
            tie_word_embeddings=False,
            vision_config=None,
            model_name_or_path=model_name_or_path,
        )

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not getattr(config, "tie_word_embeddings", False),
            mesh=self.device_mesh,
        )

        set_fsdp2_init_context(init_context)
        try:
            model = model_provider(
                tokenizer=self.tokenizer,
                model_args=self.worker_config.model_args,
                is_trainable=is_trainable,
            )
        finally:
            clear_fsdp2_init_context()

        self.is_lora = self.worker_config.model_args.lora_target is not None
        return model, torch_dtype, cp_size

    def _embeds_padding_2_no_padding(self, data: DataProto) -> None:
        required_keys = ("prompt_embeds", "prompt_embeds_mask")
        missing_keys = [k for k in required_keys if k not in data.batch.keys()]
        def _to_no_padding(embeds: torch.Tensor, embeds_mask: torch.Tensor, name: str):
            seq_lens_before = embeds_mask.to(torch.bool).sum(dim=-1)

            embeds_list: List[torch.Tensor] = []
            mask_list: List[torch.Tensor] = []
            for i in range(embeds_mask.shape[0]):
                curr_mask = embeds_mask[i].to(torch.bool)
                embeds_list.append(embeds[i, curr_mask, :])
                mask_list.append(torch.ones(int(curr_mask.sum().item()), dtype=torch.bool, device=embeds_mask.device))

            embeds_nested = torch.nested.as_nested_tensor(embeds_list, layout=torch.jagged)
            embeds_mask_nested = torch.nested.as_nested_tensor(mask_list, layout=torch.jagged)
            seq_lens_after = embeds_nested.offsets().diff()
            return embeds_nested, embeds_mask_nested

        prompt_embeds_nested, prompt_embeds_mask_nested = _to_no_padding(
            data.batch["prompt_embeds"], data.batch["prompt_embeds_mask"], "prompt"
        )
        data.batch["prompt_embeds"] = prompt_embeds_nested
        data.batch["prompt_embeds_mask"] = prompt_embeds_mask_nested

        has_negative_prompt_embeds = "negative_prompt_embeds" in data.batch.keys()
        has_negative_prompt_embeds_mask = "negative_prompt_embeds_mask" in data.batch.keys()
        if has_negative_prompt_embeds:
            negative_prompt_embeds_nested, negative_prompt_embeds_mask_nested = _to_no_padding(
                data.batch["negative_prompt_embeds"],
                data.batch["negative_prompt_embeds_mask"],
                "negative_prompt",
            )
            data.batch["negative_prompt_embeds"] = negative_prompt_embeds_nested
            data.batch["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask_nested


    def _materialize_embed_pair_from_no_padding(
        self,
        data: DataProto,
        embeds_key: str,
        embeds_mask_key: str,
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        embeds = data.batch[embeds_key]
        embeds_mask = data.batch[embeds_mask_key]
        assert isinstance(embeds, torch.Tensor) and isinstance(embeds_mask, torch.Tensor)
        assert embeds.is_nested and embeds_mask.is_nested, f"{name} expects nested tensors"

        batch_size = int(embeds.size(0))
        embed_dim = int(embeds.size(-1))
        seq_lens = embeds.offsets().diff()
        max_seq_len = int(seq_lens.max().item())
        embeds_dense = torch.nested.to_padded_tensor(
            embeds,
            padding=0,
            output_size=(batch_size, max_seq_len, embed_dim),
        )
        embeds_mask_dense = torch.nested.to_padded_tensor(
            embeds_mask,
            padding=0,
            output_size=(batch_size, max_seq_len),
        ).to(torch.bool)
        txt_seq_lens = [int(x) for x in seq_lens.tolist()]

        return embeds_dense, embeds_mask_dense, txt_seq_lens

    def forward_diffusion_model_step(
        self,
        step_latents: torch.Tensor,
        step_timesteps: torch.Tensor,
        img_shapes: torch.Tensor,
        model_ctx: dict,
    ) -> torch.Tensor:
        prompt = {
            "prompt_embeds": model_ctx["prompt_embeds"],
            "prompt_embeds_mask": model_ctx["prompt_embeds_mask"],
            "txt_seq_lens": model_ctx["prompt_txt_seq_lens"],
        }
        negative_prompt = {
            "prompt_embeds": model_ctx["negative_prompt_embeds"],
            "prompt_embeds_mask": model_ctx["negative_prompt_embeds_mask"],
            "txt_seq_lens": model_ctx["negative_prompt_txt_seq_lens"],
        }
        prediction = self.adapter.forward_step(
            latents=step_latents,
            prompt=prompt,
            timestep=step_timesteps,
            negative_prompt=negative_prompt,
            guidance_scale=model_ctx["guidance_scale"],
            img_shapes=img_shapes,
        )
        return prediction.flow_pred

    def _forward_diffusion_model(self, data: DataProto) -> torch.Tensor:
        # TODO: implement fsdp2 inference
        raise NotImplementedError

    def initialize(self, model_provider):
        model, _, _ = self._prepare_fsdp2_diffusion_model(
            model_provider,
            is_trainable=False,
            default_model_dtype=torch.bfloat16,
        )

        self.setup_fsdp2_configuration()
        self.initialize_fsdp2_model(model)
        self._initialize_diffusion_scheduler()
        self._configure_diffusion_generation_params()

        # Create adapter for unified model operations (shared with rollout side)
        variant = self.worker.pipeline_config.diffusion_model_variant
        adapter_cls = get_diffusion_model_adapter(variant)
        self.adapter = adapter_cls(
            transformer=self.model,
            scheduler=self.diffusion_scheduler,
        )

        dist.barrier()

    def forward_step(self, batch, forward_func):
        self.model.eval()
        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        num_microbatches = max(batch_size // micro_batch_size, 1)
        micro_batches = batch.chunk(chunks=num_microbatches)

        cp_size = self.worker.rank_info.cp_size
        batch_num_tokens = self._get_batch_num_tokens(batch)
        batch.meta_info["batch_num_tokens"] = {k: v // cp_size for k, v in batch_num_tokens.items()}
        global_valid_tokens = self._get_global_valid_samples(batch)
        batch.meta_info["global_valid_samples"] = {k: v // cp_size for k, v in global_valid_tokens.items()}

        loss_scale = num_microbatches * self.worker.rank_info.dp_size
        losses_reduced = []

        for data in micro_batches:
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=self.param_dtype,
            ):
                model_output = self._forward_diffusion_model(data)
                loss, loss_reduced = forward_func(data, model_output)
                if self.worker_config.apply_loss_scale:
                    loss *= loss_scale
            losses_reduced.append(loss_reduced)

        return collate_fn_to_dict_list(losses_reduced)


class FSDP2DiffusionTrainStrategy(FSDP2DiffusionInferStrategy, FSDP2TrainStrategy):
    strategy_name = "fsdp2_diffusion_train"

    def initialize(self, model_provider):
        model, _, _ = self._prepare_fsdp2_diffusion_model(
            model_provider,
            is_trainable=True,
            default_model_dtype=torch.float32,
            warmup_collective=True,
        )

        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")

        self.setup_fsdp2_configuration(is_trainable=True)
        # Cast model according to reduce dtype
        logger.info(f"[FSDP2-diffusion] Casting trainable model parameters to reduce_dtype={self.reduce_dtype}")
        model = model.to(dtype=self.reduce_dtype)

        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        self.initialize_fsdp2_model(model)
        self._initialize_diffusion_scheduler()
        self._configure_diffusion_generation_params()

        # FSDP2 fully_shard may reset requires_grad on DTensor shards.
        # Re-establish LoRA adapter flags: only LoRA params are trainable.
        if self.is_lora:
            for name, param in self.model.named_parameters():
                if "ema_lora" in name:
                    param.requires_grad_(False)
                elif "lora_" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Create adapter for unified model operations (shared with rollout side)
        variant = self.worker.pipeline_config.diffusion_model_variant
        adapter_cls = get_diffusion_model_adapter(variant)
        self.adapter = adapter_cls(
            transformer=self.model,
            scheduler=self.diffusion_scheduler,
        )

        trainable_params = (param for param in self.model.parameters() if param.requires_grad)
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.worker_config.training_args.learning_rate,
            betas=(
                self.worker_config.training_args.adam_beta1,
                self.worker_config.training_args.adam_beta2,
            ),
            weight_decay=self.worker_config.training_args.weight_decay,
            fused=self.worker_config.strategy_args.strategy_config.get("fused_optimizer", False),
        )

        self.scheduler = get_scheduler(
            self.worker_config.training_args.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=self.worker_config.training_args.get_warmup_steps(
                self.worker_config.training_args.max_steps
            ),
            num_training_steps=self.worker_config.training_args.max_steps,
        )

        dist.barrier()

    def set_active_adapter(self, adapter_name: str):
        """Switch the currently active LoRA adapter.

        PEFT's ``set_adapter`` flips ``requires_grad`` on adapter params (active
        → True, others → False).  We must re-establish the intended state:
        only ``default`` LoRA params are trainable, ``ema_lora`` and base
        model params are frozen.  Otherwise ``clip_grad_norm_`` after the last
        timestep (where ``ema_lora`` was activated) would skip the ``default``
        params that actually hold gradients, yielding grad_norm=0.
        """
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(adapter_name)
            # Diagnostic: verify active adapter actually changed
            from peft.tuners.tuners_utils import BaseTunerLayer
            for m in self.model.modules():
                if isinstance(m, BaseTunerLayer):
                    logger.info("[adapter_diag] set_adapter(%s) → active_adapters=%s", adapter_name, m.active_adapters)
                    break
            if self.is_lora:
                for name, param in self.model.named_parameters():
                    if "ema_lora" in name:
                        param.requires_grad_(False)
                    elif "lora_" in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
        else:
            logger.warning("Model does not support set_adapter; LoRA may not be properly initialized.")

    def copy_adapter(self, source: str = "default", target: str = "ema_lora"):
        """Copy all weights of the source adapter into the target adapter.

        Used to initialize the EMA shadow in DiffNFT (shadow = current). Like
        ema_update_lora, both adapters share identical shard placements, so we
        copy this rank's local shard directly via .to_local() — no all-gather,
        no extra communication overhead.
        """
        from torch.distributed.tensor import DTensor

        if not self.is_lora:
            return

        source_params = {}
        target_params = {}
        for name, param in self.model.named_parameters():
            if f".{target}." in name:
                target_params[name] = param
            elif f".{source}." in name:
                source_params[name] = param

        if not source_params or not target_params:
            logger.warning("copy_adapter: source or target LoRA params not found, skipping.")
            return

        for source_name, source_param in source_params.items():
            target_name = source_name.replace(f".{source}.", f".{target}.")
            if target_name not in target_params:
                continue
            target_param = target_params[target_name]

            source_data = source_param.data
            target_data = target_param.data
            if isinstance(source_data, DTensor):
                source_data = source_data.to_local()
            if isinstance(target_data, DTensor):
                target_data = target_data.to_local()

            target_data.copy_(source_data)

    @contextlib.contextmanager
    def lora_adapters_disabled(self):
        """Temporarily disable all LoRA adapters for the DiffNFT base-model reference forward.

        Toggles layer by layer via peft's BaseTunerLayer.enable_adapters (the
        same API as the disable_adapter path in fsdp2_strategy); the original
        active state is restored after the forward. The caller must manage
        no_grad and adapter restoration (set_active_adapter) at the outer level.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        lora_layers = [m for m in self.model.modules() if isinstance(m, BaseTunerLayer)]
        for layer in lora_layers:
            layer.enable_adapters(False)
        try:
            yield
        finally:
            for layer in lora_layers:
                layer.enable_adapters(True)

    def ema_update_lora(self, ema_decay: float):
        """EMA-update the ema_lora adapter using the default adapter's weights.

        shadow = ema_decay * shadow + (1 - ema_decay) * live

        Only called in DiffNFT mode. Both ema_lora and default parameters
        belong to the same transformer-block FSDP unit and are sharded as
        DTensors. Since both share identical shard placements, we update this
        rank's local shard in place directly via .to_local() — no all-gather,
        no extra communication or memory overhead.
        """
        from torch.distributed.tensor import DTensor

        if not self.is_lora:
            return

        live_params = {}
        shadow_params = {}
        for name, param in self.model.named_parameters():
            if "ema_lora" in name:
                shadow_params[name] = param
            elif "lora_" in name and param.requires_grad:
                live_params[name] = param

        if not live_params or not shadow_params:
            logger.warning("ema_update_lora: live or shadow LoRA params not found, skipping.")
            return

        for live_name, live_param in live_params.items():
            shadow_name = live_name.replace(".default.", ".ema_lora.")
            if shadow_name not in shadow_params:
                continue
            shadow_param = shadow_params[shadow_name]

            live_data = live_param.data
            shadow_data = shadow_param.data

            if isinstance(live_data, DTensor):
                live_data = live_data.to_local()
            if isinstance(shadow_data, DTensor):
                shadow_data = shadow_data.to_local()

            shadow_data.mul_(ema_decay).add_(live_data, alpha=1 - ema_decay)

    def _prepare_diffusion_forward_context(self, data: DataProto) -> dict:
        """Prepare model inputs shared by all diffusion algorithms (embeddings, CFG)."""
        self._embeds_padding_2_no_padding(data)
        prompt_embeds_dense, prompt_embeds_mask_dense, prompt_txt_seq_lens = self._materialize_embed_pair_from_no_padding(
            data, "prompt_embeds", "prompt_embeds_mask", "prompt",
        )
        guidance_scale = self._guidance_scale
        negative_prompt_embeds_dense = None
        negative_prompt_embeds_mask_dense = None
        negative_prompt_txt_seq_lens = None
        if "negative_prompt_embeds" in data.batch.keys():
            negative_prompt_embeds_dense, negative_prompt_embeds_mask_dense, negative_prompt_txt_seq_lens = (
                self._materialize_embed_pair_from_no_padding(
                    data, "negative_prompt_embeds", "negative_prompt_embeds_mask", "negative_prompt",
                )
            )
        else:
            assert guidance_scale <= 1.0, "guidance_scale > 1.0 requires negative_prompt_embeds"
        return {
            "prompt_embeds": prompt_embeds_dense,
            "prompt_embeds_mask": prompt_embeds_mask_dense,
            "prompt_txt_seq_lens": prompt_txt_seq_lens,
            "guidance_scale": guidance_scale,
            "negative_prompt_embeds": negative_prompt_embeds_dense,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask_dense,
            "negative_prompt_txt_seq_lens": negative_prompt_txt_seq_lens,
        }

    def train_step(self, batch, forward_and_backward, no_sync: bool = False):
        self.model.train()
        mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
        data_iter = batch.make_iterator(mini_batch_size=mini_batch_size, epochs=1)
        mini_steps = batch.batch.batch_size[0] // mini_batch_size

        cp_size = self.worker.rank_info.cp_size
        batch_num_tokens = self._get_batch_num_tokens(batch)
        batch.meta_info["batch_num_tokens"] = {k: v // cp_size for k, v in batch_num_tokens.items()}
        global_valid_tokens = self._get_global_valid_samples(batch)
        batch.meta_info["global_valid_samples"] = {k: v // cp_size for k, v in global_valid_tokens.items()}
        loss_scale = mini_steps * self.worker.rank_info.dp_size
        batch.meta_info["micro_batch_size"] = mini_batch_size

        gradient_accumulation_steps = self.worker_config.training_args.gradient_accumulation_steps
        is_offload_optimizer_states_in_train_step = batch.meta_info.get("is_offload_optimizer_states_in_train_step", True)
        metrics = {}

        for step in range(mini_steps):
            data: DataProto = next(data_iter)
            model_ctx = self._prepare_diffusion_forward_context(data)

            sync_boundary = ((step + 1) % gradient_accumulation_steps == 0 or (step + 1 == mini_steps)) and not no_sync
            if self.reduce_scatter_during_grad_accumulation:
                sync_context = contextlib.nullcontext()
            else:
                sync_context = (
                    self._grad_accumulation_context()
                    if not sync_boundary and not no_sync
                    else contextlib.nullcontext()
                )

            with (
                sync_context,
                torch.autocast(
                    device_type=current_platform.device_type,
                    dtype=self.param_dtype,
                ),
            ):
                step_metrics = forward_and_backward(
                    data,
                    model_ctx,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_scale=loss_scale if self.worker_config.apply_loss_scale else None,
                    scaler=self.scaler,
                )
                append_to_dict(metrics, step_metrics)

            if sync_boundary:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = self.clip_grad_norm(
                    max_norm=self.worker.pipeline_config.max_grad_norm,
                )
                metrics[f"{self.worker_config.name}/grad_norm"] = grad_norm.item()

                # Lazy load optimizer states just before step (Megatron pattern)
                if not self.cpu_offload_enabled:
                    self.load_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if not torch.isfinite(grad_norm):
                        logger.warning(f"WARN: rank {dist.get_rank()} grad_norm is not finite: {grad_norm}")
                    else:
                        self.optimizer.step()

                # Offload optimizer states after step, before scheduler.step (Megatron pattern)
                if not self.cpu_offload_enabled and is_offload_optimizer_states_in_train_step:
                    self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Drop stale host copies of model params after the optimizer update,
                # aligning with parent FSDP2StrategyBase.train_step. Without it the
                # write-once fast path in _offload_model_params_to_backend keeps the
                # pre-update host copy, so the next reload silently reverts the step.
                self._cleanup_offloaded_model_params()

        # Log cuda memory
        max_memory_allocated = torch.cuda.memory.max_memory_allocated() / (1024 ** 3)
        max_memory_reserved = torch.cuda.memory.max_memory_reserved() / (1024 ** 3)
        metrics[f"system/max_memory_allocated@max"] = max_memory_allocated
        metrics[f"system/max_memory_reserved@max"] = max_memory_reserved
        return metrics
