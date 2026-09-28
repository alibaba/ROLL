"""Qwen-Image diffusion model adapter.

This module implements the unified model operations for Qwen-Image diffusion models,
shared between the trainer (FSDP2) and inference (vllm_omni) sides.

Core responsibilities:
- Single-step noise prediction with Classifier-Free Guidance (CFG)
- Full diffusion generation loop with SDE/ODE integration
- Log-prob replay over existing trajectories (for FlowGRPO PPO ratio)
- Prompt encoding via Qwen2.5-VL text encoder
- Frozen weight lifecycle management (text_encoder, VAE reload after sleep)

The adapter does NOT own model construction. It receives already-constructed
components (transformer, scheduler, text_encoder, vae) and operates on them.
This keeps the adapter decoupled from FSDP2 wrapping, vllm_omni CuMemAllocator,
and other framework-specific initialization.
"""
from __future__ import annotations

import glob
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
from safetensors.torch import load_file as safetensors_load_file

from roll.pipeline.diffusion.models.base import (
    DiffusionAdapter,
    DiffusionPrediction,
    EncodedPrompt,
    DiffuseInput,
    DiffuseOutput,
)
from roll.utils.logging import get_logger


_logger = get_logger()


@dataclass
class QwenImageDiffuseInput(DiffuseInput):
    """Input parameters for a full Qwen-Image diffusion sampling loop.

    Shared fields:
        latents: Noisy latent. Shape [B, S, C].
        prompt_embeds: Text embeddings. Shape [B, T, D].
        prompt_embeds_mask: Text mask. Shape [B, T].
        img_shapes: Spatial shapes for positional encoding.
        txt_seq_lens: Text sequence lengths.
        guidance_scale: CFG scale. > 1.0 enables CFG.
        negative_prompt_embeds: Negative embeddings (for CFG).
        negative_prompt_embeds_mask: Negative mask (for CFG).
        negative_txt_seq_lens: Negative text lengths (for CFG).

    Sampling fields:
        timesteps: Full scheduler timestep sequence. Shape [num_steps].
        sde_type: "sde", "cps", or "ode".
        noise_level: Noise magnitude for SDE steps.
        sde_window: (start_idx, end_idx) for SDE-active region.
        generator: RNG for reproducible sampling.
        logprobs: Whether to compute log-probs.
    """

    # Shared
    latents: torch.Tensor = None
    prompt_embeds: torch.Tensor = None
    prompt_embeds_mask: torch.Tensor = None
    img_shapes: list = None
    txt_seq_lens: list[int] = None
    guidance_scale: float = 1.0
    negative_prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None
    negative_txt_seq_lens: Optional[list[int]] = None

    timesteps: Optional[torch.Tensor] = None
    sde_type: Literal["sde", "cps", "ode"] = "sde"
    noise_level: float = 0.7
    sde_window: tuple[int, int] = (0, 5)
    generator: Optional[torch.Generator] = None
    logprobs: bool = True


class QwenImageAdapter(DiffusionAdapter):
    """Qwen-Image diffusion model operations shared between trainer and inference.

    This adapter encapsulates all model-specific logic for the Qwen-Image architecture:
    - Transformer forward pass with CFG (positive + negative prompt)
    - FlowMatchSDEDiscreteScheduler integration (step / sample_previous_step)
    - Qwen2.5-VL text encoder prompt encoding
    - VAE and text_encoder weight reload after sleep_level=2

    Usage:
        # On inference side (vllm_omni pipeline):
        adapter = QwenImageAdapter(transformer=tf, scheduler=sched, text_encoder=te, vae=vae)
        output = adapter.diffuse(latents, timesteps, prompt_embeds, ...)

        # On training side (FSDP2 strategy):
        #   adapter = QwenImageAdapter(transformer=fsdp_model, scheduler=sched)
        #   Log-probs are computed via strategy.compute_diffusion_step_log_probs()

    Attributes:
        name: Adapter identifier string.
        transformer: The QwenImageTransformer2DModel (may be FSDP-wrapped on trainer side).
        scheduler: FlowMatchSDEDiscreteScheduler instance.
        text_encoder: Qwen2_5_VLForConditionalGeneration (None on trainer side).
        vae: AutoencoderKLQwenImage (None on trainer side).
        device: Target device for operations.
        model_path: Path to model weights on disk (for frozen weight reload).
    """

    name = "qwen_image"
    transformer_cls_name = "QwenImageTransformer2DModel"

    def __init__(
        self,
        transformer,
        scheduler,
        text_encoder=None,
        vae=None,
        device: Optional[torch.device] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize the Qwen-Image adapter with pre-constructed components.

        Args:
            transformer: QwenImageTransformer2DModel instance. Required.
            scheduler: FlowMatchSDEDiscreteScheduler instance. Required.
            text_encoder: Qwen2_5_VLForConditionalGeneration. Optional (inference-only).
            vae: AutoencoderKLQwenImage. Optional (inference-only).
            device: Target device. Defaults to transformer's device if available.
            model_path: Path to model checkpoint on disk (for load_frozen_weights).
        """
        if transformer is None:
            raise ValueError("QwenImageAdapter requires a transformer instance.")
        if scheduler is None:
            raise ValueError("QwenImageAdapter requires a scheduler instance.")

        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
        )
        self.text_encoder = text_encoder
        self.vae = vae
        self.model_path = model_path

        # Resolve device from transformer parameters if not explicitly given
        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(transformer.parameters()).device
            except StopIteration:
                self.device = torch.device("cuda")

    # =========================================================================
    # Internal: Single-step forward with CFG
    # =========================================================================

    def forward_step(
        self,
        *,
        latents: torch.Tensor,
        prompt: EncodedPrompt,
        timestep: torch.Tensor,
        negative_prompt: EncodedPrompt | None = None,
        guidance_scale: float = 0.0,
        **model_kwargs: Any,
    ) -> DiffusionPrediction:
        """Run one normalized Qwen-Image flow prediction with optional CFG.

        This is the atomic model operation shared by sampling and training. It handles:
        1. Positive prompt forward pass -> noise_pred
        2. (If CFG) Negative prompt forward pass -> neg_noise_pred
        3. CFG combination with norm rescaling
        """
        model_timestep = timestep.to(device=latents.device, dtype=latents.dtype) / 1000.0
        guidance = None
        if getattr(self.transformer, "guidance_embeds", False):
            guidance = torch.full(
                [latents.shape[0]],
                guidance_scale,
                device=latents.device,
                dtype=torch.float32,
            )

        flow_pred = self.transformer(
            hidden_states=latents,
            timestep=model_timestep,
            guidance=guidance,
            encoder_hidden_states_mask=prompt["prompt_embeds_mask"],
            encoder_hidden_states=prompt["prompt_embeds"],
            img_shapes=model_kwargs["img_shapes"],
            txt_seq_lens=prompt["txt_seq_lens"],
            attention_kwargs=None,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0:
            if negative_prompt is None:
                raise ValueError("Qwen-Image CFG requires negative prompt embeddings")
            negative_flow = self.transformer(
                hidden_states=latents,
                timestep=model_timestep,
                guidance=guidance,
                encoder_hidden_states_mask=negative_prompt["prompt_embeds_mask"],
                encoder_hidden_states=negative_prompt["prompt_embeds"],
                img_shapes=model_kwargs["img_shapes"],
                txt_seq_lens=negative_prompt["txt_seq_lens"],
                attention_kwargs=None,
                return_dict=False,
            )[0]
            combined = negative_flow + guidance_scale * (flow_pred - negative_flow)
            flow_pred = combined * (
                torch.norm(flow_pred, dim=-1, keepdim=True)
                / torch.norm(combined, dim=-1, keepdim=True)
            )

        return DiffusionPrediction(flow_pred=flow_pred)

    # =========================================================================
    # Public: diffuse (rollout generation)
    # =========================================================================

    def diffuse(self, input: QwenImageDiffuseInput) -> DiffuseOutput:
        """Run the full diffusion sampling loop (generation mode).

        Iterates over all timesteps. Within the SDE window, stochastic noise is
        injected via the scheduler and latent trajectories + log-probs are recorded.
        Outside the window, noise_level is set to 0 making the step deterministic (ODE).

        For ODE mode (sde_type="ode"), the entire loop is deterministic and only the
        final latent is recorded.

        Args:
            input: QwenImageDiffuseInput with sampling-loop fields populated
                (latents, timesteps, prompt_embeds, sde_type, etc.).

        Returns:
            DiffuseOutput with trajectory, log-probs, and timesteps.
        """
        latents = input.latents
        timesteps = input.timesteps
        guidance_scale = input.guidance_scale
        sde_type = input.sde_type
        noise_level = input.noise_level
        sde_window = input.sde_window
        generator = input.generator
        logprobs = input.logprobs

        # Fail-fast: CFG requires all three negative prompt fields together
        if guidance_scale > 1.0:
            missing = []
            if input.negative_prompt_embeds is None:
                missing.append("negative_prompt_embeds")
            if input.negative_prompt_embeds_mask is None:
                missing.append("negative_prompt_embeds_mask")
            if input.negative_txt_seq_lens is None:
                missing.append("negative_txt_seq_lens")
            if missing:
                raise ValueError(
                    f"QwenImageAdapter.diffuse: guidance_scale > 1.0 requires "
                    f"negative prompt fields for CFG, but missing: {missing}"
                )

        prompt = {
            "prompt_embeds": input.prompt_embeds,
            "prompt_embeds_mask": input.prompt_embeds_mask,
            "txt_seq_lens": input.txt_seq_lens,
        }
        negative_prompt = (
            {
                "prompt_embeds": input.negative_prompt_embeds,
                "prompt_embeds_mask": input.negative_prompt_embeds_mask,
                "txt_seq_lens": input.negative_txt_seq_lens,
            }
            if guidance_scale > 1.0
            else None
        )

        collected_latents = []
        collected_log_probs = []
        collected_timesteps = []

        self.scheduler.set_begin_index(0)

        for i, t in enumerate(timesteps):
            # Determine noise level based on SDE window
            if sde_type != "ode":
                if i < sde_window[0]:
                    cur_noise_level = 0.0
                elif i == sde_window[0]:
                    cur_noise_level = noise_level
                    collected_latents.append(latents)  # Record entry point
                elif sde_window[0] < i < sde_window[1]:
                    cur_noise_level = noise_level
                else:
                    cur_noise_level = 0.0
            else:
                cur_noise_level = 0.0

            # Broadcast timestep to the batch; forward_step applies the model convention.
            timestep_broadcast = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            # Single-step noise prediction with CFG
            prediction = self.forward_step(
                latents=latents,
                prompt=prompt,
                timestep=timestep_broadcast,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                img_shapes=input.img_shapes,
            )

            # Scheduler step: sample next latent + compute log-prob
            latents, log_prob, prev_sample_mean, std_dev_t = self.scheduler.step(
                prediction.flow_pred,
                t,
                latents,
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                logprobs=logprobs,
                return_dict=False,
            )

            # Record trajectory within SDE window
            if sde_type != "ode" and sde_window[0] <= i < sde_window[1]:
                collected_latents.append(latents)
                collected_log_probs.append(log_prob)
                collected_timesteps.append(t)
            elif sde_type == "ode":
                collected_timesteps.append(t)

        # For ODE mode, record only the final latent
        if sde_type == "ode":
            collected_latents.append(latents)
            collected_log_probs = [None]

        # Stack collected tensors
        latents_stacked = torch.stack(collected_latents, dim=1)

        if collected_log_probs and collected_log_probs[0] is not None:
            log_probs_stacked = torch.stack(collected_log_probs, dim=1)
        else:
            log_probs_stacked = torch.zeros(
                (latents.shape[0], len(collected_timesteps)),
                dtype=latents.dtype,
                device=latents.device,
            )

        timesteps_stacked = (
            torch.stack(collected_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        )

        return DiffuseOutput(
            latents=latents_stacked,
            final_latents=latents,
            log_probs=log_probs_stacked,
            timesteps=timesteps_stacked,
        )

    # =========================================================================
    # Public: encode_prompt (inference-only)
    # =========================================================================

    def encode_prompt(
        self,
        *,
        prompt_encoder: Any,
        prompt_inputs: Mapping[str, Any],
    ) -> EncodedPrompt:
        """Encode token IDs into text embeddings via the Qwen2.5-VL text encoder.

        This method runs the text encoder forward pass and extracts hidden states,
        skipping the chat template prefix tokens (first prompt_template_encode_start_idx tokens).

        Args:
            prompt_encoder: Qwen2.5-VL text encoder.
            prompt_inputs: Token IDs, attention mask, repetition factor, and maximum length.

        Returns:
            Qwen prompt embeddings, mask, and text sequence lengths.
        """
        prompt_ids = prompt_inputs["prompt_ids"]
        attention_mask = prompt_inputs["attention_mask"]
        num_images_per_prompt = prompt_inputs["num_images_per_prompt"]
        max_sequence_length = prompt_inputs["max_sequence_length"]

        # Ensure 2D input
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

        batch_size = prompt_ids.shape[0]

        # Run text encoder
        prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
            prompt_encoder,
            prompt_ids,
            attention_mask=attention_mask,
        )

        # Truncate to max_sequence_length
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        # Repeat for multi-image generation
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "txt_seq_lens": prompt_embeds_mask.sum(dim=1).tolist(),
        }

    def _get_qwen_prompt_embeds(
        self,
        prompt_encoder: Any,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal: Run Qwen2.5-VL text encoder and extract hidden states.

        Extracts the last hidden state, removes chat template prefix tokens,
        and pads variable-length sequences to a common length.
        """
        dtype = prompt_encoder.dtype
        drop_idx = self.prompt_template_encode_start_idx

        encoder_hidden_states = prompt_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]

        # Extract non-padded hidden states per sample, dropping template prefix
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        # Build attention masks for the extracted sequences
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]

        # Pad to common length
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)
        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        """Extract non-padded hidden states for each sample in the batch."""
        results = []
        for i in range(hidden_states.shape[0]):
            mask = attention_mask[i].bool()
            results.append(hidden_states[i, mask])
        return results

    # =========================================================================
    # Lifecycle: frozen weight management
    # =========================================================================

    def load_frozen_weights(self) -> None:
        """Reload frozen components (text_encoder, VAE) from disk.

        Called after sleep_level=2 wake-up where these weights were discarded
        and model_update only restores the transformer weights. In LoRA mode
        the transformer base is likewise restored via the model_update base
        weight stream (see ``load_transformer_base_weights``), so it never
        needs a disk reload here.
        """
        _logger.info("QwenImageAdapter enter load_frozen_weights")
        if self.model_path is None:
            _logger.warning("QwenImageAdapter.load_frozen_weights: model_path is None, skipping")
            return

        model_path = self.model_path
        device = self.device
        local_files_only = os.path.exists(model_path)

        # --- text_encoder: use from_pretrained (HuggingFace key remapping) ---
        if self.text_encoder is not None:
            from transformers import Qwen2_5_VLForConditionalGeneration

            _logger.info("QwenImageAdapter: loading text_encoder via from_pretrained...")
            new_te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                subfolder="text_encoder",
                local_files_only=local_files_only,
                torch_dtype=next(self.text_encoder.parameters()).dtype,
            )
            src_sd = new_te.state_dict()
            loaded = 0
            for name, param in self.text_encoder.named_parameters():
                if name in src_sd:
                    param.data.copy_(src_sd[name].to(device=device, dtype=param.dtype))
                    loaded += 1
            del src_sd, new_te
            _logger.info("QwenImageAdapter: text_encoder loaded %d params", loaded)
        else:
            _logger.info("QwenImageAdapter: text_encoder is None, skip loading!")

        # --- VAE: direct safetensors load (key names match) ---
        if self.vae is not None:
            vae_dir = os.path.join(model_path, "vae")
            st_files = sorted(glob.glob(os.path.join(vae_dir, "*.safetensors"))) if os.path.isdir(vae_dir) else []
            if st_files:
                full_state_dict = {}
                for st_file in st_files:
                    full_state_dict.update(safetensors_load_file(st_file, device="cpu"))
                missing, unexpected = self.vae.load_state_dict(full_state_dict, strict=False, assign=False)
                vae_loaded = len(full_state_dict) - len(unexpected)
                del full_state_dict
                _logger.info("QwenImageAdapter: vae loaded %d params from %d files", vae_loaded, len(st_files))
            else:
                _logger.warning("QwenImageAdapter: no vae safetensors found in %s", vae_dir)
        else:
            _logger.info("QwenImageAdapter: self.vae is None, skip loading!")

    def load_transformer_base_weights(self, base_sd: dict[str, torch.Tensor]) -> set[str]:
        """Load frozen base weights broadcast by model_update after a level-2 sleep (LoRA mode).

        Same logic as vllm-omni's ``QwenImageTransformer2DModel.load_weights``,
        except parameter names are matched with the ``.base_layer`` segment
        stripped so they resolve through vLLM's LoRA wrappers.
        """
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        # Same fused-projection mapping as vllm-omni's
        # qwen_image_transformer.load_weights.
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        raw_count = 0
        params_dict: dict[str, torch.Tensor] = {}
        for name, param in self.transformer.named_parameters():
            raw_count += 1
            params_dict[name.replace(".base_layer.", ".")] = param
        # LoRA wrapper buffers (lora_a/b_stacked) are plain tensors, not
        # nn.Parameters, so stripping ".base_layer" cannot cause collisions.
        assert len(params_dict) == raw_count, "duplicate names after .base_layer strip"
        for name, buffer in self.transformer.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name.replace(".base_layer.", ".")] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in base_sd.items():
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or param_name in name:
                    continue
                param = params_dict[name.replace(weight_name, param_name)]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                lookup_name = name
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def reinit_non_persistent_states(self) -> None:
        """Recompute RoPE pos_freqs/neg_freqs after sleep_level=2 memory discard.

        These complex-valued tensors are plain instance attributes (not nn.Parameter
        or register_buffer) so they get lost when CuMemAllocator discards GPU memory.
        Since they are deterministic functions of (theta, axes_dim), we recompute them.
        """
        pos_embed = getattr(self.transformer, "pos_embed", None)
        if pos_embed is None:
            _logger.warning("[ROLL-REINIT-ROPE] transformer has no pos_embed, skipping")
            return

        theta = pos_embed.theta
        axes_dim = pos_embed.axes_dim

        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        pos_embed.pos_freqs = torch.cat(
            [
                pos_embed.rope_params(pos_index, axes_dim[0], theta),
                pos_embed.rope_params(pos_index, axes_dim[1], theta),
                pos_embed.rope_params(pos_index, axes_dim[2], theta),
            ],
            dim=1,
        )
        pos_embed.neg_freqs = torch.cat(
            [
                pos_embed.rope_params(neg_index, axes_dim[0], theta),
                pos_embed.rope_params(neg_index, axes_dim[1], theta),
                pos_embed.rope_params(neg_index, axes_dim[2], theta),
            ],
            dim=1,
        )

        # Clear LRU cache that may hold references to old (zeroed) freqs
        if hasattr(pos_embed, "_compute_video_freqs"):
            pos_embed._compute_video_freqs.cache_clear()

        _logger.info(
            "[ROLL-REINIT-ROPE] reinit done: theta=%s axes_dim=%s "
            "pos_freqs.shape=%s neg_freqs.shape=%s device=%s abs_sum=%.4f",
            theta,
            axes_dim,
            list(pos_embed.pos_freqs.shape),
            list(pos_embed.neg_freqs.shape),
            pos_embed.pos_freqs.device,
            pos_embed.pos_freqs.abs().sum().item(),
        )

    def compute_weights_hash(self) -> dict[str, str]:
        """Compute a stable hash for each major component's weights (diagnostics).

        Returns a dict like {"transformer": "a3f8...", "text_encoder": "...", "vae": "..."}
        """
        result = {}
        for comp_name in ["transformer", "text_encoder", "vae"]:
            comp = getattr(self, comp_name, None)
            if comp is None:
                result[comp_name] = "N/A"
                continue
            h = hashlib.sha256()
            num_params = 0
            total_abs_sum = 0.0
            for name, param in comp.named_parameters():
                data = param.data.detach()
                flat = data.reshape(-1)[:1024].float().cpu().numpy()
                h.update(flat.tobytes())
                total_abs_sum += data.float().abs().sum().item()
                num_params += 1
            digest = h.hexdigest()[:16]
            result[comp_name] = f"{digest} (params={num_params}, abs_sum={total_abs_sum:.2f})"
        return result
