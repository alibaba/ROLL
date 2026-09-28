"""Wan diffusion model adapter."""

from __future__ import annotations

from collections.abc import Mapping
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.distributed._composable.fsdp import register_fsdp_forward_method

from roll.pipeline.diffusion.models.base import (
    DiffusionAdapter,
    DiffusionPrediction,
    EncodedPrompt,
)


WAN_MAX_TEXT_LEN = 512


class WanDiffusionAdapter(DiffusionAdapter):
    """Normalize Wan prompt, prediction, feature, and decode operations."""

    name = "wan2_1"
    transformer_cls_name = "WanTransformer3DModel"

    def __init__(
        self,
        transformer: nn.Module,
        scheduler: Any | None = None,
    ) -> None:
        """Bind a backend-built Wan transformer."""
        super().__init__(transformer, scheduler)

    def encode_prompt(
        self,
        *,
        prompt_encoder: Any,
        prompt_inputs: Mapping[str, Any],
    ) -> EncodedPrompt:
        """Encode prompt strings into Wan text embeddings."""
        text_encoder = prompt_encoder.text_encoder
        prompts = [" ".join(prompt.split()) for prompt in prompt_inputs["prompts"]]
        device = prompt_inputs["device"]
        dtype = prompt_inputs["dtype"]
        with torch.no_grad():
            encoded = prompt_encoder.tokenizer(
                prompts,
                padding="max_length",
                max_length=WAN_MAX_TEXT_LEN,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            attention_mask = encoded["attention_mask"].to(device)
            output = text_encoder(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = output.last_hidden_state
            for hidden_state, sequence_length in zip(
                prompt_embeds,
                attention_mask.sum(dim=1).long(),
            ):
                hidden_state[sequence_length:] = 0.0
        return {"prompt_embeds": prompt_embeds.to(dtype)}

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
        """Run one Wan flow prediction with optional CFG."""
        del model_kwargs
        if getattr(self.transformer, "uses_per_frame_timesteps", False) and timestep.ndim == 1:
            timestep = timestep[:, None].expand(-1, latents.shape[1])
        positive_inputs = self._model_inputs(latents, prompt, timestep)
        if guidance_scale == 0.0:
            raw_output = self.transformer(**positive_inputs)
            flow_pred = self._flow_from_output(raw_output)
        else:
            if negative_prompt is None:
                raise ValueError("Wan CFG requires negative prompt embeddings")
            negative_inputs = self._model_inputs(latents, negative_prompt, timestep)
            raw_output = self.transformer(**positive_inputs)
            positive_flow = self._flow_from_output(raw_output)
            negative_flow = self._flow_from_output(self.transformer(**negative_inputs))
            flow_pred = positive_flow + (positive_flow - negative_flow) * guidance_scale

        if flow_pred.shape != latents.shape:
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        pred_x0 = None
        if self.scheduler is not None:
            pred_x0 = self.scheduler.x0_from_flow_pred(
                flow_pred=flow_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])
        return DiffusionPrediction(
            flow_pred=flow_pred,
            pred_x0=pred_x0,
        )

    def feature_dim(self) -> int:
        """Return the width of Wan tokens before the output head."""
        return int(self.transformer.proj_out.in_features)

    def extract_features(
        self,
        *,
        latents: torch.Tensor,
        prompt: EncodedPrompt,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Return prompt- and timestep-conditioned Wan features as a video grid."""
        if not hasattr(self.transformer, "_roll_forward_features"):

            def forward_features(module: nn.Module, **model_inputs: Any) -> torch.Tensor:
                features: list[torch.Tensor] = []

                def capture_features(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                    features.append(args[0])

                handle = module.proj_out.register_forward_pre_hook(capture_features)
                try:
                    module.forward(**model_inputs)
                finally:
                    handle.remove()
                return torch.cat(features, dim=0)

            self.transformer._roll_forward_features = MethodType(forward_features, self.transformer)
            register_fsdp_forward_method(self.transformer, "_roll_forward_features")

        model_inputs = self._model_inputs(latents, prompt, timestep)
        hidden_states = model_inputs["hidden_states"]
        if hidden_states.ndim == 6:
            hidden_states = hidden_states[:, 0]
        batch_size, _, num_frames, height, width = hidden_states.shape
        patch_frames, patch_height, patch_width = self.transformer.config.patch_size
        tokens = self.transformer._roll_forward_features(**model_inputs)
        return tokens.transpose(1, 2).reshape(
            batch_size,
            tokens.shape[-1],
            num_frames // patch_frames,
            height // patch_height,
            width // patch_width,
        )

    def decode_latents(
        self,
        *,
        latents: torch.Tensor,
        vae: nn.Module | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode Wan latents to pixels."""
        del kwargs
        if vae is None:
            raise ValueError("WanDiffusionAdapter.decode_latents requires a VAE")
        vae_param = next(vae.parameters())
        vae_latents = latents.permute(0, 2, 1, 3, 4).to(device=vae_param.device, dtype=vae_param.dtype)
        if hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
            latents_mean = torch.tensor(
                vae.config.latents_mean,
                device=vae_latents.device,
                dtype=vae_latents.dtype,
            ).view(1, vae.config.z_dim, 1, 1, 1)
            latents_std = torch.tensor(
                vae.config.latents_std,
                device=vae_latents.device,
                dtype=vae_latents.dtype,
            ).view(1, vae.config.z_dim, 1, 1, 1)
            vae_latents = vae_latents * latents_std + latents_mean
        decoded = vae.decode(vae_latents)
        pixels = decoded.sample if hasattr(decoded, "sample") else decoded
        return pixels.clamp(-1, 1).permute(0, 2, 1, 3, 4)

    def _model_inputs(
        self,
        latents: torch.Tensor,
        prompt: EncodedPrompt,
        timestep: torch.Tensor,
    ) -> dict[str, Any]:
        hidden_states = latents.permute(0, 2, 1, 3, 4)
        model_timestep = (
            timestep
            if getattr(self.transformer, "uses_per_frame_timesteps", False)
            else timestep[:, 0] if timestep.ndim == 2 else timestep
        )
        if getattr(self.transformer, "uses_veomni_native_forward", False):
            hidden_states = hidden_states.unsqueeze(1)
            return {
                "latents": hidden_states,
                "hidden_states": hidden_states,
                "timestep": model_timestep.view(-1, 1),
                "encoder_hidden_states": prompt["prompt_embeds"].unsqueeze(1),
                "training_target": torch.zeros_like(hidden_states),
            }
        return {
            "hidden_states": hidden_states,
            "timestep": model_timestep,
            "encoder_hidden_states": prompt["prompt_embeds"],
        }

    @staticmethod
    def _flow_from_output(output: Any) -> torch.Tensor:
        if hasattr(output, "predictions") and output.predictions is not None:
            predictions = output.predictions
            if isinstance(predictions, (list, tuple)):
                return torch.cat([WanDiffusionAdapter._flow_from_output(item) for item in predictions])
            return WanDiffusionAdapter._flow_from_output(predictions)
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
