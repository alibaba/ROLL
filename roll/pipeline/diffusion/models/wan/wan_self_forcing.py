"""Wan causal-model plugin for ODE distillation and Self-Forcing DMD."""

from __future__ import annotations

import contextvars
import math
import os
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch
import torch.utils.checkpoint
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch import nn
from torch.nn.attention.flex_attention import flex_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from roll.pipeline.diffusion.models.base import EncodedPrompt
from roll.utils.checkpoint_manager import download_model

try:
    from diffusers.models.transformers.transformer_wan import (
        WanAttention,
        WanAttnProcessor,
        _get_qkv_projections,
    )
except ImportError:
    WanAttention = Any

    class WanAttnProcessor:
        """Minimal placeholder for environments without diffusers Wan support."""

        def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            raise RuntimeError("Self-Forcing Wan requires diffusers WanTransformer3DModel support")

    def _get_qkv_projections(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise RuntimeError("Self-Forcing Wan requires diffusers WanTransformer3DModel support")

KVCache = list[dict[str, torch.Tensor]]
CrossAttnCache = list[dict[str, torch.Tensor | bool]]


@dataclass(frozen=True, slots=True)
class _WanARAttentionState:
    kv_cache: KVCache | None
    crossattn_cache: CrossAttnCache | None
    current_start: int
    frame_seq_length: int
    local_attn_size: int
    sink_size: int
    block_mask: Any | None


_AR_STATE: contextvars.ContextVar[_WanARAttentionState] = contextvars.ContextVar(
    "wan_ar_attention_state",
    default=_WanARAttentionState(None, None, 0, 0, -1, 0, None),
)

WAN_AR_MAX_TEXT_LEN = 512
_compiled_flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode="max-autotune-no-cudagraphs",
)


@dataclass(frozen=True, slots=True)
class WanVAEConfig:
    """Minimal Wan VAE shape metadata required by diffusion algorithms."""

    z_dim: int
    scale_factor_spatial: int
    scale_factor_temporal: int


@dataclass(slots=True)
class WanPromptEncoder:
    """Provider-owned Wan prompt encoder kept outside FSDP model state."""

    tokenizer: Any
    text_encoder: nn.Module
    vae_config: WanVAEConfig

    @contextmanager
    def device_context(
        self,
        device: torch.device,
        *,
        offload_after: bool,
    ) -> Iterator[None]:
        """Keep the frozen text encoder on ``device`` for one prompt-encoding call."""
        self.text_encoder.to(device)
        try:
            yield
        finally:
            if offload_after:
                self.text_encoder.to("cpu")


@contextmanager
def _ar_context(
    kv_cache: KVCache | None,
    crossattn_cache: CrossAttnCache | None,
    current_start: int,
    frame_seq_length: int,
    local_attn_size: int,
    sink_size: int,
    block_mask: Any | None = None,
) -> Iterator[None]:
    """Bind AR cache context for forward and checkpoint recompute."""
    token = _AR_STATE.set(
        _WanARAttentionState(
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            frame_seq_length=frame_seq_length,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            block_mask=block_mask,
        )
    )
    try:
        yield
    finally:
        _AR_STATE.reset(token)


def _wan_rope_with_frame_offset(
    rope_module: torch.nn.Module,
    hidden_states: torch.Tensor,
    current_start_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = rope_module.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    split_sizes = [rope_module.t_dim, rope_module.h_dim, rope_module.w_dim]
    freqs_cos = rope_module.freqs_cos.split(split_sizes, dim=1)
    freqs_sin = rope_module.freqs_sin.split(split_sizes, dim=1)

    end_frames = current_start_frames + ppf
    if end_frames > freqs_cos[0].shape[0]:
        raise ValueError(
            f"AR rotary offset {current_start_frames}+{ppf} exceeds rope_max_seq_len "
            f"({freqs_cos[0].shape[0]})."
        )

    freqs_cos_f = freqs_cos[0][current_start_frames:end_frames].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

    freqs_sin_f = freqs_sin[0][current_start_frames:end_frames].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

    out_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
    out_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
    return out_cos, out_sin


def _run_wan_blocks(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    kv_cache: KVCache | None,
    crossattn_cache: CrossAttnCache | None,
    local_attn_size: int,
    sink_size: int,
    block_mask: Any | None,
) -> torch.Tensor:
    if not torch.is_grad_enabled() or not model.gradient_checkpointing:
        for block in model.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        return hidden_states

    checkpoint_fn = getattr(model, "_gradient_checkpointing_func", None)
    ar_state = _AR_STATE.get()
    current_start = ar_state.current_start
    frame_seq_length = ar_state.frame_seq_length

    for layer_idx, block in enumerate(model.blocks):
        xattn_was_init = (
            bool(crossattn_cache[layer_idx]["is_init"])
            if crossattn_cache is not None
            else False
        )

        def block_with_context(
            block: nn.Module,
            hs: torch.Tensor,
            enc_hs: torch.Tensor,
            t_proj: torch.Tensor,
            r_emb: tuple[torch.Tensor, torch.Tensor],
            current_layer_idx: int = layer_idx,
            current_xattn_was_init: bool = xattn_was_init,
        ) -> torch.Tensor:
            with _ar_context(
                kv_cache,
                crossattn_cache,
                current_start,
                frame_seq_length,
                local_attn_size,
                sink_size,
                block_mask,
            ):
                if crossattn_cache is not None:
                    crossattn_cache[current_layer_idx]["is_init"] = current_xattn_was_init
                return block(hs, enc_hs, t_proj, r_emb)

        if checkpoint_fn is None:
            hidden_states = torch.utils.checkpoint.checkpoint(
                block_with_context,
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                use_reentrant=False,
            )
            continue

        hidden_states = checkpoint_fn(
            block_with_context,
            block,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
        )
    return hidden_states


def _wan_ar_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    kv_cache = kwargs.pop("kv_cache", None)
    crossattn_cache = kwargs.pop("crossattn_cache", None)
    current_start = kwargs.pop("current_start", 0)
    frame_seq_length = kwargs.pop("frame_seq_length", 0)
    local_attn_size = kwargs.pop("local_attn_size", -1)
    sink_size = kwargs.pop("sink_size", 0)
    block_mask = kwargs.pop("block_mask", None)

    with _ar_context(
        kv_cache,
        crossattn_cache,
        current_start,
        frame_seq_length,
        local_attn_size,
        sink_size,
        block_mask,
    ):
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if current_start > 0 and frame_seq_length > 0:
            rotary_emb = _wan_rope_with_frame_offset(self.rope, hidden_states, current_start // frame_seq_length)
        else:
            rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        timestep_seq_len = None
        if timestep.ndim == 2:
            timestep = timestep.repeat_interleave(post_patch_height * post_patch_width, dim=1)
            timestep_seq_len = timestep.shape[1]
            timestep = timestep.flatten()

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            timestep_seq_len=timestep_seq_len,
        )
        if timestep_seq_len is None:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))

        freqs_cos, freqs_sin = rotary_emb
        rotary_emb = (
            freqs_cos.to(device=hidden_states.device, dtype=hidden_states.dtype),
            freqs_sin.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

        hidden_states = _run_wan_blocks(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            kv_cache,
            crossattn_cache,
            local_attn_size,
            sink_size,
            block_mask,
        )

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale.to(hidden_states.device))
            + shift.to(hidden_states.device)
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def _create_block_causal_mask(
    *,
    sequence_length: int,
    frame_sequence_length: int,
    num_frame_per_block: int,
    local_attn_size: int,
    device: torch.device,
) -> Any:
    """Create the block-causal attention mask used by ODE distillation."""
    from torch.nn.attention.flex_attention import create_block_mask

    padded_length = math.ceil(sequence_length / 128) * 128
    block_length = num_frame_per_block * frame_sequence_length

    def attention_mask(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        block_end = (query // block_length + 1) * block_length
        allowed = key < block_end
        if local_attn_size != -1:
            allowed &= key >= block_end - local_attn_size * frame_sequence_length
        valid = (query < sequence_length) & (key < sequence_length)
        return (valid & allowed) | (query == key)

    return create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        device=device,
        _compile=False,
    )


class CausalWanAttnProcessor(WanAttnProcessor):
    """Wan block-causal attention with an optional inference KV cache."""

    def __init__(self, attn_implementation: str) -> None:
        self.attn_implementation = attn_implementation
        super().__init__()

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        ar_state = _AR_STATE.get()
        is_cross = encoder_hidden_states is not None
        if not is_cross and ar_state.kv_cache is not None:
            return self._self_attn_kv_cached(attn, hidden_states, rotary_emb, ar_state.kv_cache, ar_state)
        if is_cross and ar_state.crossattn_cache is not None:
            return self._cross_attn_cached(
                attn,
                hidden_states,
                encoder_hidden_states,
                ar_state.crossattn_cache,
            )
        return self._forward_no_cache(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

    def _run_attention(
        self,
        attn: WanAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = ALL_ATTENTION_FUNCTIONS[self.attn_implementation](
            attn,
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attention_mask=None,
            dropout=0.0,
            is_causal=False,
        )[0]
        attn_out = attn_out.flatten(2, 3).type_as(query)
        attn_out = attn.to_out[0](attn_out)
        return attn.to_out[1](attn_out)

    @staticmethod
    def _run_block_causal_attention(
        attn: WanAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: Any,
    ) -> torch.Tensor:
        sequence_length = query.shape[1]
        padded_length = math.ceil(sequence_length / 128) * 128
        if padded_length != sequence_length:
            padding = (0, 0, 0, 0, 0, padded_length - sequence_length)
            query = torch.nn.functional.pad(query, padding)
            key = torch.nn.functional.pad(key, padding)
            value = torch.nn.functional.pad(value, padding)

        hidden_states = _compiled_flex_attention(
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
            block_mask=block_mask,
        )[:, :, :sequence_length].transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)

    @staticmethod
    def _apply_rope(tensor: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        freqs_cos, freqs_sin = rotary_emb
        freqs_cos = freqs_cos.to(device=tensor.device, dtype=tensor.dtype)
        freqs_sin = freqs_sin.to(device=tensor.device, dtype=tensor.dtype)
        x1, x2 = tensor.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(tensor)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(tensor)

    def _forward_no_cache(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if attn.add_k_proj is not None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = self._apply_rope(query, rotary_emb)
            key = self._apply_rope(key, rotary_emb)

        block_mask = _AR_STATE.get().block_mask
        if encoder_hidden_states is None and block_mask is not None:
            return self._run_block_causal_attention(attn, query, key, value, block_mask)
        return self._run_attention(attn, query, key, value)

    def _self_attn_kv_cached(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        kv_cache: KVCache,
        ar_state: _WanARAttentionState,
    ) -> torch.Tensor:
        current_start = ar_state.current_start
        frame_seq_len = ar_state.frame_seq_length
        local_attn_size = ar_state.local_attn_size
        sink_size = ar_state.sink_size
        entry = kv_cache[attn._sf_layer_idx]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states=None)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = self._apply_rope(query, rotary_emb)
            key = self._apply_rope(key, rotary_emb)

        num_new_tokens = key.shape[1]
        current_end = current_start + num_new_tokens
        kv_cache_size = entry["k"].shape[1]
        sink_tokens = sink_size * frame_seq_len if frame_seq_len > 0 else 0
        max_attn_size = kv_cache_size if local_attn_size == -1 else local_attn_size * frame_seq_len
        prev_global_end = int(entry["global_end_index"].item())
        prev_local_end = int(entry["local_end_index"].item())

        if local_attn_size != -1 and current_end > prev_global_end and num_new_tokens + prev_local_end > kv_cache_size:
            num_evicted = num_new_tokens + prev_local_end - kv_cache_size
            num_rolled = prev_local_end - num_evicted - sink_tokens
            with torch.no_grad():
                entry["k"][:, sink_tokens : sink_tokens + num_rolled].copy_(
                    entry["k"][:, sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled].clone()
                )
                entry["v"][:, sink_tokens : sink_tokens + num_rolled].copy_(
                    entry["v"][:, sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled].clone()
                )
            local_end = prev_local_end + current_end - prev_global_end - num_evicted
            local_start = local_end - num_new_tokens
        else:
            local_end = prev_local_end + current_end - prev_global_end
            local_start = local_end - num_new_tokens

        attn_start = max(0, local_end - max_attn_size)
        current_offset = max(0, attn_start - local_start)
        current_key = key[:, current_offset:].to(query.dtype)
        current_value = value[:, current_offset:].to(query.dtype)
        if attn_start < local_start:
            history_key = entry["k"][:, attn_start:local_start].to(query.dtype)
            history_value = entry["v"][:, attn_start:local_start].to(query.dtype)
            key_used = torch.cat([history_key, current_key], dim=1)
            value_used = torch.cat([history_value, current_value], dim=1)
        else:
            key_used = current_key
            value_used = current_value

        with torch.no_grad():
            entry["k"][:, local_start:local_end].copy_(key.detach().to(entry["k"].dtype))
            entry["v"][:, local_start:local_end].copy_(value.detach().to(entry["v"].dtype))

        attn_out = self._run_attention(
            attn,
            query,
            key_used,
            value_used,
        )
        entry["global_end_index"].fill_(current_end)
        entry["local_end_index"].fill_(local_end)
        return attn_out

    def _cross_attn_cached(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        crossattn_cache: CrossAttnCache,
    ) -> torch.Tensor:
        entry = crossattn_cache[attn._sf_layer_idx]
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))

        if torch.is_grad_enabled():
            key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)[1:]
            key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
            with torch.no_grad():
                entry["k"][:, : key.shape[1]].copy_(key.detach().to(entry["k"].dtype))
                entry["v"][:, : value.shape[1]].copy_(value.detach().to(entry["v"].dtype))
            entry["is_init"] = True
            key_used, value_used = key, value
        elif not entry.get("is_init", False):
            key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)[1:]
            key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
            entry["k"][:, : key.shape[1]].copy_(key.to(entry["k"].dtype))
            entry["v"][:, : value.shape[1]].copy_(value.to(entry["v"].dtype))
            entry["is_init"] = True
            key_used, value_used = key, value
        else:
            key_used = entry["k"].to(query.dtype)
            value_used = entry["v"].to(query.dtype)

        return self._run_attention(attn, query, key_used, value_used)


class WanDiffusionWrapper(nn.Module):
    """Causal Wan2.1 transformer used by ODE distillation and Self-Forcing."""

    _no_split_modules = ["WanTransformerBlock"]

    def __init__(
        self,
        model_name_or_path: str,
        transformer_name_or_path: str | None = None,
        num_frame_per_block: int = 3,
        local_attn_size: int = -1,
        sink_size: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        input_dtype: torch.dtype | None = None,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        super().__init__()
        self.uses_per_frame_timesteps = True
        self.num_frame_per_block = num_frame_per_block
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.parallelize_weights_path = transformer_name_or_path or os.path.join(model_name_or_path, "transformer")
        self.input_dtype = input_dtype
        self._block_causal_masks: dict[tuple[int, int, int, torch.device], Any] = {}

        from diffusers import WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel

        text_encoder = UMT5EncoderModel.from_pretrained(
            model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
        ).eval()
        text_encoder.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            subfolder="tokenizer",
            trust_remote_code=True,
        )
        if transformer_name_or_path is None:
            self.model = WanTransformer3DModel.from_pretrained(
                model_name_or_path,
                subfolder="transformer",
                torch_dtype=dtype,
            )
        else:
            self.model = WanTransformer3DModel.from_pretrained(transformer_name_or_path, torch_dtype=dtype)
        self.prompt_encoder = WanPromptEncoder(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae_config=WanVAEConfig(
                z_dim=int(getattr(self.model.config, "in_channels", 16)),
                scale_factor_spatial=8,
                scale_factor_temporal=4,
            ),
        )
        self.model.eval()
        self.model.forward = types.MethodType(_wan_ar_forward, self.model)
        processor = CausalWanAttnProcessor(attn_implementation=attn_implementation)
        attention_config = SimpleNamespace(_attn_implementation=attn_implementation)

        for idx, block in enumerate(self.model.blocks):
            block.attn1.set_processor(processor)
            block.attn2.set_processor(processor)
            block.attn1.config = attention_config
            block.attn2.config = attention_config
            block.attn1._sf_layer_idx = idx
            block.attn2._sf_layer_idx = idx

    def set_trainable(self, is_trainable: bool) -> "WanDiffusionWrapper":
        """Set transformer trainability while keeping prompt encoding frozen."""
        self.model.requires_grad_(is_trainable)
        self.model.train(is_trainable)
        self.prompt_encoder.text_encoder.requires_grad_(False)
        self.prompt_encoder.text_encoder.eval()
        return self

    def train(self, mode: bool = True) -> "WanDiffusionWrapper":
        """Set transformer mode while keeping the frozen text encoder in eval mode."""
        super().train(mode)
        self.prompt_encoder.text_encoder.eval()
        return self

    def save_pretrained(
        self,
        save_directory: str,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        """Save the trainable transformer as a reusable Diffusers model artifact."""
        if state_dict is None:
            self.model.save_pretrained(save_directory, **kwargs)
            return

        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        self.model.save_config(save_directory)
        transformer_state = {
            key.removeprefix("model."): value.contiguous()
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        save_file(
            transformer_state,
            os.path.join(save_directory, "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt"},
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_cache: KVCache | None = None,
        crossattn_cache: CrossAttnCache | None = None,
        current_start: int | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Run causal Wan with the same prediction-forward signature as Diffusers Wan."""
        del kwargs
        if self.input_dtype is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype=self.input_dtype)
            hidden_states = hidden_states.to(dtype=self.input_dtype)
        input_timestep = timestep[:, 0] if timestep.ndim == 2 and kv_cache is not None else timestep
        ar_kwargs: dict[str, Any] = {}
        if kv_cache is not None:
            ar_kwargs["kv_cache"] = kv_cache
        if crossattn_cache is not None:
            ar_kwargs["crossattn_cache"] = crossattn_cache
        if current_start is not None:
            ar_kwargs["current_start"] = current_start
        ar_kwargs["local_attn_size"] = self.local_attn_size
        ar_kwargs["sink_size"] = self.sink_size
        _, _, _, height, width = hidden_states.shape
        _, patch_height, patch_width = self.model.config.patch_size
        ar_kwargs["frame_seq_length"] = (height // patch_height) * (width // patch_width)
        if kv_cache is None:
            mask_key = (
                hidden_states.shape[2],
                ar_kwargs["frame_seq_length"],
                self.num_frame_per_block,
                hidden_states.device,
            )
            block_mask = self._block_causal_masks.get(mask_key)
            if block_mask is None:
                block_mask = _create_block_causal_mask(
                    sequence_length=hidden_states.shape[2] * ar_kwargs["frame_seq_length"],
                    frame_sequence_length=ar_kwargs["frame_seq_length"],
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                    device=hidden_states.device,
                )
                self._block_causal_masks[mask_key] = block_mask
            ar_kwargs["block_mask"] = block_mask

        output = self.model(
            hidden_states=hidden_states,
            timestep=input_timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            **ar_kwargs,
        )
        flow_pred = output[0] if isinstance(output, tuple) else output
        if return_dict:
            return Transformer2DModelOutput(sample=flow_pred)
        return (flow_pred,)

    def create_ar_state(
        self,
        *,
        latent_shape: list[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Any]:
        """Allocate model-specific caches for one AR generation pass."""
        batch_size = latent_shape[0]
        _, patch_height, patch_width = self.model.config.patch_size
        height, width = latent_shape[-2:]
        frame_seq_length = (height // patch_height) * (width // patch_width)
        first_attn = self.model.blocks[0].attn1
        num_layers = len(self.model.blocks)
        kv_heads = int(first_attn.heads)
        kv_dim = int(first_attn.to_k.out_features) // kv_heads
        cache_frames = (
            latent_shape[1]
            if self.local_attn_size == -1 or self.model.gradient_checkpointing
            else self.local_attn_size
        )
        cache_size = cache_frames * frame_seq_length
        return {
            "frame_seq_length": frame_seq_length,
            "kv_cache": [
                {
                    "k": torch.zeros([batch_size, cache_size, kv_heads, kv_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, cache_size, kv_heads, kv_dim], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
                for _ in range(num_layers)
            ],
            "crossattn_cache": [
                {
                    "k": torch.zeros([batch_size, WAN_AR_MAX_TEXT_LEN, kv_heads, kv_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, WAN_AR_MAX_TEXT_LEN, kv_heads, kv_dim], dtype=dtype, device=device),
                    "is_init": False,
                }
                for _ in range(num_layers)
            ],
        }

    def forward_ar(
        self,
        *,
        noisy_image_or_video: torch.Tensor,
        prompt: EncodedPrompt,
        timestep: torch.Tensor,
        ar_state: dict[str, Any],
        frame_start: int,
    ) -> torch.Tensor:
        """Run one causal AR model step using opaque model-owned cache state."""
        output = self(
            hidden_states=noisy_image_or_video.permute(0, 2, 1, 3, 4),
            timestep=timestep,
            encoder_hidden_states=prompt["prompt_embeds"],
            kv_cache=ar_state["kv_cache"],
            crossattn_cache=ar_state["crossattn_cache"],
            current_start=frame_start * ar_state["frame_seq_length"],
            return_dict=False,
        )
        return output[0].permute(0, 2, 1, 3, 4)


def build_model(
    tokenizer: Any,
    model_args: Any,
    training_args: Any = None,
    is_trainable: bool = False,
) -> nn.Module:
    """Build the provider-owned causal Wan transformer for FSDP2."""
    del tokenizer, training_args
    input_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[model_args.dtype]
    attn_implementation = model_args.attn_implementation
    if attn_implementation in (None, "auto"):
        attn_implementation = "flash_attention_2"
    # FSDP2 invokes this provider before wrapping the module. Wan-specific loading,
    # causal mutation, prompt encoding, and AR cache capabilities therefore stay here.
    # Trainable parameters remain FP32 masters; FSDP mixed precision and input_dtype
    # control BF16/FP16 computation after sharding.
    model = WanDiffusionWrapper(
        model_name_or_path=download_model(model_args.model_name_or_path),
        transformer_name_or_path=(
            download_model(model_args.model_config_kwargs["transformer_name_or_path"])
            if "transformer_name_or_path" in model_args.model_config_kwargs
            else None
        ),
        num_frame_per_block=int(model_args.model_config_kwargs.get("num_frame_per_block", 3)),
        local_attn_size=int(model_args.model_config_kwargs["local_attn_size"]),
        sink_size=int(model_args.model_config_kwargs["sink_size"]),
        dtype=torch.float32 if is_trainable else input_dtype,
        input_dtype=input_dtype,
        attn_implementation=attn_implementation,
    ).set_trainable(is_trainable)
    if not model_args.disable_gradient_checkpointing:
        model.model.enable_gradient_checkpointing()
        model.model._gradient_checkpointing_func = partial(
            torch.utils.checkpoint.checkpoint,
            use_reentrant=bool(model_args.gradient_checkpointing_use_reentrant),
        )
    return model
