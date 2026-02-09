import os

import torch
import torch.nn.functional as F
from numpy import save

# Try to import the capture utilities
try:
    from tests.distributed.strategy.log_probs.layer_states_capture import is_enabled, save_dict, save_tensor
except ImportError:
    # If not available, create no-op functions
    def is_enabled():
        return False

    def save_tensor(tensor, name, subdir=""):
        pass

    def save_dict(data, name, subdir=""):
        pass


def apply_qwen3vl_patches():
    """Apply patches to Qwen3VL model classes."""
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            ALL_ATTENTION_FUNCTIONS,
            Callable,
            CustomBaseModelOutputWithPast,
            DynamicCache,
            Qwen3VLModel,
            Qwen3VLTextDecoderLayer,
            Qwen3VLTextMLP,
            Qwen3VLTextModel,
            Qwen3VLVisionAttention,
            Qwen3VLVisionBlock,
            Qwen3VLVisionModel,
            apply_rotary_pos_emb_vision,
            create_causal_mask,
            eager_attention_forward,
        )

        # Patch Qwen3VLTextModel.forward
        original_text_model_forward = Qwen3VLTextModel.forward

        def patched_text_model_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **kwargs,
        ):
            # Capture inputs_embeds
            if inputs_embeds is not None and is_enabled():
                save_tensor(inputs_embeds, "inputs_embeds", subdir="embeddings")

            # Capture visual embeddings
            if deepstack_visual_embeds is not None and is_enabled():
                for i, visual_embed in enumerate(deepstack_visual_embeds):
                    save_tensor(visual_embed, f"deepstack_visual_embeds_{i}", subdir="embeddings")

            if visual_pos_masks is not None and is_enabled():
                save_tensor(visual_pos_masks, "visual_pos_masks", subdir="embeddings")

            # Call original forward
            if is_enabled():
                if (input_ids is None) ^ (inputs_embeds is not None):
                    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

                # torch.jit.trace() doesn't support cache objects in the output
                if use_cache and past_key_values is None and not torch.jit.is_tracing():
                    past_key_values = DynamicCache(config=self.config)

                if inputs_embeds is None:
                    inputs_embeds = self.embed_tokens(input_ids)

                if cache_position is None:
                    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                    )

                # the hard coded `3` is for temporal, height and width.
                if position_ids is None:
                    position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
                elif position_ids.ndim == 2:
                    position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

                if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                    text_position_ids = position_ids[0]
                    position_ids = position_ids[1:]
                else:
                    text_position_ids = position_ids[0]

                attention_mask = create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )

                hidden_states = inputs_embeds

                # create position embeddings to be shared across the decoder layers
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                # decoder layers
                layer_states = {}
                for layer_idx, decoder_layer in enumerate(self.layers):
                    layer_outputs, layer_state = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        layer_ids=layer_idx,
                        **kwargs,
                    )
                    hidden_states = layer_outputs
                    layer_states[f"layer_{layer_idx}"] = layer_state
                    layer_states[f"layer_{layer_idx}_visual_pos_masks"] = visual_pos_masks

                    # add visual features to the hidden states of first several layers
                    if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                        layer_states[f"layer_{layer_idx}_deepstack_visual_embeds"] = deepstack_visual_embeds[layer_idx]
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_idx],
                        )
                        layer_states[f"layer_{layer_idx}_deepstack"] = hidden_states

                hidden_states = self.norm(hidden_states)

                return CustomBaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values,
                    layer_states=layer_states,
                )
            else:
                output = original_text_model_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    **kwargs,
                )

            # Capture layer_states
            if hasattr(output, "layer_states") and output.layer_states is not None and is_enabled():
                save_dict(output.layer_states, "layer_states", subdir="layers")

            return output

        Qwen3VLTextModel.forward = patched_text_model_forward

        # Patch Qwen3VLModel.forward to capture visual embeddings
        original_model_forward = Qwen3VLModel.forward

        def patched_model_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            cache_position=None,
            **kwargs,
        ):
            # Call original forward
            output = original_model_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

            # Capture layer_states from output
            if hasattr(output, "layer_states") and output.layer_states is not None and is_enabled():
                save_dict(output.layer_states, "layer_states", subdir="layers")

            return output

        Qwen3VLModel.forward = patched_model_forward

        # Patch Qwen3VLVisionModel.forward to capture visual embeddings
        original_vision_forward = Qwen3VLVisionModel.forward

        def patched_vision_forward(self, hidden_states, grid_thw, **kwargs):
            if is_enabled():
                save_tensor(hidden_states, "visual_hidden_states", subdir="embeddings")

                hidden_states = self.patch_embed(hidden_states)

                pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
                hidden_states = hidden_states + pos_embeds

                rotary_pos_emb = self.rot_pos_emb(grid_thw)

                seq_len, _ = hidden_states.size()
                hidden_states = hidden_states.reshape(seq_len, -1)
                rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                position_embeddings = (emb.cos(), emb.sin())

                cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                    dim=0,
                    # Select dtype based on the following factors:
                    #  - FA2 requires that cu_seqlens_q must have dtype int32
                    #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
                    # See https://github.com/huggingface/transformers/pull/34852 for more information
                    dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                )
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

                deepstack_feature_lists = []
                for layer_num, blk in enumerate(self.blocks):
                    hidden_states = blk(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings,
                        layer_ids=layer_num,
                        **kwargs,
                    )
                    save_tensor(hidden_states, f"visual_hidden_states_{layer_num}", subdir="embeddings")
                    if layer_num in self.deepstack_visual_indexes:
                        deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                            hidden_states
                        )
                        save_tensor(deepstack_feature, f"visual_deepstack_feature_{layer_num}", subdir="embeddings")
                        deepstack_feature_lists.append(deepstack_feature)

                hidden_states = self.merger(hidden_states)
                save_tensor(hidden_states, "final_visual_image_embeds", subdir="embeddings")
                print(f"[DEBUG] Visual Atten Type: {self.blocks[0].attn.config._attn_implementation}")

                output = hidden_states, deepstack_feature_lists
            else:
                return original_vision_forward(self, hidden_states, grid_thw, **kwargs)

            # Visual model returns (image_embeds, deepstack_image_embeds)
            if is_enabled():
                if isinstance(output, tuple) and len(output) >= 1:
                    image_embeds = output[0]
                    save_tensor(image_embeds, "visual_image_embeds", subdir="embeddings")

                    if len(output) >= 2 and output[1] is not None:
                        deepstack_embeds = output[1]
                        for i, embed in enumerate(deepstack_embeds):
                            save_tensor(embed, f"visual_deepstack_embeds_{i}", subdir="embeddings")

            return output

        Qwen3VLVisionModel.forward = patched_vision_forward

        original_vision_decoder_block_forward = Qwen3VLVisionBlock.forward

        def patched_vision_decoder_block_forward(
            self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs
        ):
            if is_enabled():
                layer_ids = kwargs.pop("layer_ids", 0)
                norm_result = self.norm1(hidden_states)
                save_tensor(norm_result, f"visual_block_{layer_ids}_after_norm1", subdir="embeddings")

                attn_result = self.attn(
                    norm_result,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                    layer_ids=layer_ids,
                    **kwargs,
                )
                save_tensor(attn_result, f"visual_block_{layer_ids}_after_attn", subdir="embeddings")

                hidden_states = hidden_states + attn_result

                norm_result = self.norm2(hidden_states)
                save_tensor(norm_result, f"visual_block_{layer_ids}_after_norm2", subdir="embeddings")

                mlp_result = self.mlp(norm_result)
                save_tensor(mlp_result, f"visual_block_{layer_ids}_after_mlp", subdir="embeddings")

                hidden_states = hidden_states + mlp_result
                return hidden_states
            return original_vision_decoder_block_forward(
                self, hidden_states, cu_seqlens, position_embeddings, **kwargs
            )

        Qwen3VLVisionBlock.forward = patched_vision_decoder_block_forward

        original_vision_attention_forward = Qwen3VLVisionAttention.forward

        def patched_vision_attention_forward(
            self,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=None,
            position_embeddings=None,
            **kwargs,
        ):
            if is_enabled():
                layer_ids = kwargs.pop("layer_ids", 0)
                seq_length = hidden_states.shape[0]
                query_states, key_states, value_states = (
                    self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
                )
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

                query_states = query_states.transpose(0, 1).unsqueeze(0)
                key_states = key_states.transpose(0, 1).unsqueeze(0)
                value_states = value_states.transpose(0, 1).unsqueeze(0)

                if layer_ids == 0:
                    save_tensor(query_states, f"visual_block_{layer_ids}_query_states", subdir="embeddings")
                    save_tensor(key_states, f"visual_block_{layer_ids}_key_states", subdir="embeddings")
                    save_tensor(value_states, f"visual_block_{layer_ids}_value_states", subdir="embeddings")
                    save_tensor(cos, f"visual_block_{layer_ids}_cos", subdir="embeddings")
                    save_tensor(sin, f"visual_block_{layer_ids}_sin", subdir="embeddings")

                attention_interface: Callable = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                if self.config._attn_implementation == "flash_attention_2":
                    # Flash Attention 2: Use cu_seqlens for variable length attention
                    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                    attn_output, _ = attention_interface(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        cu_seq_lens_q=cu_seqlens,
                        cu_seq_lens_k=cu_seqlens,
                        max_length_q=max_seqlen,
                        max_length_k=max_seqlen,
                        is_causal=False,
                        **kwargs,
                    )

                    if layer_ids == 0:
                        save_tensor(attn_output, f"visual_block_{layer_ids}_after_attn_output", subdir="embeddings")
                else:
                    # Other implementations: Process each chunk separately
                    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                    splits = [
                        torch.split(tensor, lengths.tolist(), dim=2)
                        for tensor in (query_states, key_states, value_states)
                    ]

                    attn_outputs = [
                        attention_interface(
                            self,
                            q,
                            k,
                            v,
                            attention_mask=None,
                            scaling=self.scaling,
                            dropout=0.0 if not self.training else self.attention_dropout,
                            is_causal=False,
                            **kwargs,
                        )[0]
                        for q, k, v in zip(*splits)
                    ]
                    attn_output = torch.cat(attn_outputs, dim=1)

                attn_output = attn_output.reshape(seq_length, -1).contiguous()
                attn_output = self.proj(attn_output)

                if layer_ids == 0:
                    save_tensor(attn_output, f"visual_block_{layer_ids}_after_o_output", subdir="embeddings")
                return attn_output
            else:
                return original_vision_attention_forward(
                    self, hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings, **kwargs
                )

        Qwen3VLVisionAttention.forward = patched_vision_attention_forward

        original_text_mlp_forward = Qwen3VLTextMLP.forward

        def patched_text_mlp_forward(self, x, layer_ids=0):
            if is_enabled():
                up_proj = self.up_proj(x)
                save_tensor(up_proj, f"text_block_{layer_ids}_up_proj", subdir="layers")
                gate_proj = self.gate_proj(x)
                save_tensor(gate_proj, f"text_block_{layer_ids}_gate_proj", subdir="layers")
                act_fn = self.act_fn(gate_proj)
                save_tensor(act_fn, f"text_block_{layer_ids}_act_fn", subdir="layers")
                down_proj = self.down_proj(act_fn * up_proj)
                save_tensor(down_proj, f"text_block_{layer_ids}_down_proj", subdir="layers")

                if layer_ids == 0:
                    up_proj_weight = self.up_proj.weight
                    save_tensor(up_proj_weight, f"text_block_{layer_ids}_up_proj_weight", subdir="layers")
                    gate_proj_weight = self.gate_proj.weight
                    save_tensor(gate_proj_weight, f"text_block_{layer_ids}_gate_proj_weight", subdir="layers")
                    down_proj_weight = self.down_proj.weight
                    save_tensor(down_proj_weight, f"text_block_{layer_ids}_down_proj_weight", subdir="layers")
                return down_proj
            return original_text_mlp_forward(self, x)

        Qwen3VLTextMLP.forward = patched_text_mlp_forward

        original_text_decoder_layer_forward = Qwen3VLTextDecoderLayer.forward

        def patched_text_decoder_layer_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):
            if is_enabled():
                layer_ids = kwargs.pop("layer_ids", 0)
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)

                before_attn = hidden_states
                # Self Attention
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

                after_attn = hidden_states

                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)

                after_post_norm = hidden_states

                hidden_states = self.mlp(hidden_states, layer_ids=layer_ids)
                after_mlp = hidden_states

                hidden_states = residual + hidden_states

                after_mlp_res = hidden_states

                layer_states = {
                    "before_attn": before_attn,
                    "after_attn": after_attn,
                    "after_post_norm": after_post_norm,
                    "after_mlp": after_mlp,
                    "after_mlp_res": after_mlp_res,
                }
                return hidden_states, layer_states
            else:
                return original_text_decoder_layer_forward(
                    self,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    **kwargs,
                )

        Qwen3VLTextDecoderLayer.forward = patched_text_decoder_layer_forward

        return True
    except ImportError as e:
        print(f"Warning: Could not import Qwen3VL models for patching: {e}")
        return False


# -----------------------------
# Megatron/mcore patches
# -----------------------------
def apply_qwen3vl_megatron_patches():
    """
    Apply patches to mcore_adapter Qwen3-VL classes to capture per-layer states
    (similar naming/layout to the HF patch above) for divergence debugging.
    """
    try:
        from mcore_adapter.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLGPTModel  # type: ignore[import-not-found]
        from mcore_adapter.models.qwen3_vl.transformer_block import (
            Qwen3VLTransformerBlock,
        )  # type: ignore[import-not-found]

        # -------------------------
        # Patch Qwen3VLGPTModel.forward
        # Capture embeddings + visual injection inputs at the text stack boundary.
        # -------------------------
        original_gpt_forward = Qwen3VLGPTModel.forward

        def patched_gpt_forward(
            self,
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=None,
            labels=None,
            inference_context=None,
            packed_seq_params=None,
            extra_block_kwargs=None,
            runtime_gather_output=None,
            *,
            inference_params=None,
            loss_mask=None,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        ):
            if is_enabled():
                if decoder_input is not None:
                    save_tensor(decoder_input, "inputs_embeds", subdir="embeddings")
                if visual_pos_masks is not None:
                    save_tensor(visual_pos_masks, "visual_pos_masks", subdir="embeddings")
                if deepstack_visual_embeds is not None:
                    for i, visual_embed in enumerate(deepstack_visual_embeds):
                        save_tensor(visual_embed, f"deepstack_visual_embeds_{i}", subdir="embeddings")

            return original_gpt_forward(
                self,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

        Qwen3VLGPTModel.forward = patched_gpt_forward

        # -------------------------
        # Patch Qwen3VLTransformerBlock to capture per-layer intermediates.
        # Uses hooks to avoid changing model math.
        # Also patches _deepstack_process to attribute "deepstack" state to the last executed layer.
        # -------------------------
        original_block_forward = Qwen3VLTransformerBlock.forward
        original_deepstack_process = Qwen3VLTransformerBlock._deepstack_process

        def _first_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, (list, tuple)):
                for item in x:
                    t = _first_tensor(item)
                    if t is not None:
                        return t
                return None
            if hasattr(x, "unwrap"):  # WrappedTensor
                try:
                    return x.unwrap()
                except Exception:
                    return None
            return None

        def patched_deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
            out = original_deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds)
            if is_enabled():
                idx = getattr(self, "_capture_last_layer_idx", None)
                if idx is not None:
                    save_tensor(out, f"layer_states_layer_{idx}_deepstack", subdir="layers")
            return out

        Qwen3VLTransformerBlock._deepstack_process = patched_deepstack_process

        def patched_block_forward(self, *args, **kwargs):
            if not is_enabled():
                return original_block_forward(self, *args, **kwargs)

            # Last layer idx (global layer number - 1) whose forward just ran on this PP rank.
            self._capture_last_layer_idx = None
            handles = []

            def _register(module, fn):
                try:
                    h = module.register_forward_hook(fn)
                    handles.append(h)
                except Exception:
                    pass

            try:
                for layer in getattr(self, "layers", []):
                    layer_idx = getattr(layer, "layer_number", None)
                    if layer_idx is not None:
                        layer_idx = int(layer_idx) - 1

                    # input_layernorm -> before_attn
                    ln = getattr(layer, "input_layernorm", None)
                    if ln is not None:
                        _register(
                            ln,
                            (
                                lambda idx: (
                                    lambda _m, _inp, out: (
                                        save_tensor(
                                            _first_tensor(out),
                                            f"layer_states_layer_{idx}_before_attn",
                                            subdir="layers",
                                        )
                                        if idx is not None
                                        else None
                                    )
                                )
                            )(layer_idx),
                        )

                    # self_attention -> after_attn (attention output before residual)
                    attn = getattr(layer, "self_attention", None)
                    if attn is not None:
                        _register(
                            attn,
                            (
                                lambda idx: (
                                    lambda _m, _inp, out: (
                                        save_tensor(
                                            _first_tensor(out), f"layer_states_layer_{idx}_after_attn", subdir="layers"
                                        )
                                        if idx is not None
                                        else None
                                    )
                                )
                            )(layer_idx),
                        )

                    # post-attn norm (naming differs across versions)
                    post_ln = getattr(layer, "pre_mlp_layernorm", None) or getattr(
                        layer, "post_attention_layernorm", None
                    )
                    if post_ln is not None:
                        _register(
                            post_ln,
                            (
                                lambda idx: (
                                    lambda _m, _inp, out: (
                                        save_tensor(
                                            _first_tensor(out),
                                            f"layer_states_layer_{idx}_after_post_norm",
                                            subdir="layers",
                                        )
                                        if idx is not None
                                        else None
                                    )
                                )
                            )(layer_idx),
                        )

                    # mlp -> after_mlp
                    mlp = getattr(layer, "mlp", None)
                    if mlp is not None:
                        _register(
                            mlp,
                            (
                                lambda idx: (
                                    lambda _m, _inp, out: (
                                        save_tensor(
                                            _first_tensor(out), f"layer_states_layer_{idx}_after_mlp", subdir="layers"
                                        )
                                        if idx is not None
                                        else None
                                    )
                                )
                            )(layer_idx),
                        )

                    # layer output -> after_mlp_res (final hidden after residuals)
                    def _layer_out_hook(idx):
                        def _hook(_m, _inp, out):
                            t = _first_tensor(out)
                            if idx is not None:
                                self._capture_last_layer_idx = idx
                                if t is not None:
                                    save_tensor(t, f"layer_states_layer_{idx}_after_mlp_res", subdir="layers")

                        return _hook

                    _register(layer, _layer_out_hook(layer_idx))

                return original_block_forward(self, *args, **kwargs)
            finally:
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass

        Qwen3VLTransformerBlock.forward = patched_block_forward

        return True
    except Exception as e:
        print(f"Warning: Could not import mcore Qwen3VL models for patching: {e}")
        return False


# Auto-apply patches when module is imported if enabled
if os.getenv("AUTO_APPLY_MODEL_PATCHES", "0") == "1":
    apply_qwen3vl_patches()
    apply_qwen3vl_megatron_patches()
