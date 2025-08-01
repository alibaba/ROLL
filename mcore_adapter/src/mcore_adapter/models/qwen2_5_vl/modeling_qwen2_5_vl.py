from typing import List, Optional, Tuple

import torch
from megatron.core import mpu
from megatron.core.transformer.attention import SelfAttention
from torch import nn

from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from ..model_utils import ModuleUtilsMixin
from .config_qwen2_5_vl import Qwen2_5_VLConfig


# copy from transformers
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# copy from transformer, same as Qwen2VL
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    q: [s, b, head_num, dim]
    k: [s, b, grouped_head_num, dim]
    """
    mrope_section = mrope_section * 2
    cos = (
        torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
        .unsqueeze(unsqueeze_dim)
        .transpose(0, 2)
        .transpose(1, 2)
    )
    sin = (
        torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
        .unsqueeze(unsqueeze_dim)
        .transpose(0, 2)
        .transpose(1, 2)
    )
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# copy from transformers, use default rope
class Qwen2_5_VLRotaryEmbedding(nn.Module): # same as Qwen2_VL
    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        use_cpu_initialization: bool = False,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        device = "cpu" if use_cpu_initialization else torch.cuda.current_device()
        self.inv_freq = 1.0 / (rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb   

class Qwen2_5_VLAttention(SelfAttention): # replace rotary_pos_emb by Qwen2.5VL multimodal_rotary_pos_emb
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs,
    ):
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        assert packed_seq_params is None, "Qwen2_5_VLAttention does not support packed seq."
        query, key = apply_multimodal_rotary_pos_emb(
            query,
            key,
            rotary_pos_emb.cos().to(query.dtype),
            rotary_pos_emb.sin().to(query.dtype),
            mrope_section=self.config.rope_scaling["mrope_section"],
        )
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        output, bias = self.linear_proj(core_attn_out)
        return output, bias


# language model for Qwen2.5VL, replace rotary_pos_emb and attention
class Qwen2_5_VLBaseModel(McaGPTModel):
    config_class = Qwen2_5_VLConfig

    def __init__(self, config: "Qwen2_5_VLConfig", **kwargs):
        super().__init__(config, **kwargs)
        self.rotary_pos_emb = Qwen2_5_VLRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=self.config.rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            rotary_base=self.config.rotary_base,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_params=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
    ):
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = self.decoder.input_tensor
        rotary_pos_emb = self.rotary_pos_emb(decoder_input, position_ids)
        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        if not self.post_process:
            return hidden_states
        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)
        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()
        loss = self.compute_language_model_loss(labels, logits)
        return loss

    def _get_transformer_layer_spec(self, config=None):
        module_spec = super()._get_transformer_layer_spec(config)
        module_spec.submodules.self_attention.module = Qwen2_5_VLAttention
        return module_spec


@register_model("qwen2_5_vl")
class Qwen2_5_VLModel(Qwen2_5_VLBaseModel, ModuleUtilsMixin):
    config_class = Qwen2_5_VLConfig

    def __init__(self, config: "Qwen2_5_VLConfig", **kwargs):
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

        super().__init__(config, **kwargs)
        self.pre_process = kwargs.get("pre_process", mpu.is_pipeline_first_stage())
        if self.pre_process:
            self.vision_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(
                Qwen2_5_VLVisionConfig(**config.vision_config),
                attn_implementation="flash_attention_2",
                torch_dtype=self.config.params_dtype,
            ).to(torch.cuda.current_device())
            for param in self.vision_model.parameters():
                setattr(param, "sequence_parallel", config.sequence_parallel)

    def _handle_missing_visual(self, inputs_embeds: "torch.FloatTensor"):
        mock_pixel_values = torch.zeros(
            4, self.config.pixel_values_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        mock_grid_thw = torch.LongTensor([[1, 2, 2]]).to(inputs_embeds.device)
        image_embeddings = self.vision_model(mock_pixel_values, grid_thw=mock_grid_thw)
        inputs_embeds = inputs_embeds + image_embeddings.mean() * 0
        return inputs_embeds

    def construct_inputs_embeds(
        self,
        input_ids: "torch.LongTensor",
        inputs_embeds: "torch.FloatTensor",
        pixel_values: "torch.Tensor",
        grid_thw: "torch.LongTensor",
        input_ranges: List[List[int]],
        media_token_id: int,
    ):
        """
        inputs_embeds: [s, b, h] or [s/tp, b, h] when sequence parallel
        ranges: sequence range
        """
        image_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        flatten_grid_thw = torch.repeat_interleave(grid_thw, grid_thw[:, 0], dim=0)
        flatten_grid_thw[:, 0] = 1
        image_embeds_seqlens = image_seqlens // (self.config.merge_size**2)
        assert (
            image_seqlens[-1] == pixel_values.shape[0]
        ), f"pixel_values.shape[0] {pixel_values.shape[0]} != image_seqlens[-1] {image_seqlens[-1]}"
        assert (
            sum([r[1] - r[0] for r in input_ranges]) == inputs_embeds.shape[0]
        ), f"sum of input_ranges {input_ranges} not match inputs_embeds.shape {inputs_embeds.shape}"
        image_mask = input_ids == media_token_id

        valid_image_embeds_nums = []  # indicate the ranges of needed image embeds
        required_pixel_values, required_grid_thws = [], []  # image features input to vision tower
        added_image_indexes = []
        for i in range(image_mask.shape[0]):
            for inputs_start, inputs_end in input_ranges:
                valid_image_embeds_start = image_mask[:i].sum().item()
                valid_image_embeds_start += image_mask[i, :inputs_start].sum().item()
                embeds_num = image_mask[i, inputs_start:inputs_end].sum().item()
                valid_image_embeds_end = valid_image_embeds_start + embeds_num
                used_embeds_seqlen_start = 0  # embeds seqlens used in this range
                new_embeds_seqlen_start = (
                    0  # embeds seqlens new added in this range, new_embeds_seqlen_start >= used_embeds_seqlen_start
                )
                embeds_seqlen_end = image_embeds_seqlens[-1]
                added_seqlen_before_used = 0
                for image_index, image_embeds_seqlen in enumerate(image_embeds_seqlens):
                    if valid_image_embeds_start < image_embeds_seqlen:
                        if image_index not in added_image_indexes:
                            required_grid_thws.append(flatten_grid_thw[image_index])
                            added_image_indexes.append(image_index)
                        else:
                            new_embeds_seqlen_start = image_embeds_seqlen
                    else:
                        used_embeds_seqlen_start = image_embeds_seqlen
                        new_embeds_seqlen_start = image_embeds_seqlen
                        if image_index in added_image_indexes:
                            before_seqlen = 0 if image_index == 0 else image_embeds_seqlens[image_index - 1].item()
                            added_seqlen_before_used += image_embeds_seqlen - before_seqlen
                    if valid_image_embeds_end <= image_embeds_seqlen:
                        embeds_seqlen_end = image_embeds_seqlen
                        break

                if new_embeds_seqlen_start < embeds_seqlen_end:
                    required_pixel_values.append(
                        pixel_values[
                            new_embeds_seqlen_start * (self.config.merge_size**2) : embeds_seqlen_end
                            * (self.config.merge_size**2)
                        ]
                    )
                embeds_needed_start = valid_image_embeds_start - used_embeds_seqlen_start + added_seqlen_before_used
                embeds_needed_end = valid_image_embeds_end - used_embeds_seqlen_start + added_seqlen_before_used
                if embeds_needed_start < embeds_needed_end:
                    valid_image_embeds_nums.append((embeds_needed_start, embeds_needed_end))

        if len(required_pixel_values) == 0:
            return self._handle_missing_visual(inputs_embeds)

        required_pixel_values = torch.cat(required_pixel_values, dim=0)
        required_grid_thw = torch.stack(required_grid_thws, dim=0)
        vision_model_dtype = self.vision_model.blocks[0].mlp.down_proj.weight.dtype
        required_pixel_values = required_pixel_values.type(vision_model_dtype)
        image_embeds = self.vision_model(required_pixel_values, grid_thw=required_grid_thw)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        image_mask = torch.cat(
            [image_mask[:, inputs_start:inputs_end] for inputs_start, inputs_end in input_ranges], dim=1
        )
        needed_image_embeds_num = image_mask.sum().item()
        needed_image_embeds = torch.zeros(
            [needed_image_embeds_num] + list(image_embeds.shape[1:]),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        added_num = 0
        for start, end in valid_image_embeds_nums:
            embeds_num = end - start
            needed_image_embeds[added_num : added_num + embeds_num] = image_embeds[start:end]
            added_num += embeds_num
        assert added_num == needed_image_embeds_num

        inputs_embeds = inputs_embeds.transpose(0, 1)  # [s, b, h] -> [b, s, h]
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, needed_image_embeds)
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        return inputs_embeds

    # copy from transformers, add time_tensor
    # TODO: need test video input
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_batch_on_this_cp_rank(self, batch, dim3_keys: List[str] = ["attention_mask"]):
        # VLM need to view all input_ids and media features
        loss_needed_items = {
            "labels": batch.pop("labels", None),
        }
        loss_needed_items = super().get_batch_on_this_cp_rank(loss_needed_items, dim3_keys=dim3_keys)
        batch.update(loss_needed_items)
        return batch

    def get_input_ranges(self, total_seqlen):
        # context parallel 的计算有问题
        slice_rank, slice_size = 0, 1
        if self.config.sequence_parallel:
            slice_rank = mpu.get_tensor_model_parallel_rank()
            slice_size = mpu.get_tensor_model_parallel_world_size()

        def get_sequence_range(start, end, rank, size):
            return start + (end - start) * rank // size, start + (end - start) * (rank + 1) // size

        if self.config.context_parallel_size <= 1:
            return [list(get_sequence_range(0, total_seqlen, slice_rank, slice_size))]
        cp_rank = mpu.get_context_parallel_rank()
        cp_size = mpu.get_context_parallel_world_size()
        left_start = (total_seqlen // cp_size // 2) * cp_rank
        left_end = (total_seqlen // cp_size // 2) * (cp_rank + 1)
        right_start = total_seqlen - left_end
        right_end = total_seqlen - left_start
        slice_len = (left_end - left_start + right_end - right_start) // slice_size
        start = left_start + slice_len * slice_rank
        end = start + slice_len
        if start >= left_end:
            start = start - left_end + right_start
            end = start + slice_len
            return [[start, end]]
        if end <= left_end:
            return [[start, end]]
        end = end - left_end + right_start
        return [[start, left_end], [right_start, end]]

    def forward(
        self,
        input_ids: "torch.Tensor",
        position_ids: Optional["torch.Tensor"] = None,
        attention_mask: Optional["torch.Tensor"] = None,
        decoder_input: Optional["torch.Tensor"] = None,
        labels: Optional["torch.Tensor"] = None,
        pixel_values: Optional["torch.Tensor"] = None,
        pixel_values_videos: Optional["torch.Tensor"] = None,
        image_grid_thw: Optional["torch.LongTensor"] = None,
        video_grid_thw: Optional["torch.LongTensor"] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None, # for videos
        **kwargs,
    ) -> "torch.Tensor":
        if position_ids is None and input_ids is not None:
            position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask)

        cp_batch = {
            "position_ids": position_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.config.context_parallel_size > 1:
            cp_batch = {k: v.clone() if v is not None else None for k, v in cp_batch.items()}
            cp_batch = super().get_batch_on_this_cp_rank(cp_batch, dim3_keys=["attention_mask", "position_ids"])

        if not self.pre_process or (pixel_values is None and pixel_values_videos is None) or decoder_input is not None:
            return super().forward(decoder_input=decoder_input, labels=labels, **cp_batch, **kwargs)

        inputs_ranges = self.get_input_ranges(input_ids.shape[1])

        inputs_embeds = self.embedding(input_ids=cp_batch["input_ids"], position_ids=cp_batch["position_ids"])
        if pixel_values is not None:
            inputs_embeds = self.construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values,
                image_grid_thw,
                inputs_ranges,
                self.config.image_token_id,
            )
        if pixel_values_videos is not None:
            inputs_embeds = self.construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values_videos,
                video_grid_thw,
                inputs_ranges,
                self.config.video_token_id,
            )
        decoder_input = inputs_embeds

        return super().forward(decoder_input=decoder_input, labels=labels, **cp_batch, **kwargs)
