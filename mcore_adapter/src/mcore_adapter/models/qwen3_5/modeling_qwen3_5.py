import heapq
import itertools
from typing import Optional

import torch
from megatron.core import mpu

from ...parallel_functions import encoder_sequence_parallel_gather, encoder_small_batch_size_gather
from ...platforms import current_platform
from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from ..qwen3_vl.rope_utils import Qwen3VLMultimodalRotaryEmbedding, get_rope_index
from .config_qwen3_5 import Qwen3_5Config


class Qwen3_5McaGPTModel(McaGPTModel):
    def __init__(
        self,
        config: Qwen3_5Config,
        seq_len_interpolation_factor: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            **kwargs,
        )

        # rebuild rope
        self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=self.config.rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=self.config.rotary_base,
        )
        self.mrope_section = self.config.mrope_section
        assert self.mrope_section is not None, (
            "mrope require mrope_section setting, but we got None from TransformerConfig"
        )


@register_model("qwen3_5")
class Qwen3_5Model(Qwen3_5McaGPTModel):
    config_class = Qwen3_5Config

    def __init__(self, config: "Qwen3_5Config", **kwargs):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        super().__init__(config, **kwargs)

        if self.pre_process:
            self.vision_model = Qwen3_5VisionModel._from_config(
                Qwen3_5VisionConfig(**config.vision_config),
                attn_implementation="sdpa",
                torch_dtype=self.config.params_dtype,
            ).to(current_platform.current_device())
            # TODO: use_reentrant=True might cause error by twice forward/backward when
            # training images and videos simultaneously, https://github.com/pytorch/pytorch/issues/81296
            if config.recompute_granularity == "full" and self.training:
                self.vision_model.gradient_checkpointing_enable({"use_reentrant": False})
            for param in self.vision_model.parameters():
                setattr(param, "sequence_parallel", config.sequence_parallel)

    def _get_transformer_layer_spec(self, config: Optional[Qwen3_5Config] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )

        config = config or self.config
        assert config.transformer_impl == "transformer_engine", (
            "Qwen3_5Model only supports 'transformer_engine' implementation"
        )
        if config.experimental_attention_variant is not None:
            transformer_block_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config=config, vp_stage=self.vp_stage
            )
        else:
            transformer_block_spec = super()._get_transformer_layer_spec(config)
        return transformer_block_spec

    def _handle_missing_visual(self, inputs_embeds: "torch.FloatTensor"):
        mock_pixel_values = torch.zeros(
            4, self.config.pixel_values_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        mock_grid_thw = torch.LongTensor([[1, 2, 2]]).to(inputs_embeds.device)
        image_embeddings = self.vision_model(mock_pixel_values, grid_thw=mock_grid_thw)
        if not isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.pooler_output
        inputs_embeds = inputs_embeds + image_embeddings.mean() * 0
        return inputs_embeds

    def construct_inputs_embeds(
        self,
        input_ids: "torch.LongTensor",
        inputs_embeds: "torch.FloatTensor",
        pixel_values: "torch.Tensor",
        grid_thw: "torch.LongTensor",
        input_ranges: list[list[int]],
        media_token_id: int,
    ):
        """
        inputs_embeds: [s, b, h] or [s/tp, b, h] when sequence parallel
        ranges: sequence range
        """
        image_mask = input_ids == media_token_id
        image_indices = torch.full_like(image_mask, -1, dtype=torch.long)
        image_indices[image_mask] = torch.arange(image_mask.sum(), device=image_indices.device)
        vision_token_compress = self.config.merge_size**2

        image_input_lengths = grid_thw.prod(-1).tolist()
        image_output_lengths = [_ // vision_token_compress for _ in image_input_lengths]

        split_plan, pixel_values, grid_thw, _ = self.build_encoder_inputs(
            image_input_lengths, pixel_values, grid_thw, None
        )

        vision_model_dtype = self.vision_model.blocks[0].mlp.linear_fc1.weight.dtype
        pixel_values = pixel_values.type(vision_model_dtype)
        image_embeds = self.vision_model(pixel_values, grid_thw=grid_thw)
        if not isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.pooler_output
        image_embeds = self.gather_encoder_outputs(image_embeds, split_plan, image_output_lengths)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        selected_mask = torch.cat([image_mask[:, start:end] for start, end in input_ranges], dim=1)
        selected_indices = torch.cat([image_indices[:, start:end] for start, end in input_ranges], dim=1)
        selected_indices = selected_indices[selected_indices != -1]

        inputs_embeds = inputs_embeds.transpose(0, 1)  # [s, b, h] -> [b, s, h]
        selected_mask = selected_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(selected_mask, image_embeds[selected_indices])
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        return inputs_embeds

    def build_encoder_inputs(
        self,
        input_lengths: list[int],
        input_features: torch.Tensor,
        input_position_infos: torch.LongTensor,
        input_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        calculate split plan and local data according to workload, assuming workload proportional to length
        Args:
            input_lengths (list[int]): length of each sample
            input_features (torch.Tensor): flatted input features, input_features.shape[0] == sum(input_lengths)
            input_position_infos (torch.LongTensor): additional position info, len(input_position_infos) == len(input_lengths)
        """
        world_size = mpu.get_tensor_and_context_parallel_world_size()

        if world_size == 1 or len(input_lengths) < world_size:  # encoder has small batch size
            return None, input_features, input_position_infos, input_attention_mask

        # sorted by length
        indexed_items = sorted([(length, i) for i, length in enumerate(input_lengths)], reverse=True)

        # min_heap for tracking current load on each GPU
        min_heap = [(0, i) for i in range(world_size)]

        # (length, original_index)
        split_plan = [[] for _ in range(world_size)]

        # heap sort
        for length, original_index in indexed_items:
            current_load, rank = heapq.heappop(min_heap)
            split_plan[rank].append((length, original_index))
            new_load = current_load + length
            heapq.heappush(min_heap, (new_load, rank))

        # start indices for each sample in input_features
        start_indices = [
            0,
        ] + list(itertools.accumulate(input_lengths[:-1]))
        # local inputs for each rank
        local_rank = mpu.get_tensor_and_context_parallel_rank()

        local_features_slices = []
        local_position_infos_slices = []
        local_attention_mask_slices = None
        if input_attention_mask is not None:
            if len(input_attention_mask) != len(input_position_infos):
                raise ValueError("input_attention_mask and input_position_infos must have the same length.")
            local_attention_mask_slices = []

        for length, source_index in split_plan[local_rank]:
            start, end = start_indices[source_index], start_indices[source_index] + length
            local_features_slices.append(input_features[start:end])
            start, end = source_index, source_index + 1
            local_position_infos_slices.append(input_position_infos[start:end])
            if local_attention_mask_slices is not None:
                local_attention_mask_slices.append(input_attention_mask[start:end])

        # no workload on current GPU
        if not local_features_slices:
            raise ValueError("No workload assigned to the current GPU in encoder.")

        input_features_split = torch.cat(local_features_slices, dim=0)
        input_position_infos_split = torch.cat(local_position_infos_slices, dim=0)

        input_attention_mask_split = None
        if local_attention_mask_slices is not None:
            input_attention_mask_split = torch.cat(local_attention_mask_slices, dim=0)

        return split_plan, input_features_split, input_position_infos_split, input_attention_mask_split

    def gather_encoder_outputs(
        self,
        output_features: torch.Tensor,
        split_plan: Optional[list[list[int]]] = None,
        output_lengths: Optional[list[int]] = None,
    ):
        if split_plan is not None:
            return encoder_sequence_parallel_gather(output_features, split_plan, output_lengths)
        return encoder_small_batch_size_gather(output_features)

    def get_batch_on_this_cp_rank(self, batch, dim3_keys: list[str] = ["attention_mask"]):
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
        **kwargs,
    ) -> "torch.Tensor":
        force_vit_image = kwargs.pop("force_vit_image", False)
        force_vit_video = kwargs.pop("force_vit_video", False)
        if position_ids is None and input_ids is not None:
            position_ids, _ = get_rope_index(self.config, input_ids, image_grid_thw, video_grid_thw)

        cp_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.config.context_parallel_size > 1:
            cp_batch = {k: v.clone() if v is not None else None for k, v in cp_batch.items()}
            cp_batch = super().get_batch_on_this_cp_rank(cp_batch, dim3_keys=[])

        if not self.pre_process or decoder_input is not None:
            return super().forward(
                decoder_input=decoder_input, labels=labels, position_ids=position_ids, **cp_batch, **kwargs
            )

        inputs_ranges = self.get_input_ranges(input_ids.shape[1])

        inputs_embeds = self.embedding(input_ids=cp_batch["input_ids"], position_ids=None)
        if pixel_values is not None:
            inputs_embeds = self.construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values,
                image_grid_thw,
                inputs_ranges,
                self.config.image_token_id,
            )
        elif force_vit_image:
            inputs_embeds = self._handle_missing_visual(inputs_embeds)
        if pixel_values_videos is not None:
            inputs_embeds = self.construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values_videos,
                video_grid_thw,
                inputs_ranges,
                self.config.video_token_id,
            )
        elif force_vit_video:
            inputs_embeds = self._handle_missing_visual(inputs_embeds)
        decoder_input = inputs_embeds

        return super().forward(
            decoder_input=decoder_input, labels=labels, position_ids=position_ids, **cp_batch, **kwargs
        )
