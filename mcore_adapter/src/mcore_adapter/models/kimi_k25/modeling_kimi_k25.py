import heapq
import itertools
from typing import Optional

import torch
from megatron.core import mpu
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ...parallel_functions import encoder_sequence_parallel_gather, encoder_small_batch_size_gather
from ...platforms import current_platform
from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from .config_kimi_k25 import KimiK2_5Config


@register_model("kimi_k25")
class KimiK2_5Model(McaGPTModel):
    config_class = KimiK2_5Config

    def __init__(self, config: KimiK2_5Config, **kwargs):
        super().__init__(config, **kwargs)

        vt_config_class = get_class_from_dynamic_module(
            "modeling_kimi_k25.VisionTowerConfig", self.config.name_or_path
        )
        self.vt_config = vt_config_class(config.hf_vision_config, **kwargs)

        if self.pre_process:
            vision_tower_class = get_class_from_dynamic_module(
                "modeling_kimi_k25.MoonViT3dPretrainedModel", self.config.name_or_path
            )
            self.vision_tower = vision_tower_class(self.vt_config)
            if not config.init_model_with_meta_device:
                self.vision_tower = self.vision_tower.to(device=current_platform.current_device(), dtype=config.params_dtype)
            if config.recompute_granularity == "full" and self.training:
                if hasattr(self.vision_tower, "supports_gradient_checkpointing") and self.vision_tower.supports_gradient_checkpointing:
                    self.vision_tower.gradient_checkpointing_enable({"use_reentrant": False})
            for param in self.vision_tower.parameters():
                setattr(param, "sequence_parallel", config.sequence_parallel)

            self.mm_projector = self._get_mm_projector()
            if not config.init_model_with_meta_device:
                self.mm_projector = self.mm_projector.to(device=current_platform.current_device(), dtype=config.params_dtype)
            for param in self.mm_projector.parameters():
                setattr(param, "sequence_parallel", config.sequence_parallel)

    def _get_mm_projector(self):
        proj_config_class = get_class_from_dynamic_module(
            "modeling_kimi_k25.ProjectorConfig", self.config.name_or_path
        )
        IdentityMap = get_class_from_dynamic_module(
            "modeling_kimi_k25.IdentityMap", self.config.name_or_path
        )
        MLP = get_class_from_dynamic_module(
            "modeling_kimi_k25.MLP", self.config.name_or_path
        )
        PatchMergerMLP = get_class_from_dynamic_module(
            "modeling_kimi_k25.PatchMergerMLP", self.config.name_or_path
        )

        proj_config = proj_config_class(self.config.hf_vision_config)
        if proj_config.mm_projector_type == "identity":
            return IdentityMap()
        elif proj_config.mm_projector_type == "mlp":
            return MLP(proj_config)
        elif proj_config.mm_projector_type == "patchmerger":
            return PatchMergerMLP(proj_config)
        else:
            raise ValueError(f"Unsupported mm_projector_type: {proj_config.mm_projector_type}")

    def _get_vision_dtype(self) -> torch.dtype:
        """Infer the dtype for the vision tower, defaulting to bf16 for flash attention compatibility."""
        weight_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        if weight_dtype in (torch.float16, torch.bfloat16):
            return weight_dtype
        return torch.bfloat16

    def _handle_missing_visual(self, inputs_embeds: torch.FloatTensor):
        """Run a dummy forward through vision_tower + mm_projector to keep gradients alive."""
        vision_dtype = self._get_vision_dtype()
        mock_pixel_values = torch.zeros(
            4, 3, 14, 14, device=inputs_embeds.device, dtype=vision_dtype
        )
        mock_grid_thws = torch.LongTensor([[1, 2, 2]]).to(inputs_embeds.device)
        image_features = self.vision_tower(mock_pixel_values, mock_grid_thws)
        image_features = self.mm_projector(image_features)
        dummy = sum(feat.mean() for feat in image_features) * 0
        inputs_embeds = inputs_embeds + dummy
        return inputs_embeds

    def construct_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        pixel_values: torch.Tensor,
        grid_thws: torch.LongTensor,
        input_ranges: list[list[int]],
    ):
        """Merge vision features into text embeddings.

        Args:
            input_ids: [batch_size, seq_len] full (un-sliced) input ids.
            inputs_embeds: [s, b, h] or [s/tp, b, h] when sequence parallel.
            pixel_values: flattened pixel values for the vision encoder.
            grid_thws: [num_images, 3] (temporal, height, width) per image.
            input_ranges: sequence ranges on the current rank.
        """
        media_token_id = self.config.media_placeholder_token_id
        vision_token_compress = self.config.vision_token_compress

        image_mask = input_ids == media_token_id
        image_indices = torch.full_like(image_mask, -1, dtype=torch.long)
        image_indices[image_mask] = torch.arange(image_mask.sum(), device=image_indices.device)

        # sd2_tpool: spatial 2x downsample + temporal pooling
        # input_length = t * h * w; output_length = (h * w) / vision_token_compress
        image_input_lengths = grid_thws.prod(-1).tolist()
        image_output_lengths = [
            (int(thw[1]) * int(thw[2])) // vision_token_compress for thw in grid_thws
        ]

        split_plan, pixel_values, grid_thws, _ = self.build_encoder_inputs(
            image_input_lengths, pixel_values, grid_thws, None
        )

        vision_dtype = self._get_vision_dtype()
        pixel_values = pixel_values.to(vision_dtype)
        image_features = self.vision_tower(pixel_values, grid_thws)
        image_features = self.mm_projector(image_features)

        image_embeds = torch.cat(image_features, dim=0)
        image_embeds = self.gather_encoder_outputs(image_embeds, split_plan, image_output_lengths)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        selected_mask = torch.cat(
            [image_mask[:, start:end] for start, end in input_ranges], dim=1
        )
        selected_indices = torch.cat(
            [image_indices[:, start:end] for start, end in input_ranges], dim=1
        )
        selected_indices = selected_indices[selected_indices != -1]

        inputs_embeds = inputs_embeds.transpose(0, 1)
        selected_mask_expanded = selected_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(selected_mask_expanded, image_embeds[selected_indices])
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        return inputs_embeds

    def build_encoder_inputs(
        self,
        input_lengths: list[int],
        input_features: torch.Tensor,
        input_position_infos: torch.LongTensor,
        input_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Split encoder inputs across tensor/context parallel ranks for load balancing."""
        world_size = mpu.get_tensor_and_context_parallel_world_size()

        if world_size == 1 or len(input_lengths) < world_size:
            return None, input_features, input_position_infos, input_attention_mask

        indexed_items = sorted(
            [(length, i) for i, length in enumerate(input_lengths)], reverse=True
        )

        min_heap = [(0, i) for i in range(world_size)]
        split_plan: list[list[tuple[int, int]]] = [[] for _ in range(world_size)]

        for length, original_index in indexed_items:
            current_load, rank = heapq.heappop(min_heap)
            split_plan[rank].append((length, original_index))
            heapq.heappush(min_heap, (current_load + length, rank))

        start_indices = [0] + list(itertools.accumulate(input_lengths[:-1]))
        local_rank = mpu.get_tensor_and_context_parallel_rank()

        local_features_slices = []
        local_position_infos_slices = []
        local_attention_mask_slices = None
        if input_attention_mask is not None:
            if len(input_attention_mask) != len(input_position_infos):
                raise ValueError(
                    "input_attention_mask and input_position_infos must have the same length."
                )
            local_attention_mask_slices = []

        for length, source_index in split_plan[local_rank]:
            start = start_indices[source_index]
            local_features_slices.append(input_features[start : start + length])
            local_position_infos_slices.append(input_position_infos[source_index : source_index + 1])
            if local_attention_mask_slices is not None:
                local_attention_mask_slices.append(input_attention_mask[source_index : source_index + 1])

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
        """Gather encoder outputs back from parallel ranks."""
        if split_plan is not None:
            return encoder_sequence_parallel_gather(output_features, split_plan, output_lengths)
        return encoder_small_batch_size_gather(output_features)

    def get_batch_on_this_cp_rank(self, batch, dim3_keys: list[str] = ["attention_mask"]):
        """VLM needs to see all input_ids and media features; only split labels for CP."""
        loss_needed_items = {
            "labels": batch.pop("labels", None),
        }
        loss_needed_items = super().get_batch_on_this_cp_rank(loss_needed_items, dim3_keys=dim3_keys)
        batch.update(loss_needed_items)
        return batch

    def get_input_ranges(self, total_seqlen: int) -> list[list[int]]:
        """Compute the local sequence range(s) for the current SP/CP rank."""
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
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thws: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for KimiK2_5 multimodal model.

        When pixel_values is provided (pre_process stage), extracts vision features
        via vision_tower + mm_projector and merges them into text embeddings before
        passing to the LLM decoder.
        """
        cp_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.config.context_parallel_size > 1:
            cp_batch = {k: v.clone() if v is not None else None for k, v in cp_batch.items()}
            cp_batch = super().get_batch_on_this_cp_rank(cp_batch, dim3_keys=[])

        if not self.pre_process or pixel_values is None or decoder_input is not None:
            return super().forward(
                decoder_input=decoder_input,
                labels=labels,
                position_ids=position_ids,
                **cp_batch,
                **kwargs,
            )

        input_ranges = self.get_input_ranges(input_ids.shape[1])

        inputs_embeds = self.embedding(input_ids=cp_batch["input_ids"], position_ids=None)

        if pixel_values is not None and len(pixel_values) > 0:
            inputs_embeds = self.construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values,
                grid_thws,
                input_ranges,
            )
        else:
            inputs_embeds = self._handle_missing_visual(inputs_embeds)

        return super().forward(
            decoder_input=inputs_embeds,
            labels=labels,
            position_ids=position_ids,
            **cp_batch,
            **kwargs,
        )
