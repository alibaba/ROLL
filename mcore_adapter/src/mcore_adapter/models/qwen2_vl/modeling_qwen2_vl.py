import heapq
import itertools
from typing import Optional

import torch
from megatron.core import mpu

from ...parallel_functions import encoder_sequence_parallel_gather, encoder_small_batch_size_gather
from ...platforms import current_platform
from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from ..model_utils import ModuleUtilsMixin
from .config_qwen2_vl import Qwen2VLConfig


@register_model("qwen2_vl")
class Qwen2VLModel(McaGPTModel, ModuleUtilsMixin):
    config_class = Qwen2VLConfig

    def __init__(self, config: "Qwen2VLConfig", **kwargs):
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

        super().__init__(config, **kwargs)

        if self.pre_process:
            self.vision_model = Qwen2VisionTransformerPretrainedModel._from_config(
                Qwen2VLVisionConfig(**config.vision_config),
                attn_implementation="sdpa",
                torch_dtype=self.config.params_dtype,
            ).to(current_platform.current_device())
            # TODO: use_reentrant=True might cause error by twice forward/backward when
            # training images and videos simultaneously, https://github.com/pytorch/pytorch/issues/81296
            if config.recompute_granularity == "full" and self.training:
                self.vision_model.gradient_checkpointing_enable({"use_reentrant": False})
            for param in self.vision_model.parameters():
                setattr(param, "sequence_parallel", config.sequence_parallel)

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

        vision_model_dtype = self.vision_model.blocks[0].mlp.down_proj.weight.dtype
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

    # copy from transformers
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # default value 2 from transformers code
        spatial_merge_size = self.config.merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        attention_mask = torch.ones(input_ids.shape, dtype=input_ids.dtype, device=input_ids.device)
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
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
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
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

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
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
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
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
            position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw)

        cp_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.config.context_parallel_size > 1:
            cp_batch = {k: v.clone() if v is not None else None for k, v in cp_batch.items()}
            cp_batch = super().get_batch_on_this_cp_rank(cp_batch, dim3_keys=["attention_mask", "position_ids"])

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
