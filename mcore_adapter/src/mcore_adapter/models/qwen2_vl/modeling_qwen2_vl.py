from typing import List, Optional, Tuple

import torch
from megatron.core import mpu

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
        assert image_seqlens[-1] == pixel_values.shape[0], (
            f"pixel_values.shape[0] {pixel_values.shape[0]} != image_seqlens[-1] {image_seqlens[-1]}"
        )
        assert sum([r[1] - r[0] for r in input_ranges]) == inputs_embeds.shape[0], (
            f"sum of input_ranges {input_ranges} not match inputs_embeds.shape {inputs_embeds.shape}"
        )
        image_mask = input_ids == media_token_id

        valid_image_embeds_nums = []  # indicate the ranges of needed image embeds
        required_pixel_values, required_grid_thws = [], []  # image features input to vision tower
        added_image_indexes = []
        for i in range(image_mask.shape[0]):
            for inputs_start, inputs_end in input_ranges:
                valid_image_embeds_start = image_mask[:i].sum().item()
                valid_image_embeds_start += image_mask[i, :inputs_start].sum().item()
                embeds_num = image_mask[i, inputs_start:inputs_end].sum().item()
                if embeds_num == 0:
                    continue
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
        required_pixel_values = required_pixel_values.type(self.vision_model.get_dtype())
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

    # copy from transformers
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
