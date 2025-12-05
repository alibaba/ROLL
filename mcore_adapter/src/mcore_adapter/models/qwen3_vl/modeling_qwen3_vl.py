from typing import List, Optional

import torch
from megatron.core import mpu
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import deprecate_inference_params
from torch import Tensor

from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from ..model_utils import ModuleUtilsMixin
from .config_qwen3_vl import Qwen3VLConfig
from .rope_utils import Qwen3VLMultimodalRotaryEmbedding, get_rope_index
from .transformer_block import Qwen3VLTransformerBlock


class Qwen3VLGPTModel(McaGPTModel):
    def __init__(
        self,
        config: Qwen3VLConfig,
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config,
            rotary_percent=rotary_percent,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            **kwargs,
        )

        # rebuild rope
        self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=self.rotary_base,
        )
        self.mrope_section = self.config.mrope_section
        assert self.mrope_section is not None, (
            "mrope require mrope_section setting, but we got None from TransformerConfig"
        )

        # rebuild the transformer block
        self.decoder = Qwen3VLTransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=self.vp_stage,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )


@register_model("qwen3_vl")
class Qwen3VLModel(Qwen3VLGPTModel, ModuleUtilsMixin):
    config_class = Qwen3VLConfig

    def __init__(self, config: "Qwen3VLConfig", **kwargs):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        super().__init__(config, **kwargs)

        if mpu.get_pipeline_model_parallel_rank() == 0 and self.vp_stage == 0:
            assert self.decoder.num_layers_per_pipeline_rank >= len(
                config.vision_config.get("deepstack_visual_indexes", [8, 16, 24])
            ), "Current pp and vp not support deepstack"

        if self.pre_process:
            self.vision_model = Qwen3VLVisionModel._from_config(
                Qwen3VLVisionConfig(**config.vision_config),
                attn_implementation="sdpa",
                torch_dtype=self.config.params_dtype,
            ).to(torch.cuda.current_device())
            for param in self.vision_model.parameters():
                setattr(param, "sequence_parallel", config.sequence_parallel)

    def _handle_missing_visual(self, inputs_embeds: "torch.FloatTensor"):
        mock_pixel_values = torch.zeros(
            4, self.config.pixel_values_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        mock_grid_thw = torch.LongTensor([[1, 2, 2]]).to(inputs_embeds.device)
        image_embeds, deepstack_image_embeds = self.vision_model(mock_pixel_values, grid_thw=mock_grid_thw)
        inputs_embeds = inputs_embeds + image_embeds.mean() * 0
        return (
            inputs_embeds,
            torch.zeros((inputs_embeds.size(1), inputs_embeds.size(0)), device=inputs_embeds.device, dtype=torch.bool),
            deepstack_image_embeds,
        )

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
        vision_model_dtype = self.vision_model.blocks[0].mlp.linear_fc1.weight.dtype
        required_pixel_values = required_pixel_values.type(vision_model_dtype)
        image_embeds, deepstack_image_embeds = self.vision_model(required_pixel_values, grid_thw=required_grid_thw)
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

        # construct deepstack embedding
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = []
        for deepstack_image_embed in deepstack_image_embeds:
            needed_deepstack_image_embeds = torch.zeros(
                [needed_image_embeds_num] + list(deepstack_image_embed.shape[1:]),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            added_num = 0
            for start, end in valid_image_embeds_nums:
                embeds_num = end - start
                needed_deepstack_image_embeds[added_num : added_num + embeds_num] = deepstack_image_embed[start:end]
                added_num += embeds_num
            assert added_num == needed_image_embeds_num
            deepstack_visual_embeds.append(needed_deepstack_image_embeds)

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

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
            position_ids, _ = get_rope_index(self.config, input_ids, image_grid_thw, video_grid_thw)

        cp_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.config.context_parallel_size > 1:
            cp_batch = {k: v.clone() if v is not None else None for k, v in cp_batch.items()}
            cp_batch = super().get_batch_on_this_cp_rank(cp_batch, dim3_keys=[])

        if not self.pre_process or (pixel_values is None and pixel_values_videos is None) or decoder_input is not None:
            return super().forward(
                decoder_input=decoder_input, labels=labels, position_ids=position_ids, **cp_batch, **kwargs
            )

        inputs_ranges = self.get_input_ranges(input_ids.shape[1])

        inputs_embeds = self.embedding(input_ids=cp_batch["input_ids"], position_ids=None)
        if pixel_values is not None:
            # get deepstack emb
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self.construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values,
                image_grid_thw,
                inputs_ranges,
                self.config.image_token_id,
            )
        elif force_vit_image:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._handle_missing_visual(inputs_embeds)
        if pixel_values_videos is not None:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self.construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values_videos,
                video_grid_thw,
                inputs_ranges,
                self.config.video_token_id,
            )
        elif force_vit_video:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._handle_missing_visual(inputs_embeds)
        return super().forward(
            decoder_input=inputs_embeds,
            labels=labels,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **cp_batch,
            **kwargs,
        )
