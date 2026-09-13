import heapq
import itertools
from typing import List, Optional

import torch
from megatron.core import mpu
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import deprecate_inference_params
from torch import Tensor

from ...parallel_functions import encoder_sequence_parallel_gather, encoder_small_batch_size_gather
from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from ..model_utils import ModuleUtilsMixin
from ..sequence_packing_mixin import MultimodalEmbeddingMixin
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

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = preproc_output[:5]

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
class Qwen3VLModel(Qwen3VLGPTModel, ModuleUtilsMixin, MultimodalEmbeddingMixin):
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
            )
            if not config.init_model_with_meta_device:
                self.vision_model = self.vision_model.to(torch.cuda.current_device())
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
        image_outputs = self.vision_model(mock_pixel_values, grid_thw=mock_grid_thw)
        if isinstance(image_outputs, tuple):
            image_embeds, deepstack_image_embeds = image_outputs
        else:
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
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
        media_token_id: int
    ):
        """
        Construct inputs_embeds with vision embeddings replacing media tokens.

        Args:
            input_ids: [b, s] original input_ids (full sequence, used for image_mask)
            inputs_embeds: [s, b, h] or [s/tp/cp, b, h] when sequence/context parallel
            pixel_values: flattened pixel values for vision model
            grid_thw: grid dimensions for each image/video
            input_ranges: sequence ranges for the current rank (CP+SP sliced)
            media_token_id: token id for the media placeholder
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
        image_outputs = self.vision_model(pixel_values, grid_thw=grid_thw)
        if isinstance(image_outputs, tuple):
            image_embeds, deepstack_image_embeds = image_outputs
        else:
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
        image_embeds = self.gather_encoder_outputs(image_embeds, split_plan, image_output_lengths)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        deepstack_image_embeds = [
            self.gather_encoder_outputs(deepstack_image_embed, split_plan, image_output_lengths)
            for deepstack_image_embed in deepstack_image_embeds
        ]

        # If True, inputs_embeds is the full sequence (not CP/SP sliced) and masked_scatter operates on the
        # full sequence. Used in sequence packing mode where CP+packing is done after construct_inputs_embeds.
        full_sequence_output = input_ranges is None
        if full_sequence_output:
            # In packing mode: inputs_embeds is the full sequence (not sliced).
            # Use the full image_mask for masked_scatter so dimensions match.
            all_selected_indices = image_indices[image_mask]
            inputs_embeds = inputs_embeds.transpose(0, 1)  # [s, b, h] -> [b, s, h]
            full_selected_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(full_selected_mask, image_embeds[all_selected_indices])
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            # Return full-sequence image_mask and all deepstack embeds (not
            # sliced). The caller will do CP+packing on visual_pos_masks and
            # then re-select deepstack embeds from the packed mask.
            return inputs_embeds, image_mask, deepstack_image_embeds
        else:
            # Normal mode: inputs_embeds is already CP+SP sliced.
            # Slice image_mask and indices by input_ranges.
            sliced_image_mask = torch.cat([image_mask[:, start:end] for start, end in input_ranges], dim=1)
            sliced_indices = torch.cat([image_indices[:, start:end] for start, end in input_ranges], dim=1)
            selected_indices = sliced_indices[sliced_indices != -1]
            for i, deepstack_image_embed in enumerate(deepstack_image_embeds):
                deepstack_image_embeds[i] = deepstack_image_embed[selected_indices]
            inputs_embeds = inputs_embeds.transpose(0, 1)  # [s, b, h] -> [b, s, h]
            selected_mask = sliced_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(selected_mask, image_embeds[selected_indices])
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            return inputs_embeds, sliced_image_mask, deepstack_image_embeds

    def build_encoder_inputs(
        self,
        input_lengths: List[int],
        input_features: torch.Tensor,
        input_position_infos: torch.LongTensor,
        input_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        calculate split plan and local data according to workload, assuming workload proportional to length
        Args:
            input_lengths (List[int]): length of each sample
            input_features (torch.Tensor): flatted input features, input_features.shape[0] == sum(input_lengths)
            input_position_infos (torch.LongTensor): additional position info, len(input_position_infos) == len(input_lengths)
        """
        world_size = mpu.get_tensor_and_context_parallel_world_size()

        if world_size == 1 or len(input_lengths) < world_size:  # encoder has small batch size
            # TODO: support encoder small batch size
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
        split_plan: Optional[List[List[int]]] = None,
        output_lengths: Optional[List[int]] = None,
    ):
        if split_plan is not None:
            return encoder_sequence_parallel_gather(output_features, split_plan, output_lengths)
        return encoder_small_batch_size_gather(output_features)

    def get_batch_on_this_cp_rank(self, batch, dim3_keys: List[str] = ["attention_mask"]):
        # VLM forward() handles input_ids and attention_mask splitting internally
        skipped = {}
        for key in ("input_ids", "attention_mask"):
            if key in batch:
                skipped[key] = batch.pop(key)
        batch = super().get_batch_on_this_cp_rank(batch, dim3_keys=dim3_keys)
        batch.update(skipped)
        return batch


    def merge_deepstack_embeds(self, image_mask, deepstack_image_embeds, video_mask, deepstack_video_embeds):
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds
        return visual_pos_masks, deepstack_visual_embeds

    def construct_multimodal_inputs(
        self,
        input_ids,
        inputs_embeds,
        inputs_ranges,
        *,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        force_vit_image=False,
        force_vit_video=False,
        **kwargs,
    ):
        image_pos_masks = video_pos_masks = deepstack_image_embeds = deepstack_video_embeds = None
        if pixel_values is not None:
            inputs_embeds, image_pos_masks, deepstack_image_embeds = self.construct_inputs_embeds(
                input_ids, inputs_embeds, pixel_values, image_grid_thw,
                inputs_ranges, self.config.image_token_id
            )
        elif force_vit_image:
            inputs_embeds, image_pos_masks, deepstack_image_embeds = self._handle_missing_visual(inputs_embeds)
        if pixel_values_videos is not None:
            inputs_embeds, video_pos_masks, deepstack_video_embeds = self.construct_inputs_embeds(
                input_ids, inputs_embeds, pixel_values_videos, video_grid_thw,
                inputs_ranges, self.config.video_token_id
            )
        elif force_vit_video:
            inputs_embeds, video_pos_masks, deepstack_video_embeds = self._handle_missing_visual(inputs_embeds)
        visual_pos_masks, deepstack_visual_embeds = self.merge_deepstack_embeds(
            image_pos_masks, deepstack_image_embeds, video_pos_masks, deepstack_video_embeds
        )
        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

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

        packed_seq_params = kwargs.get("packed_seq_params", None)

        state = self.prepare_packing_state(
            input_ids, position_ids, attention_mask, packed_seq_params,
        )

        if not self.pre_process or decoder_input is not None:
            return super().forward(
                decoder_input=decoder_input, labels=labels,
                position_ids=state.position_ids,
                **state.cp_batch, **kwargs
            )

        result = self.build_multimodal_embeddings(
            input_ids, attention_mask, packed_seq_params, state,
            multimodal_kwargs=dict(
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                force_vit_image=force_vit_image, force_vit_video=force_vit_video,
            ),
        )

        return super().forward(
            decoder_input=result.inputs_embeds,
            labels=labels,
            position_ids=result.position_ids,
            visual_pos_masks=result.visual_pos_masks,
            deepstack_visual_embeds=result.deepstack_visual_embeds,
            **result.cp_batch,
            **kwargs,
        )
