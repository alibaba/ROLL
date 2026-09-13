import types
from typing import Optional, List

import torch
from megatron.core import mpu

from ..auto.modeling_auto import register_model
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLGPTModel, Qwen3VLModel
from .config_qwen3_omni import Qwen3OmniMoeConfig


@register_model("qwen3_omni_moe")
class Qwen3OmniMoeModel(Qwen3VLModel):
    config_class = Qwen3OmniMoeConfig

    def __init__(self, config: "Qwen3OmniMoeConfig", **kwargs):
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoderConfig,
            Qwen3OmniMoeAudioEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoder,
            Qwen3OmniMoeAudioEncoder,
            Qwen3OmniMoePreTrainedModelForConditionalGeneration,
            _get_feat_extract_output_lengths,
        )

        Qwen3VLGPTModel.__init__(self, config, **kwargs)

        if mpu.get_pipeline_model_parallel_rank() == 0 and self.vp_stage == 0:
            assert self.decoder.num_layers_per_pipeline_rank >= len(
                config.vision_config.get("deepstack_visual_indexes", [8, 16, 24])
            ), "Current pp and vp not support deepstack"

        if self.pre_process:
            # add audio model to make it can be saved and used in hf
            # although the audio_model weights can be put into template.hf_invalid_keys
            self.audio_model = Qwen3OmniMoeAudioEncoder._from_config(
                Qwen3OmniMoeAudioEncoderConfig(**config.audio_config),
                attn_implementation="sdpa",
                torch_dtype=self.config.params_dtype,
            )
            if not config.init_model_with_meta_device:
                self.audio_model = self.audio_model.to(torch.cuda.current_device())
            for param in self.audio_model.parameters():
                setattr(param, "sequence_parallel", config.sequence_parallel)
            self.vision_model = Qwen3OmniMoeVisionEncoder._from_config(
                Qwen3OmniMoeVisionEncoderConfig(**config.vision_config),
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

        if self.post_process:
            if config.enable_audio_output:
                # not support talker with audio output yet
                from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                    Qwen3OmniMoeTalkerForConditionalGeneration,
                    Qwen3OmniMoeCode2Wav,
                )
                from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                    Qwen3OmniMoeTalkerConfig,
                    Qwen3OmniMoeCode2WavConfig,
                )
                self.talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(
                    Qwen3OmniMoeTalkerConfig(**config.talker_config),
                    torch_dtype=self.config.params_dtype,
                )
                self.code2wav = Qwen3OmniMoeCode2Wav._from_config(
                    Qwen3OmniMoeCode2WavConfig(**config.code2wav_config),
                    torch_dtype=self.config.params_dtype,
                )
                if not config.init_model_with_meta_device:
                    self.talker = self.talker.to(torch.cuda.current_device())
                    self.code2wav = self.code2wav.to(torch.cuda.current_device())

        # construct get_rope_index needed method and attrs
        self.get_rope_index = types.MethodType(
            Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index, self
        )
        self.get_llm_pos_ids_for_vision = types.MethodType(
            Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision, self
        )
        self.spatial_merge_size = self.config.merge_size

        self._get_feat_extract_output_lengths = _get_feat_extract_output_lengths

    def construct_inputs_embeds(
        self,
        input_ids: "torch.LongTensor",
        inputs_embeds: "torch.FloatTensor",
        pixel_values: "torch.Tensor",
        grid_thw: "torch.LongTensor",
        pixel_values_videos: "torch.Tensor",
        video_grid_thw: "torch.LongTensor",
        input_features: "torch.Tensor",
        feature_lens: "torch.Tensor",
        feature_attention_mask: "torch.Tensor",
        input_ranges: List[List[int]],
        image_token_id: int,
        video_token_id: int,
        audio_token_id: int,
    ):
        """
        inputs_embeds: [s, b, h] or [s/tp, b, h] when sequence parallel
        ranges: sequence range
        """
        image_pos_masks = video_pos_masks = deepstack_image_embeds = deepstack_video_embeds = None
        if pixel_values is not None:
            inputs_embeds, image_pos_masks, deepstack_image_embeds = super().construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values,
                grid_thw,
                input_ranges,
                image_token_id,
            )
        if pixel_values_videos is not None:
            inputs_embeds, video_pos_masks, deepstack_video_embeds = super().construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values_videos,
                video_grid_thw,
                input_ranges,
                video_token_id,
            )
        visual_pos_masks, deepstack_visual_embeds = self.merge_deepstack_embeds(
            image_pos_masks, deepstack_image_embeds, video_pos_masks, deepstack_video_embeds
        )

        if input_features is None:
            return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

        # for audio input embeds
        # Follow the same pattern as image/video: build global mask & indices, use
        # build_encoder_inputs / gather_encoder_outputs for load-balanced encoder
        # execution across SP/CP ranks, then masked_scatter into inputs_embeds.

        # (bs, freqs, frames) -> (total_frames, freqs)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()]

        audio_mask = input_ids == audio_token_id
        audio_indices = torch.full_like(audio_mask, -1, dtype=torch.long)
        audio_indices[audio_mask] = torch.arange(audio_mask.sum(), device=audio_indices.device)

        # audio_input_lengths: raw feature frame counts per audio segment
        audio_input_lengths = feature_lens.tolist()
        # audio_output_lengths: embedding token counts per audio segment after encoder
        audio_output_lengths = self._get_feat_extract_output_lengths(feature_lens).tolist()

        split_plan, input_features_split, feature_lens_split, _ = self.build_encoder_inputs(
            audio_input_lengths, input_features, feature_lens, None
        )

        feat_model_dtype = self.audio_model.layers[0].fc1.weight.dtype
        input_features_split = input_features_split.type(feat_model_dtype)
        # convert to (freqs, total_frames) for audio_tower from hf
        input_features_split = input_features_split.permute(1, 0)
        audio_outputs = self.audio_model(input_features_split, feature_lens_split)
        audio_embeds = audio_outputs.last_hidden_state

        audio_embeds = self.gather_encoder_outputs(audio_embeds, split_plan, audio_output_lengths)
        audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        full_sequence_output = input_ranges is None
        if full_sequence_output:
            # Packing mode: inputs_embeds is the full sequence.
            # Use the full audio_mask for masked_scatter.
            all_selected_indices = audio_indices[audio_mask]
            inputs_embeds = inputs_embeds.transpose(0, 1)  # [s, b, h] -> [b, s, h]
            full_selected_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(full_selected_mask, audio_embeds[all_selected_indices])
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        else:
            # Normal mode: slice mask & indices to current rank's input_ranges
            audio_mask = torch.cat([audio_mask[:, start:end] for start, end in input_ranges], dim=1)
            selected_indices = torch.cat([audio_indices[:, start:end] for start, end in input_ranges], dim=1)
            selected_indices = selected_indices[selected_indices != -1]

            inputs_embeds = inputs_embeds.transpose(0, 1)  # [s, b, h] -> [b, s, h]
            selected_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(selected_mask, audio_embeds[selected_indices])
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

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
        input_features=None,
        feature_lens=None,
        feature_attention_mask=None,
        **kwargs,
    ):
        return self.construct_inputs_embeds(
            input_ids,
            inputs_embeds,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            input_features,
            feature_lens,
            feature_attention_mask,
            inputs_ranges,
            self.config.image_token_id,
            self.config.video_token_id,
            self.config.audio_token_id,
        )

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
        use_audio_in_video: Optional[bool] = None,
        video_second_per_grid: Optional[torch.Tensor] = None,
        input_features: Optional["torch.Tensor"] = None,
        feature_attention_mask: Optional["torch.Tensor"] = None,
        **kwargs,
    ) -> "torch.Tensor":
        kwargs.pop("force_vit_image", None)
        kwargs.pop("force_vit_video", None)
        packed_seq_params = kwargs.get("packed_seq_params", None)

        feature_lens = None
        if position_ids is None and input_ids is not None:
            if feature_attention_mask is not None:
                feature_lens = torch.sum(feature_attention_mask, dim=1)
            position_ids, _ = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=torch.ones(input_ids.shape, dtype=input_ids.dtype, device=input_ids.device),
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=feature_lens,
                second_per_grids=video_second_per_grid,
            )

        state = self.prepare_packing_state(
            input_ids, position_ids, attention_mask, packed_seq_params,
        )

        if not self.pre_process or decoder_input is not None:
            return super(Qwen3VLModel, self).forward(
                decoder_input=decoder_input, labels=labels,
                position_ids=state.position_ids,
                **state.cp_batch, **kwargs
            )

        result = self.build_multimodal_embeddings(
            input_ids, attention_mask, packed_seq_params, state,
            multimodal_kwargs=dict(
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                input_features=input_features, feature_lens=feature_lens,
                feature_attention_mask=feature_attention_mask,
            ),
        )

        return super(Qwen3VLModel, self).forward(
            decoder_input=result.inputs_embeds,
            labels=labels,
            position_ids=result.position_ids,
            visual_pos_masks=result.visual_pos_masks,
            deepstack_visual_embeds=result.deepstack_visual_embeds,
            **result.cp_batch,
            **kwargs,
        )
