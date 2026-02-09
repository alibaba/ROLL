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
            Qwen3OmniMoeAudioEncoderConfig,
            Qwen3OmniMoeVisionEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
            Qwen3OmniMoePreTrainedModelForConditionalGeneration,
            Qwen3OmniMoeVisionEncoder,
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
            ).to(torch.cuda.current_device())
            for param in self.audio_model.parameters():
                setattr(param, "sequence_parallel", config.sequence_parallel)
            self.vision_model = Qwen3OmniMoeVisionEncoder._from_config(
                Qwen3OmniMoeVisionEncoderConfig(**config.vision_config),
                attn_implementation="sdpa",
                torch_dtype=self.config.params_dtype,
            ).to(torch.cuda.current_device())
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
                ).to(torch.cuda.current_device())
                self.code2wav = Qwen3OmniMoeCode2Wav._from_config(
                    Qwen3OmniMoeCode2WavConfig(**config.code2wav_config),
                    torch_dtype=self.config.params_dtype,
                ).to(torch.cuda.current_device())

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
        visual_pos_masks, deepstack_visual_embeds = None, None
        # TODO: same as qwen3-vl, only support images or videos since no deepstack_visual_embeds merge process currently
        # maybe merge images and videos first to run vision_model and get deepstack_visual_embeds for images and videos simultaneously
        assert pixel_values is None or pixel_values_videos is None, (
            "inputs with both images and videos are not supported temporarily"
        )
        if pixel_values is not None:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = super().construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values,
                grid_thw,
                input_ranges,
                image_token_id,
            )
        elif pixel_values_videos is not None:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = super().construct_inputs_embeds(
                input_ids,
                inputs_embeds,
                pixel_values_videos,
                video_grid_thw,
                input_ranges,
                video_token_id,
            )

        if input_features is None:
            return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

        # for audio input embeds
        # (bs, freqs, frames) -> (total_frames, freqs)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()]
        # TODO: audio can be treated as chunks of frames with chunk_size for sp/cp actually,
        # chunk_size = 100 * (self.n_window_infer // (self.n_window * 2))
        # temporarily only split audios instead of chunks to simplify which may cause duplicated calculation for same audio
        # maybe scatter chunks to sp/cp group for load balance furthermore
        feat_mask = input_ids == audio_token_id
        feat_culens = feature_lens.cumsum(dim=0, dtype=torch.int32).tolist()  # use list
        feat_embeds_culens = self._get_feat_extract_output_lengths(feature_lens).cumsum(dim=0, dtype=torch.int32)
        required_feat = []  # features to vision tower
        required_feat_lens = []  # feature lengths to vision tower
        valid_feat_embeds_nums = []  # indicate the ranges of needed feature embeds
        added_feat_indexes = []  # feature indexes included in input_ranges
        for i in range(feat_mask.shape[0]):
            for inputs_start, inputs_end in input_ranges:
                # same as qwen-vl, get features included in a sub-range corresponding to each sample
                valid_feat_embeds_start = feat_mask[:i].sum().item()
                valid_feat_embeds_start += feat_mask[i, :inputs_start].sum().item()
                embeds_num = feat_mask[i, inputs_start:inputs_end].sum().item()
                valid_feat_embeds_end = valid_feat_embeds_start + embeds_num
                used_embeds_culen_start = 0  # embeds seqlens before this sub-range
                new_embeds_culen_start = 0  # embeds seqlens new added in this sub-range, new_embeds_seqlen_start >= used_embeds_seqlen_start
                added_culen_before_used = 0  # embeds seqlens in before sub-ranges of input_ranges
                embed_culen_end = feat_embeds_culens[-1]
                for feat_index, feat_embeds_culen in enumerate(feat_embeds_culens):
                    if valid_feat_embeds_start < feat_embeds_culen:  # included in current sub-range
                        if feat_index not in added_feat_indexes:
                            # included in current sub-range and have not been added before, add it
                            required_feat_lens.append(feature_lens[feat_index])
                            # maybe extend together at last, while mapping from embeds length to feature length is not direct
                            required_feat.append(
                                input_features[
                                    (0 if feat_index == 0 else feat_culens[feat_index - 1]) : feat_culens[feat_index]
                                ]
                            )
                            added_feat_indexes.append(feat_index)
                        else:
                            # included in current sub-range but have been added by previous sub-range of this sample, skip it
                            new_embeds_culen_start = feat_embeds_culen
                    else:  # not included in current sub-range
                        used_embeds_culen_start = feat_embeds_culen
                        new_embeds_culen_start = feat_embeds_culen
                        if feat_index in added_feat_indexes:  # included in before sub-ranges of input_ranges
                            before_culen = 0 if feat_index == 0 else feat_embeds_culens[feat_index - 1].item()
                            added_culen_before_used += feat_embeds_culen - before_culen
                    if valid_feat_embeds_end <= feat_embeds_culen:
                        embed_culen_end = feat_embeds_culen
                        break

                # embeds offset in range for this sub-range: offset_in_range = offset_in_start_feat + emb_len_of_pre_subranges
                embeds_needed_start = valid_feat_embeds_start - used_embeds_culen_start + added_culen_before_used
                embeds_needed_end = valid_feat_embeds_end - used_embeds_culen_start + added_culen_before_used
                if embeds_needed_start < embeds_needed_end:
                    valid_feat_embeds_nums.append((embeds_needed_start, embeds_needed_end))

        if len(valid_feat_embeds_nums) == 0:
            # should we use dummy feature input to handle this, _handle_missing_visual is used in qwen-vl
            return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

        required_feat = torch.cat(required_feat, dim=0)
        required_feat_lens = torch.stack(required_feat_lens, dim=0)
        feat_model_dtype = self.audio_model.layers[0].fc1.weight.dtype
        required_feat = required_feat.type(feat_model_dtype)
        # convert to (freqs, total_frames) for input_features to use audio_tower from hf
        required_feat = required_feat.permute(1, 0)
        feat_embeds = self.audio_model(required_feat, required_feat_lens)
        feat_embeds = feat_embeds.last_hidden_state.to(inputs_embeds.device, inputs_embeds.dtype)
        feat_mask = torch.cat(
            [feat_mask[:, inputs_start:inputs_end] for inputs_start, inputs_end in input_ranges], dim=1
        )
        needed_feat_embeds_num = feat_mask.sum().item()
        needed_feat_embeds = torch.zeros(
            [needed_feat_embeds_num] + list(feat_embeds.shape[1:]),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        added_num = 0
        for start, end in valid_feat_embeds_nums:
            embeds_num = end - start
            needed_feat_embeds[added_num : added_num + embeds_num] = feat_embeds[start:end]
            added_num += embeds_num
        assert added_num == needed_feat_embeds_num

        inputs_embeds = inputs_embeds.transpose(0, 1)  # [s, b, h] -> [b, s, h]
        feat_mask = feat_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(feat_mask, needed_feat_embeds)
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

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
        use_audio_in_video: Optional[bool] = None,
        video_second_per_grid: Optional[torch.Tensor] = None,
        input_features: Optional["torch.Tensor"] = None,
        feature_attention_mask: Optional["torch.Tensor"] = None,
        **kwargs,
    ) -> "torch.Tensor":
        force_vit_image = kwargs.pop("force_vit_image", False)
        force_vit_video = kwargs.pop("force_vit_video", False)
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

        cp_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.config.context_parallel_size > 1:
            cp_batch = {k: v.clone() if v is not None else None for k, v in cp_batch.items()}
            cp_batch = super(Qwen3VLModel, self).get_batch_on_this_cp_rank(cp_batch, dim3_keys=[])

        if not self.pre_process or decoder_input is not None:
            return super(Qwen3VLModel, self).forward(
                decoder_input=decoder_input, labels=labels, position_ids=position_ids, **cp_batch, **kwargs
            )

        inputs_ranges = self.get_input_ranges(input_ids.shape[1])

        inputs_embeds = self.embedding(input_ids=cp_batch["input_ids"], position_ids=None)

        if pixel_values is not None or pixel_values_videos is not None:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self.construct_inputs_embeds(
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
        elif force_vit_image or force_vit_video:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._handle_missing_visual(inputs_embeds)

        return super(Qwen3VLModel, self).forward(
            decoder_input=inputs_embeds,
            labels=labels,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **cp_batch,
            **kwargs,
        )
