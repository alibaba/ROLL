import os
import re
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import BatchFeature, PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy

from roll.utils.logging import get_logger


logger = get_logger()


def collate_fn_to_dict_list(data_list: list[dict]) -> dict:
    """将list[dict]数据转成dict[list]"""
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.cat(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.empty(len(val), dtype=object)
        non_tensors[key][:] = val

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


@dataclass
class DataCollatorWithPaddingForDPO:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def pad_sequences(self, sequences: List[List[int]], pad_value: int = 0) -> torch.Tensor:
        padded = [seq + [pad_value] * (self.max_length - len(seq)) for seq in sequences]
        return torch.tensor(padded)

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        origin_batch_size = len(prompt_id_lens)
        input_ids = torch.stack((chosen_ids, reject_ids), dim=1).view(2 * origin_batch_size, -1)
        att_masks = torch.stack((c_mask, r_mask), dim=1).view(2 * origin_batch_size, -1)
        prompt_id_lens = torch.stack((prompt_id_lens, prompt_id_lens), dim=1).view(2 * origin_batch_size)
        return input_ids, att_masks, prompt_id_lens

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosen_ids = []
        c_mask = []
        reject_ids = []
        r_mask = []
        prompt_ids_lens = []

        for item in batch:
            chosen_ids.append(item["chosen_ids"])
            c_mask.append(item["c_mask"])
            reject_ids.append(item["reject_ids"])
            r_mask.append(item["r_mask"])
            prompt_ids_lens.append(item["prompt_ids_lens"])

        chosen_ids = self.pad_sequences(chosen_ids, pad_value=self.tokenizer.pad_token_id)
        c_mask = self.pad_sequences(c_mask)
        reject_ids = self.pad_sequences(reject_ids, pad_value=self.tokenizer.pad_token_id)
        r_mask = self.pad_sequences(r_mask)
        prompt_ids_lens = torch.tensor(prompt_ids_lens)

        input_ids, attention_mask, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_ids_lens
        )
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "prompt_id_lens": prompt_id_lens, "position_ids": position_ids}


@dataclass
class DataCollatorWithPaddingForPaddedKeys:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_features = [{k: v for k, v in feature.items() if k in self.padded_keys} for feature in features]
        un_padded_features = [{k: v for k, v in feature.items() if k not in self.padded_keys} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["position_ids"] = torch.clip(torch.cumsum(batch["attention_mask"], dim=-1) - 1, min=0, max=None)
        un_padded_batch = collate_fn_to_dict_list(un_padded_features)
        batch.update(un_padded_batch)
        return batch


# from transformers 4.57.0
def is_valid_video_frame(frame):
    # processor should load videos from image paths if frame is a path
    return isinstance(frame, PIL.Image.Image) or (
        (isinstance(frame, np.ndarray) or isinstance(frame, torch.Tensor)) and frame.ndim == 3
    )

def is_valid_video(video):
    if not isinstance(video, (list, tuple)):
        return (isinstance(video, np.ndarray) or isinstance(video, torch.Tensor)) and video.ndim == 4
    return video and all(is_valid_video_frame(frame) for frame in video)

is_valid_image = is_valid_video_frame

def convert_pil_frames_to_video(videos) :
    """
    Given a batch of videos, converts each video to a 4D array. If video is already in array type,
    it is simply returned. We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """

    if not (isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0])):
        return videos

    video_converted = []
    for video in videos:
        video = [np.array(frame) for frame in video]
        video = np.stack(video)
        video_converted.append(video)
    return video_converted


def make_batched_videos(videos):
    # Early exit for deeply nested list of image frame paths. We shouldn't flatten them
    try:
        if isinstance(videos[0][0], list) and isinstance(videos[0][0][0], str):
            return [image_paths for sublist in videos for image_paths in sublist]
    except (IndexError, TypeError):
        pass

    if isinstance(videos, str) or is_valid_video(videos):
        return convert_pil_frames_to_video([videos])
    # only one frame passed, thus we unsqueeze time dim
    elif is_valid_image(videos):
        if isinstance(videos, PIL.Image.Image):
            videos = np.array(videos)
        return [videos[None, ...]]
    elif not isinstance(videos, list):
        raise ValueError(
            f"Invalid video input. Expected either a list of video frames or an input of 4 or 5 dimensions, but got"
            f" type {type(videos)}."
        )

    # Recursively flatten any nested structure
    flat_videos_list = []
    for item in videos:
        if isinstance(item, str) or is_valid_video(item):
            flat_videos_list.append(item)
        elif isinstance(item, list) and item:
            flat_videos_list.extend(make_batched_videos(item))

    flat_videos_list = convert_pil_frames_to_video(flat_videos_list)
    return flat_videos_list


def make_batched_metadata(videos, video_metadata: dict):
    if video_metadata is None:
        video_metadata = [{} for video in videos]

    if isinstance(video_metadata, list):
        # Flatten if nested list
        if isinstance(video_metadata[0], list):
            video_metadata = [dict(**metadata) for metadata_list in video_metadata for metadata in metadata_list]
        # Simply wrap in VideoMetadata if simple dict
        elif isinstance(video_metadata[0], dict):
            video_metadata = [dict(**metadata) for metadata in video_metadata]
    else:
        # Create a batched list from single object, differ with hf's single element list
        video_metadata = [dict(**video_metadata)] * len(videos)
    return video_metadata


def load_videos(videos, video_metas, **kwargs):
    # NOTE: qwen3-vl has different usage with qwen2.5-vl/qwen3-omni due to the new video processor
    # refer to https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#new-qwen-vl-utils-usage
    # qwen2.5-vl/qwen3-omni get sampled fps from kwargs
    # qwen3-vl get raw video fps from video_processor returned video_metadata
    # from qwen_vl_utils import fetch_video
    # use patched version to fetch video
    from roll.utils.qwen_vl_utils import fetch_video

    sampled_videos, sampled_kwargs = [], []
    for video, video_meta in zip(videos, video_metas):
        video = dict(video=video, **kwargs)
        video.update(video_meta)
        video_inputs, sample_fps = (
            fetch_video(
                video,
                image_patch_size=kwargs["image_patch_size"],
                return_video_sample_fps=True,
                return_video_metadata=True,
                video_resize_chunk_frames=kwargs.get("video_resize_chunk_frames", 64),
            )
            if "image_patch_size" in kwargs
            else fetch_video(video, return_video_sample_fps=True, return_video_metadata=True, 
                             video_resize_chunk_frames=kwargs.get("video_resize_chunk_frames", 64),)
        )
        video, video_meta = video_inputs
        sampled_videos.append(video)
        sampled_kwargs.append({"fps": sample_fps, "video_metadata": video_meta})
        if "use_audio_in_video" in video_meta:  # use sample level use_audio_in_video if provided
            sampled_kwargs[-1]["use_audio_in_video"] = video_meta["use_audio_in_video"]
    return sampled_videos, sampled_kwargs


def load_audios(audios, audio_metas, is_video: bool = False, **kwargs):
    # refer to audio_process.py in qwen_omni_utils, maybe use process_audio_info directly
    import av
    import librosa

    def _check_if_video_has_audio(video_path):
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True

    sample_rate = kwargs.get("sample_rate", 16000)
    sampled_audios, sampled_kwargs = [], []
    for audio, audio_meta in zip(audios, audio_metas):
        assert not is_video or _check_if_video_has_audio(audio), (
            "Video must has audio track when use_audio_in_video=True"
        )
        audio_start = audio_meta.get("audio_start", audio_meta.get("video_start", 0.0))
        audio_end = audio_meta.get("audio_end", audio_meta.get("video_end", None))
        sampled_audios.append(
            librosa.load(
                audio,
                sr=sample_rate,
                offset=audio_start,
                duration=(audio_end - audio_start) if audio_end is not None else None,
            )[0]
        )
    return sampled_audios, sampled_kwargs


def load_images(images, image_metas, **kwargs):
    from qwen_vl_utils import fetch_image

    sampled_images, sampled_kwargs = [], []
    for image, image_meta in zip(images, image_metas):
        image = dict(image=image, **kwargs)
        image.update(image_meta)
        sampled_images.append(fetch_image(image, image_patch_size=kwargs["image_patch_size"]))
    return sampled_images, sampled_kwargs


@dataclass
class DataCollatorWithPaddingForMM:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    processor: Optional[ProcessorMixin] = None
    # if key exists in video_kwargs returned by video_load_sample_fn, get value from video_kwargs
    processor_mm_kwargs: Optional[Dict] = field(default_factory=dict)
    extra_data_provider: Optional[callable] = None
    prompt_key: str = "prompt"
    is_template_applied: bool = True
    answer_key: Optional[str] = "ground_truth"
    image_key: Optional[str] = "image"
    # meta keys can include sample level fields such as images with different folders
    image_meta_keys: List[str] = field(default_factory=lambda: [])
    image_load_sample_fn: Optional[callable] = None
    image_sample_kwargs: Optional[Dict] = field(default_factory=dict)
    image_flag_key: Optional[str] = "image_flag"
    video_key: Optional[str] = "video"
    video_meta_keys: List[str] = field(default_factory=lambda: [])
    video_flag_key: Optional[str] = "video_flag"
    # load and sample, return (video, video_kwargs)
    video_load_sample_fn: Optional[callable] = None
    video_sample_kwargs: Optional[Dict] = field(default_factory=dict)
    audio_key: Optional[str] = "audio"
    audio_meta_keys: List[str] = field(default_factory=lambda: [])
    audio_flag_key: Optional[str] = "audio_flag"
    audio_load_sample_fn: Optional[callable] = None
    audio_sample_kwargs: Optional[Dict] = field(default_factory=dict)
    # use_audio_in_video defult to False, and whether or not to use can be
    # overrided by sample level value. When using sample level, `video_load_sample_fn`
    # should return sampled_kwargs including use_audio_in_video
    use_audio_in_video: bool = False
    image_placeholder: Optional[str] = None
    image_token: str = "<|vision_start|><|image_pad|><|vision_end|>"
    image_folder: Optional[str] = None
    video_placeholder: Optional[str] = None
    video_token: str = "<|vision_start|><|video_pad|><|vision_end|>"
    video_folder: Optional[str] = None
    audio_placeholder: Optional[str] = None
    audio_token: str = "<|audio_start|><|audio_pad|><|audio_end|>"
    audio_folder: Optional[str] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    ### keys for outputs of collator
    # padded fields from processor outputs, mainly for llm input fields, maybe rename later
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])
    # unused fields from processor outputs
    processor_unused_keys: List[str] = field(default_factory=lambda: ["prompt", "position_ids", "rope_deltas"])
    # unpaded fields from feature
    extra_unpadded_keys: List[str] = field(default_factory=lambda: [])
    return_tensors: str = "pt"
    return_infer_inputs: bool = True  # whether to include infer engine inputs which differs with train
    return_train_inputs: bool = True  # maybe set to False in multi-turn rollout to reduce overhead
    # use lower precision for multi-modal feature values to reduce store and transfer overhead by ray
    mm_feature_dtype: Optional[str] = None
    mm_feature_names: List[str] = field(
        default_factory=lambda: ["pixel_values", "pixel_values_videos", "input_features"]  # image, video, audio
    )

    def __post_init__(self):
        if self.video_load_sample_fn is None:
            self.video_load_sample_fn = partial(
                load_videos, image_patch_size=self.processor.image_processor.patch_size
            )
            self._default_video_processor_mm_kwargs = {
                # sample and resize have been done in default video_load_sample_fn
                "do_sample_frames": False,  # avoid duplicate operation in processor, qwen3-vl
                "do_resize": False,  # avoid duplicate operation in processor
                # qwen2.5-vl/qwen3-omni use sampled fps as processor kwargs for both hf and vllm,
                # qwen3-vl use raw video fps from video_meta
                "fps": None,
                "video_metadata": None,
            }
        if self.image_load_sample_fn is None:
            self.image_load_sample_fn = partial(
                load_images, image_patch_size=self.processor.image_processor.patch_size
            )
            # avoid duplicate operation in processor, while it might conflict with video do_resize
            # if one does and the other does not
            self._default_image_processor_mm_kwargs = {"do_resize": False}
        if self.audio_load_sample_fn is None:
            self.audio_load_sample_fn = load_audios
            self._default_audio_processor_mm_kwargs = {}
        if self.processor.__class__.__name__.startswith("Qwen3Omni"):
            self.processor_mm_kwargs.update({"use_audio_in_video": self.use_audio_in_video})
            # refer to https://github.com/vllm-project/vllm/issues/26630
            from packaging.version import Version
            from transformers import __version__ as TRANSFORMERS_VERSION

            if Version(TRANSFORMERS_VERSION) < Version("4.58.0") and "truncation" not in self.processor_mm_kwargs:
                self.processor_mm_kwargs["truncation"] = False
        else:
            if "use_audio_in_video" in self.processor_mm_kwargs:
                logger.warning(f"{self.processor.__class__.__name__} not support use_audio_in_video, thus remove it here")
            self.processor_mm_kwargs.pop("use_audio_in_video", None)

        if self.mm_feature_dtype:
            self.mm_feature_dtype = (
                torch.float32
                if self.mm_feature_dtype == "fp32"
                else torch.float16
                if self.mm_feature_dtype == "fp16"
                else torch.bfloat16
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.tokenizer and self.processor
        padded_features = defaultdict(list)
        un_padded_features = defaultdict(list)
        mm_feature_keys = set()
        for feature in features:
            # cannot process as batch directly though processor output as batch
            # since pixel_values would be packed among batch images while DataProto
            # requires all data fields has same batch size
            # if image is None, model_inputs would not include image feature field
            prompt = feature[self.prompt_key]
            if not isinstance(prompt, str):
                prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            elif not self.is_template_applied:
                prompt = [{"role": "user", "content": prompt}]
                prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            valid_image = (
                True
                if self.image_key
                and feature.get(self.image_key, None)
                and (not self.image_flag_key or feature.get(self.image_flag_key, True))
                else False
            )
            valid_video = (
                True
                if self.video_key
                and feature.get(self.video_key, None)
                and (not self.video_flag_key or feature.get(self.video_flag_key, True))
                else False
            )
            valid_audio = (
                True
                if self.audio_key
                and feature.get(self.audio_key, None)
                and (not self.audio_flag_key or feature.get(self.audio_flag_key, True))
                else False
            )
            processor_kwargs = {}
            images = []
            image_kwargs = []
            if valid_image:
                images = (
                    feature[self.image_key] if isinstance(feature[self.image_key], list) else [feature[self.image_key]]
                )
                image_metas = make_batched_metadata(
                    images, dict((key, feature[key]) for key in self.image_meta_keys if key in feature)
                )
                if self.image_placeholder:
                    prompt = prompt.replace(self.image_placeholder, self.image_token)
                if self.image_folder and isinstance(images[0], str):
                    images = [os.path.join(self.image_folder, image) for image in images]
                if self.image_load_sample_fn:
                    images, image_kwargs = self.image_load_sample_fn(images, image_metas, **self.image_sample_kwargs)
                processor_kwargs.update(self._default_image_processor_mm_kwargs)
            audios = []
            audio_kwargs = []
            if valid_audio:
                audios = (
                    feature[self.audio_key] if isinstance(feature[self.audio_key], list) else [feature[self.audio_key]]
                )
                audio_metas = make_batched_metadata(
                    audios, dict((key, feature[key]) for key in self.audio_meta_keys if key in feature)
                )
                if self.audio_placeholder:
                    # use_audio_in_video use video token in text
                    prompt = prompt.replace(self.audio_placeholder, self.audio_token)
                if self.audio_folder and isinstance(audios[0], str):
                    audios = [os.path.join(self.audio_folder, audio) for audio in audios]
                if self.audio_load_sample_fn:
                    audios, audio_kwargs = self.audio_load_sample_fn(audios, audio_metas, **self.audio_sample_kwargs)
                processor_kwargs.update(self._default_audio_processor_mm_kwargs)
            videos = []
            video_kwargs = []
            if valid_video:
                videos = make_batched_videos([feature[self.video_key]])
                video_metas = make_batched_metadata(
                    videos, dict((key, feature[key]) for key in self.video_meta_keys if key in feature)
                )
                if self.video_placeholder:
                    prompt = prompt.replace(self.video_placeholder, self.video_token)
                # video path or video with frame paths
                videos_for_audio = videos
                if isinstance(videos[0], str) or (isinstance(videos[0], list) and isinstance(videos[0][0], str)):
                    if self.video_folder:
                        videos_for_audio = videos = [
                            (
                                os.path.join(self.video_folder, video)
                                if isinstance(video, str)
                                else [os.path.join(self.video_folder, frame) for frame in video]
                            )
                            for video in videos
                        ]
                # processor should load and sample if video_load_sample_fn is not provided
                if self.video_load_sample_fn:
                    # video_kwargs stands for kwargs might be used in processor
                    videos, video_kwargs = self.video_load_sample_fn(
                        videos, video_metas, **self.video_sample_kwargs
                    )
                processor_kwargs.update(self._default_video_processor_mm_kwargs)
            processor_kwargs.update(self.processor_mm_kwargs)
            if image_kwargs:
                processor_kwargs.update(
                    dict(
                        (
                            key,
                            [kwargs[key] for kwargs in image_kwargs]
                            if key in image_kwargs[0]
                            else processor_kwargs[key],
                        )
                        if isinstance(image_kwargs, list) and image_kwargs
                        else (
                            key,
                            image_kwargs[key]
                            if isinstance(image_kwargs, dict) and key in image_kwargs
                            else processor_kwargs[key],
                        )
                        for key in processor_kwargs
                    )
                )
            # video agnostic single value can always be provided in self.processor_mm_kwargs
            # otherwise the value should have same length as videos
            if video_kwargs:
                processor_kwargs.update(
                    dict(
                        (
                            key,
                            [kwargs[key] for kwargs in video_kwargs]
                            if key in video_kwargs[0]
                            else processor_kwargs[key],
                        )
                        if isinstance(video_kwargs, list) and video_kwargs
                        else (
                            key,
                            video_kwargs[key]
                            if isinstance(video_kwargs, dict) and key in video_kwargs
                            else processor_kwargs[key],
                        )
                        for key in processor_kwargs
                    )
                )
            # compatibility for qwen2.5-vl/qwen3-vl/qwen3-omni, subject to change
            # NOTE: qwen3-omni processor only supports single value fps, hack to work around temporarily
            if (
                self.processor.__class__.__name__.startswith("Qwen3Omni")
                and "fps" in processor_kwargs
                and isinstance(processor_kwargs["fps"], list)
            ):
                assert all(fps == processor_kwargs["fps"][0] for fps in processor_kwargs["fps"]), (
                    f"{self.processor.__class__} only support single value fps currently"
                )
                processor_kwargs["fps"] = processor_kwargs["fps"][0]

            # use_audio_in_video should be single value
            if "use_audio_in_video" in processor_kwargs and isinstance(processor_kwargs["use_audio_in_video"], list):
                assert all(
                    use_audio_in_video == processor_kwargs["use_audio_in_video"][0]
                    for use_audio_in_video in processor_kwargs["use_audio_in_video"]
                ), "only support same use_audio_in_video value for videos in one sample"
                processor_kwargs["use_audio_in_video"] = processor_kwargs["use_audio_in_video"][0]
            # sample level use_audio_in_video got from processor_kwargs
            if processor_kwargs.get("use_audio_in_video", self.use_audio_in_video) and valid_video:
                assert self.audio_load_sample_fn and isinstance(videos_for_audio[0], str)
                video_audios, video_audio_kwargs = self.audio_load_sample_fn(
                    videos_for_audio, video_metas, is_video=True, **self.audio_sample_kwargs
                )
                if valid_audio:
                    merged_audios = []
                    merged_audio_kwargs = []
                    # when use_audio_in_video, order of audios and audios from videos make effect and
                    # should be consistent with the order of multi-modal tokens in text, similar with
                    # the logic in qwen3-omni processor
                    # NOTE: thre is still a not supported case: when videos w/ and w/o audios mixed in
                    # the same sample, processor cannot handle it since no mapping between videos and
                    # audios can be passed to processor
                    special_tokens = [
                        re.escape(tok) for tok in [self.processor.audio_token, self.processor.video_token]
                    ]
                    pattern = "|".join(special_tokens)
                    positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, prompt)])
                    audio_index = video_index = 0
                    for _, special_token in positions:
                        if special_token == self.processor.audio_token:
                            merged_audios.append(audios[audio_index])
                            merged_audio_kwargs.append(audio_kwargs[audio_index])
                            audio_index += 1
                        else:
                            merged_audios.append(video_audios[audio_index])
                            merged_audio_kwargs.append(video_audio_kwargs[audio_index])
                            video_index += 1
                    audios, audio_kwargs = merged_audios, merged_audio_kwargs
                else:
                    audios.extend(video_audios)
                    audio_kwargs.extend(video_audio_kwargs)

            if audio_kwargs:
                processor_kwargs.update(
                    dict(
                        (
                            key,
                            [kwargs[key] for kwargs in audio_kwargs]
                            if key in audio_kwargs[0]
                            else processor_kwargs[key],
                        )
                        if isinstance(audio_kwargs, list) and audio_kwargs
                        else (
                            key,
                            audio_kwargs[key]
                            if isinstance(audio_kwargs, dict) and key in audio_kwargs
                            else processor_kwargs[key],
                        )
                        for key in processor_kwargs
                    )
                )

            # IndexError occurs in processor when using empty list
            images = images if images else None
            videos = videos if videos else None
            audios = audios if audios else None

            if self.return_train_inputs:
                # model_inputs are mainly for train engine
                model_inputs: BatchFeature = self.processor(
                    images=images, videos=videos, audio=audios, text=prompt, **processor_kwargs
                )
                if not isinstance(model_inputs, BatchFeature):
                    model_inputs = BatchFeature(data=model_inputs)
                # TODO: maybe use processor produced position_ids
                for key in self.processor_unused_keys:
                    if key in model_inputs:
                        model_inputs.pop(key)
                for key in filter(lambda k: k in model_inputs, self.padded_keys):
                    padded_features[key].append(model_inputs.pop(key)[0])
                # mm feature fileds can be different because of mixed data
                mm_feature_keys = mm_feature_keys.union(model_inputs.keys())
                # to tensors except padded_keys which would be converted after padding
                model_inputs.convert_to_tensors(tensor_type=self.return_tensors)
                if self.mm_feature_dtype:
                    for key in [name for name in model_inputs.keys() if name in self.mm_feature_names]:
                        model_inputs[key] = model_inputs[key].to(self.mm_feature_dtype)
                # allow mixed text and multi-modal data
                # assert model_inputs, "should have multi-modal features"
                # tensors in multi_modal_inputs dict have bsz=1 and should be
                # concat at dim=0 before model forward
                un_padded_features["multi_modal_inputs"].append(dict(model_inputs))

            # inputs for infer engine, not tensors
            # TODO: maybe rename multi_modal_data as multi_modal_infer_inputs
            if self.return_infer_inputs:
                # only support vllm as infer engine currently
                un_padded_features["multi_modal_data"].append(
                    {
                        "prompt_token_ids":  # different with input_ids
                        self.tokenizer.encode(prompt, add_special_tokens=False),
                    }
                )
                multi_modal_data = {}
                if images:
                    multi_modal_data["image"] = images
                if audios:
                    multi_modal_data["audio"] = audios
                if videos:
                    # compatibility for qwen2.5-vl/qwen3-vl/qwen3-omni, subject to change
                    # NOTE: video_mata is used as kwargs in hf while it is put into video for qwen3-vl in vllm==0.11.1,
                    # see: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#offline-inference
                    # hash error occurs for video_metadata when used as mm_kwargs in vllm==0.11.1
                    # avoid vllm version incompatibility for video meta and only use it for qwen3-vl
                    # since video meta only after https://github.com/vllm-project/vllm/pull/19331
                    video_metas = processor_kwargs.pop("video_metadata", video_metas)
                    if self.processor.__class__.__name__.startswith("Qwen3VL") and isinstance(video_metas, list):
                        # vllm gets video num using `n = len(data) if isinstance(data, list) else 1` for mm_uuids
                        # in `_maybe_build_mm_uuids` and gets video and video_meta from tuple in `_get_video_with_metadata`
                        videos = list(zip(videos, video_metas))
                    multi_modal_data["video"] = videos
                if multi_modal_data:
                    un_padded_features["multi_modal_data"][-1]["multi_modal_data"] = multi_modal_data
                    # vllm use mm_processor_kwargs to call processor and select processor output fileds by model defination
                    un_padded_features["multi_modal_data"][-1]["mm_processor_kwargs"] = processor_kwargs

            if self.answer_key:
                un_padded_features[self.answer_key].append(feature[self.answer_key])
            if self.extra_unpadded_keys:
                for key in self.extra_unpadded_keys:
                    un_padded_features[key].append(feature[key])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch.update(un_padded_features)

        # other custom data fields: mainly for specific position_ids currently
        # position_ids for qwen2-vl is optional and make sure it is a 3D tensor
        # shaped with `(3, bs, seq_len)` for 3D-RoPE if provided, while we use
        # `(bs, 3, seq_len)` to put it into DataProto which limits batch size dim
        if self.extra_data_provider:
            fun_params = inspect.signature(self.extra_data_provider).parameters
            kwargs = {}
            for key in fun_params:
                if key in batch:
                    kwargs[key] = batch[key]
                elif key in mm_feature_keys:
                    mm_inputs = [inputs[key] for inputs in batch["multi_modal_inputs"] if key in inputs]
                    kwargs[key] = torch.concat(mm_inputs, dim=0) if mm_inputs else fun_params[key].default
                else:
                    kwargs[key] = fun_params[key].default
            extra_data = self.extra_data_provider(**kwargs)
            batch.update(extra_data)

        # each field should be a tensor or np.array(val=list_data, dtype=object)
        # to be stored in DataProto
        for key in batch:
            if isinstance(batch[key], (torch.Tensor, np.ndarray)):
                assert batch[key].shape[0] == batch["input_ids"].shape[0]
            else:
                assert len(batch[key]) == batch["input_ids"].shape[0]
                val = batch[key]
                batch[key] = np.empty(len(batch[key]), dtype=object)
                batch[key][:] = val
        return batch

@dataclass
class DataCollatorWithPaddingForMMWithLabels(DataCollatorWithPaddingForMM):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorForSFT(DataCollatorWithPaddingForPaddedKeys):
    label_pad_token_id: int = -100
    shift_feature: bool = True

    def __call__(self, features):
        padded_batch = super().__call__(features)
        labels = padded_batch.pop("labels")
        padded_labels = []
        for label in labels:
            seq_len = len(label)
            if seq_len > self.max_length:
                padded_labels.append(label[:self.max_length])
            else:
                padded_labels.append(label + [self.label_pad_token_id] * (self.max_length - seq_len))
        
        padded_batch.update({"labels": torch.tensor(padded_labels, dtype=torch.int64)})

        if self.shift_feature:
            labels = padded_batch.pop("labels")
            labels = labels[:, 1:]
            labels = torch.cat([labels, torch.tensor([self.label_pad_token_id] * labels.shape[0], dtype=torch.int64).reshape(-1, 1)], dim=1)
            padded_batch["labels"] = labels

        return padded_batch
