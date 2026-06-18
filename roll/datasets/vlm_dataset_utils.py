import os
from io import BytesIO
from typing import List, Tuple, Union

import datasets
import PIL.Image as Image
from datasets import load_from_disk
from transformers import ProcessorMixin
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_utils import load_images
from roll.datasets.dataset import get_dataset
from roll.utils.logging import get_logger
from roll.utils.import_utils import safe_import_class


logger = get_logger()


def create_pipeline_data_kwargs(data_args, tokenizer, processor, is_val=False):
    data_kwargs_getter = getattr(data_args, "custom_data_kwargs_func")
    if data_kwargs_getter is None:
        data_kwargs_getter = get_vlm_data_kwargs
    elif isinstance(data_kwargs_getter, str):
        data_kwargs_getter = safe_import_class(data_kwargs_getter)
    return data_kwargs_getter(data_args, tokenizer, processor, is_val=is_val)


def format_prompt(prompt, processor, use_image=True, prompt_image_token=None):
    question_template = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_template.format(Question=prompt)},
                ]
                if use_image and not prompt_image_token
                else [
                    {"type": "text", "text": question_template.format(Question=prompt)}
                ],  # image_token has been included in prompt
            }
        ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_image_token:
        text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text


def process_image(image: Image.Image, processor: ProcessorMixin):
    # same as qwen2-vl image processor
    image_processor = processor.image_processor
    factor = (
        image_processor.patch_size * image_processor.merge_size
        if "Qwen" in image_processor.image_processor_type
        else 28
    )
    height, width = image.height, image.width
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        # Qwen2VLImageProcessorFast uses size["shortest_edge"]/size["longest_edge"] instead of min_pixels/max_pixels
        # thus set min_pixels/max_pixels attrs before using min_pixels/max_pixels
        min_pixels=image_processor.min_pixels,
        max_pixels=image_processor.max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height), resample=image_processor.resample)
    return resized_image


def process_images(
    images: Union[List, Tuple, str, Image.Image], processor: ProcessorMixin
) -> Union[Image.Image, List[Image.Image], List[List[Image.Image]]]:
    """Process images, handling different levels of nesting.

    Args:
      images: A single image, a list of images, or a list of lists of images to load.
      timeout: Timeout for loading images.

    Returns:
      A single image, a list of images, a list of lists of images.
    """
    if isinstance(images, (list, tuple)):
        if len(images) and isinstance(images[0], (list, tuple)):
            return [[process_image(image, processor=processor) for image in image_group] for image_group in images]
        else:
            return [process_image(image, processor=processor) for image in images]
    else:
        return process_image(images, processor=processor)


def encode_function(
    data, processor, prompt_getter, ground_truth_getter, image_getter, tag_getter, prompt_image_token=None
):
    image_flag = [True] * len(prompt_getter(data))
    image_list = []
    for idx, image in enumerate(image_getter(data)):
        if not image:
            image_flag[idx] = False
        try:
            if isinstance(image, bytes):  # bytes data
                # TODO: support multiple images
                image_out = Image.open(BytesIO(image))
            else:
                image_out = load_images(image if isinstance(image, (list, tuple)) else [image], timeout=None)
        except Exception as e:
            if isinstance(image, bytes):
                image_out = [Image.new("RGB", (224, 224), (255, 255, 255))]
                logger.error(f"Failed to get image with type: {type(image)}")
            else:
                image_out = [Image.new("RGB", (224, 224), (255, 255, 255))] * len(image)
                logger.error(f"Failed to get image: {image}")
        # since infer-image use pil image as input while train-engine use
        # processed data, process image here to make them use same image
        # refer to the following for Spatial Understanding with Qwen2.5-VL
        # https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb
        # NOTE: process_image from qwen2.5-vl keeps aspect ratio almostly and
        # bboxes would be normalized in detection verifier, thus nearly no need
        # to change ground-truth bboxes
        # process in collator, no need to process here
        # image_out = process_images(image_out, processor)
        image_list.append(image_out)
    text_list = []
    for idx, instruct in enumerate(prompt_getter(data)):
        # provide prompt_image_token if image_token in prompt
        text = format_prompt(instruct, processor, use_image=image_flag[idx], prompt_image_token=prompt_image_token)
        text_list.append(text)
    encodings = {
        "tag": tag_getter(data),
        "images": image_list,
        "prompt": text_list,
        "ground_truth": ground_truth_getter(data),
        "reward_model": data["reward_model"],
        # for text and multi-modal mixed data usage, indicating valid image
        "image_flag": image_flag,
    }
    return encodings


def get_vlm_dataset(data_args, encode_function, processor, get_eval=False):
    cache_path = getattr(data_args, "cache_path", None)
    if cache_path:
        cache_path = os.path.join(cache_path, "val" if get_eval else "train")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset

    dataset = get_dataset(data_args=data_args)
    # regularized data filed
    features = datasets.Features(
        {
            "tag": datasets.Value(dtype="string"),  # from data_source
            "images": datasets.Sequence(feature=datasets.Image(mode=None, decode=True)),
            "prompt": datasets.Value(dtype="string"),
            "ground_truth": datasets.Value(dtype="string"),
            "reward_model": dataset.features["reward_model"],
            # for text and multi-modal mixed data usage, indicating valid image
            "image_flag": datasets.Value("bool"),
        }
    )
    remove_columns = list(dataset.features.keys() - features.keys())
    # suit to both VLM-RL/Ocean-R1 and MiniMax-AI/One-RL-to-See-Them-All data
    prompt_getter = lambda data: data["prompt"]
    ground_truth_getter = lambda data: [x["ground_truth"] for x in data["reward_model"]]
    image_getter = lambda data: data["images"]
    tag_getter = lambda data: data["data_source"]
    processor.image_processor.min_pixels, processor.image_processor.max_pixels = (
        data_args.image_min_pixels,
        data_args.image_max_pixels,
    )
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(
            data, processor, prompt_getter, ground_truth_getter, image_getter, tag_getter, prompt_image_token="<image>"
        ),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        features=features,
        remove_columns=remove_columns,
        desc="Encoding dataset",
    )
    print(f"Encoding: {dataset}")
    if cache_path:
        dataset.save_to_disk(cache_path)
    return dataset


def get_vlm_data_kwargs(data_args, tokenizer, processor, is_val=False):
    dataset = get_vlm_dataset(data_args, encode_function, processor, get_eval=is_val)
    collect_fn_kwargs = dict(
        extra_unpadded_keys=["reward_model", "tag"],
        prompt_key="prompt",
        answer_key="ground_truth",
        image_key="images",
        image_flag_key="image_flag",
        image_sample_kwargs={"min_pixels": data_args.image_min_pixels, "max_pixels": data_args.image_max_pixels},
    )
    return dict(dataset=dataset, collect_fn_kwargs=collect_fn_kwargs)


### for video rlvr demo
# we use video-r1 116k video data with adjusted format, case for training data:
#   {
#     "modality_type": "video",
#     "problem": "Why does the video conclude with a red screen displaying text?Options:\nA. To promote the BBC one program 'Super Cute Animals'\nB. To provide a warning message\nC. To show the credits of the video\nD. To indicate a change in scene\n",
#     "solution": "<answer>A</answer>",
#     "video": "videos/youtube_video_2024/ytb_HBxn56l9WcU.mp4",
#     "problem_type": "multiple choice"
#   }
# reference Video-R1
QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
}


def video_r1_format_prompt(prompt, processor, problem_type, modality_type=None):
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": modality_type},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=prompt) + TYPE_TEMPLATE[problem_type]},
                ]
                if modality_type
                else [
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=prompt) + TYPE_TEMPLATE[problem_type]}
                ]
                + TYPE_TEMPLATE[problem_type],
            }
        ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def video_r1_encode_function(data, processor):
    encodings = {
        "prompt": [
            video_r1_format_prompt(problem, processor, problem_type, modality_type)
            for problem, problem_type, modality_type in zip(
                data["problem"], data["problem_type"], data["modality_type"]
            )
        ],
        "video": data["video"],
        "ground_truth": data["solution"],
        "reward_model": data["problem_type"],
        "tag": data["modality_type"],
    }
    if "fps" in data:
        encodings["fps"] = data["fps"]
    return encodings


def video_r1_get_dataset(data_args, encode_function, processor, get_eval=False):
    cache_path = getattr(data_args, "cache_path", None)
    if cache_path:
        cache_path = os.path.join(cache_path, "val" if get_eval else "train")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset
    dataset = get_dataset(data_args=data_args)
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(data, processor),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        desc="Encoding dataset",
    )
    print(f"Encoding: {dataset}")
    if cache_path:
        dataset.save_to_disk(cache_path)
    return dataset


def video_r1_get_data_kwargs(data_args, tokenizer, processor, is_val=False):
    dataset = video_r1_get_dataset(data_args, video_r1_encode_function, processor, get_eval=is_val)
    collect_fn_kwargs = dict(
        extra_unpadded_keys=["reward_model", "tag"],
        video_folder=data_args.video_folder,
        video_sample_kwargs={  # settings same with RewatchR1 and VideoR1
            "total_pixels": data_args.video_total_pixels,
            "min_pixels": data_args.video_min_pixels,
            "max_pixels": data_args.video_max_pixels,
            "fps": data_args.video_fps,
            "max_frames": data_args.video_max_frames,
            "video_resize_chunk_frames": data_args.video_resize_chunk_frames,
        },
        video_meta_keys=[],
        is_template_applied=True,
    )
    get_data_item_kwargs = dict(use_dataloader=True, use_collect_fn=True, num_workers=4)
    return dict(dataset=dataset, collect_fn_kwargs=collect_fn_kwargs, get_data_item_kwargs=get_data_item_kwargs)


def audio_test_format_prompt(prompt, processor, use_audio=True, prompt_audio_token=None):
    question_template = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": question_template.format(Question=prompt)},
                ]
                if use_audio and not prompt_audio_token
                else [
                    {"type": "text", "text": question_template.format(Question=prompt)}
                ],  # audio_token has been included in prompt
            }
        ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_audio_token:
        text = text.replace(prompt_audio_token, "<|audio_start|><|audio_pad|><|audio_end|>")
    return text


### for video rlvr demo
def audio_test_encode_function(
    data, processor, prompt_getter, ground_truth_getter, audio_getter, tag_getter, prompt_audio_token=None
):
    audio_list = audio_getter(data)
    audio_flag = [True] * len(audio_list)
    text_list = []
    for idx, instruct in enumerate(prompt_getter(data)):
        # provide prompt_audio_token if audio_token in prompt
        text = audio_test_format_prompt(
            instruct, processor, use_audio=audio_flag[idx], prompt_audio_token=prompt_audio_token
        )
        text_list.append(text)
    encodings = {
        "tag": tag_getter(data),
        "audios": audio_list,
        "prompt": text_list,
        "ground_truth": ground_truth_getter(data),
        "reward_model": data["reward_model"],
        # for text and multi-modal mixed data usage, indicating valid audio
        "audio_flag": audio_flag,
    }
    return encodings


def audio_test_get_dataset(data_args, encode_function, processor, get_eval=False):
    cache_path = getattr(data_args, "cache_path", None)
    if cache_path:
        cache_path = os.path.join(cache_path, "val" if get_eval else "train")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset
    dataset = get_dataset(data_args=data_args)
    # regularized data filed
    features = datasets.Features(
        {
            "tag": datasets.Value(dtype="string"),  # from data_source
            "audios": datasets.Sequence(datasets.Value(dtype="string")),
            "prompt": datasets.Value(dtype="string"),
            "ground_truth": datasets.Value(dtype="string"),
            "reward_model": dataset.features["reward_model"],
            # for text and multi-modal mixed data usage, indicating valid audio
            "audio_flag": datasets.Value("bool"),
        }
    )
    remove_columns = list(dataset.features.keys() - features.keys())
    prompt_getter = lambda data: data["prompt"]
    ground_truth_getter = lambda data: [x["ground_truth"] for x in data["reward_model"]]
    audio_getter = lambda data: data["audios"]
    tag_getter = lambda data: data["data_source"]
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(
            data, processor, prompt_getter, ground_truth_getter, audio_getter, tag_getter, prompt_audio_token=None
        ),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        features=features,
        remove_columns=remove_columns,
        desc="Encoding dataset",
    )
    print(f"Encoding: {dataset}")
    if cache_path:
        dataset.save_to_disk(cache_path)
    return dataset


def audio_test_get_data_kwargs(data_args, tokenizer, processor, is_val=False):
    dataset = audio_test_get_dataset(data_args, audio_test_encode_function, processor, get_eval=is_val)
    collect_fn_kwargs = dict(
        extra_unpadded_keys=["reward_model", "tag"],
        prompt_key="prompt",
        answer_key="ground_truth",
        audio_key="audios",
        audio_flag_key="audio_flag",
        audio_folder=data_args.audio_folder,
        audio_sample_kwargs={},
        audio_meta_keys=[],
        is_template_applied=True,
    )
    return dict(dataset=dataset, collect_fn_kwargs=collect_fn_kwargs)
