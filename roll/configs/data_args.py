from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    template: Optional[str] = field(
        default="native",
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    enable_thinking: Optional[bool] = field(
        default=None,
        metadata={"help": "Control thinking mode in chat template. None = use template default (True for qwen3). "
                  "False = skip thinking (direct answers). True = enable thinking."},
    )
    domain_interleave_probs: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from domains in one batch."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    file_name: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "The name of file path name for train. Conflicts with `--dataset_name`"},
    )
    eval_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of file path name for eval. Conflicts with `--eval_dataset_name`"},
    )
    dataset_type: Optional[Union[List[str], str]] = field(
        default="json",
        metadata={"help": "The dataset type, for example, json."},
    )
    tag: Optional[str] = field(default="tag", metadata={"help": "Which column in file to use as domain selection"})
    id: Optional[str] = field(default="id", metadata={"help": "Which column in file to use as id"})
    prompt: Optional[str] = field(default=None, metadata={"help": "Which column in file to use as prompt"})
    response: Optional[str] = field(default="solution", metadata={"help": "Which column in file to use as label"})
    messages: Optional[str] = field(default=None, metadata={"help": "Which column in file to use as messages"})
    # args for multi-modal, corresponding to qwen-vl-utils
    image_max_pixels: int = field(
        default=4096 * 4096,
        metadata={"help": "The maximum number of pixels of image inputs."},
    )
    image_min_pixels: int = field(
        default=4 * 28 * 28,
        metadata={"help": "The minimum number of pixels of image inputs."},
    )
    video_max_pixels: int = field(
        default=2048 * 2048,
        metadata={"help": "The maximum number of pixels of video frames."},
    )
    video_min_pixels: int = field(
        default=4 * 28 * 28,
        metadata={"help": "The minimum number of pixels of video frames."},
    )
    video_total_pixels: int = field(
        default=16384 * 28 * 28,
        metadata={"help": "The total number of pixels for a video which can be used to limit sequence length."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_max_frames: int = field(
        default=128,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )
    video_resize_chunk_frames: int = field(
        default=64,
        metadata={"help": "The frames of the chunk to resize the video."},
    )
    rollout_dataloader_num_workers: int = field(
        default=1,
        metadata={
            "help": (
                "The number of DataLoader workers used by rollout schedulers for data loading. "
                "The number of prefetched samples is roughly "
                "rollout_dataloader_num_workers * prefetch_factor * rollout_dataloader_batch_size."
            )
        },
    )
    rollout_dataloader_batch_size: int = field(
        default=1,
        metadata={
            "help": (
                "The DataLoader batch size used by rollout schedulers for data loading. "
                "This is not the training rollout_batch_size; keep it small for video data because each worker "
                "prefetches batches and may decode many videos ahead of use."
            )
        },
    )
    image_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the images."})
    video_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the videos."})
    audio_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the audios."})
    custom_data_kwargs_func: Optional[str] = field(
        default=None, metadata={"help": "Path to custom data kwargs function."}
    )

    def __post_init__(self):
        assert not (
            self.prompt is not None and self.messages is not None
        ), "prompt and messages are mutually exclusive"
        assert self.rollout_dataloader_num_workers >= 0, "rollout_dataloader_num_workers must be non-negative"
        assert self.rollout_dataloader_batch_size > 0, "rollout_dataloader_batch_size must be positive"
