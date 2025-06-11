import dataclasses
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Dict, Union, List

from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

logger = get_logger()

@dataclass
class ScheduleConfig:
    generate_opt_level: int = field(
        default=1,
        metadata={
            "help": "generate optimizing level: 0 use base batch generate interface, 1 use scheduler process requests"
        },
    )
    is_num_return_sequences_expand: bool = field(
        default=False,
        metadata={"help": "whether replicate `num_return_sequences` times in prompts or not."}
    )
    max_running_requests: int = field(
        default=128,
        metadata={"help": "The maximum number of running requests."}
    )
    is_use_additional_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to use additional prompts or not."}
    )
    max_additional_running_prompts: int = field(
        default=16, metadata={"help": "The additional number of running prompts, beyond batch_size."}
    )


@dataclass
class BaseConfig:

    exp_name: str = field(
        default=os.path.basename(sys.argv[0])[: -len(".py")],
        metadata={"help": "The name of this experiment (defaults to the file name without the .py extension)."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initializations."}
    )
    rpc_timeout: int = field(
        default=3600,
        metadata={"help": "Timeout duration for RPC calls in seconds."}
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    logging_dir: str = field(
        default="./output/logs",
        metadata={"help": "Directory to store logs."})
    track_with: str = field(
        default="tensorboard",
        metadata={"help": "The type of tracker to be used for tracking, one of ['wandb', 'tensorboard', 'stdout']."}
    )
    tracker_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Additional keyword arguments to pass to the Tracker class."}
    )
    max_steps: int = field(
        default=500,
        metadata={"help": "If > 0: set total number of pipeline steps"},
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "Save checkpoint every X update steps."}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Number of steps between logging information."}
    )
    eval_steps: int = field(
        default=10,
        metadata={"help": "Run an evaluation every X steps."},
    )
    rollout_batch_size: int = field(
        default=128, metadata={"help": "The number of samples to rollout in each inference batch."}
    )
    val_batch_size: int = field(
        default=128,
        metadata={"help": "The number of samples to rollout in each val batch."})
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training; set to -1 if not applicable."}
    )
    resume_from_checkpoint: Union[bool, str] = field(
        default=False,
        metadata={"help": "load the last checkpoint in *output_dir* as saved by a previous instance or MOS URI."},
    )
    checkpoint_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration checkpoint, this field will be written to worker_config."},
    )
    prompt_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum length of a prompt to be padded."},
    )
    response_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the generated tokens to be padded."},
    )
    sequence_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the sequence to be padded."},
    )
    alive_check_interval: int = field(
        default=10,
        metadata={"help": "The interval of worker alive check."}
    )
    profiler_timeline: bool = field(default=False, metadata={"help": "Whether to use profiler mode or not."})
    profiler_memory: bool = field(default=False, metadata={"help": "Whether to use profiler memory or not."})
    profiler_output_dir: str = field(
        default="./output/profiler", metadata={"help": "Directory to write profiler logs."}
    )
    system_envs: dict = field(
        default_factory=dict,
        metadata={"help": "system environment variables."}
    )
    num_nodes: int = field(
        default=1,
        metadata={"help": "Number of nodes available for distributed training."}
    )
    num_gpus_per_node: int = field(
        default=8,
        metadata={
            "help": "Specifies the number of GPUs available per node. When the number of nodes is greater than 1, "
                    "num_gpus_per_node should request the total number of GPUs in the entire node."
                    "Ensure that GPU resource allocation aligns with the request in a multi-node setup."
        }
    )
    model_download_type: str = field(
        default="MODELSCOPE",
        metadata={"help": "snapshot_download func source type, such as MODELSCOPE, HUGGINGFACE_HUB."},
    )


    def to_dict(self):
        return dataclasses.asdict(self)

    def __post_init__(self):

        assert self.response_length or self.sequence_length, "response_length or sequence_length must be set"

        if self.sequence_length is None:
            self.sequence_length = self.response_length + self.prompt_length
            logger.warning(
                f"sequence_length is not set, use response_length + prompt_length as sequence_length: {self.sequence_length}"
            )

        if self.response_length is not None:
            logger.warning(
                f"response_length is deprecated, use sequence_length instead, sequence_length is {self.sequence_length}"
            )
            self.response_length = None

        if self.track_with == "tensorboard":
            self.tracker_kwargs["log_dir"] = os.path.join(
                self.tracker_kwargs.get("log_dir", self.output_dir), self.exp_name, datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            logger.info(f"add timestamp to tensorboard log_dir {self.tracker_kwargs['log_dir']}")

        self.logging_dir = os.path.join(self.logging_dir, self.exp_name)
        logger.info(f"add exp_name to logging_dir {self.logging_dir}")
        os.environ["ROLL_LOG_DIR"] = self.logging_dir
        get_logger()

        os.environ["MODEL_DOWNLOAD_TYPE"] = self.model_download_type

        upload_type = self.checkpoint_config.get("type", None)
        if upload_type == "file_system":
            output_dir = self.checkpoint_config.get("output_dir")
            self.checkpoint_config["output_dir"] = os.path.join(output_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
            logger.info(f"add timestamp to output_dir {self.checkpoint_config['output_dir']}")

        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if hasattr(attribute, "checkpoint_config"):
                    setattr(attribute, "checkpoint_config", self.checkpoint_config)

            if isinstance(attribute, WorkerConfig):
                if hasattr(attribute, "training_args"):
                    setattr(attribute.training_args, "seed", self.seed)

        assert not (
            self.profiler_timeline and self.profiler_memory
        ), f"ensure that only one profiling mode is enabled at a time"

        self.profiler_output_dir = os.path.join(
            self.profiler_output_dir, self.exp_name, datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        os.environ["PROFILER_OUTPUT_DIR"] = self.profiler_output_dir
        if self.profiler_timeline:
            os.environ["PROFILER_TIMELINE"] = "1"
        if self.profiler_memory:
            os.environ["PROFILER_MEMORY"] = "1"
        if self.rpc_timeout is not None:
            os.environ["roll_RPC_TIMEOUT"] = str(self.rpc_timeout)
        os.environ.update(self.system_envs)

        # the required num nodes
        total_devices = []
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if attribute.device_mapping is not None:
                    total_devices.extend(attribute.device_mapping)
        max_gpu_num = max(total_devices)
        if max_gpu_num <= self.num_gpus_per_node:
            self.num_nodes = 1
        else:
            self.num_nodes = (max_gpu_num + self.num_gpus_per_node - 1) // self.num_gpus_per_node


    def set_max_steps(self, max_steps: int):
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if hasattr(attribute, "training_args"):
                    setattr(attribute.training_args, "max_steps", max_steps)
