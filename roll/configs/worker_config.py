from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from roll.configs import DataArguments, GeneratingArguments, ModelArguments
from roll.configs.training_args import TrainingArguments
from roll.utils.logging import get_logger
logger = get_logger()


@dataclass
class StrategyArguments:
    strategy_name: Literal[
        "deepspeed_train",
        "hf_infer",
        "deepspeed_infer",
        "vllm",
        "sglang",
        "megatron_infer",
        "megatron_train",
        "diffusion_deepspeed_train",
        "fsdp2_train",
        "fsdp2_infer",
    ] = field(
        default="deepspeed_train",
        metadata={
            "help": "The name of the strategy. Options: 'deepspeed_train', 'diffusion_deepspeed_train', 'hf_infer', 'deepspeed_infer', 'vllm', 'sglang', "
            "'megatron_infer', 'megatron_train', 'fsdp2_train', 'fsdp2_infer'."
        },
    )
    strategy_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration dictionary for the strategy."},
    )

@dataclass
class SequencePackingConfig:
    algorithm: str = field(
        default="none",
        metadata={"help": "Sequence packing algorithm: 'none' (default partitioning) or 'load_balance' "
                          "(redistribute sentences across microbatches for better load balancing). "
                          "Note: 'load_balance' requires proper loss scaling as microbatches contain "
                          "different numbers of sentences."}
    )

    max_packed_sequence_length_forward: int = field(
        default=None,
        metadata={"help": "Maximum sequence length after packing sentences in a microbatch during inference. "
                          "With context parallelism enabled, each CP rank handles "
                          "max_packed_sequence_length_forward // cp_size."}
    )

    max_packed_sequence_length_train: int = field(
        default=None,
        metadata={"help": "Maximum sequence length after packing sentences in a microbatch during training. "
                          "With context parallelism enabled, each CP rank handles "
                          "max_packed_sequence_length_train // cp_size."}
    )

    min_num_micro_batches_forward: int = field(
        default=1,
        metadata={"help": "Minimum number of microbatches per mini-batch during inference. "
                          "Used with 'load_balance' algorithm to control samples per microbatch "
                          "and memory usage."}
    )

    min_num_micro_batches_train: int = field(
        default=1,
        metadata={"help": "Minimum number of microbatches per mini-batch (per gradient update) during training. "
                          "Used with 'load_balance' algorithm to control samples per microbatch "
                          "and memory usage."}
    )


@dataclass
class WorkerConfig:
    name: str = field(
        default=None,
        metadata={"help": "name of this role."},
    )
    worker_cls: Optional[str] = field(default=None, metadata={"help": "The class of the worker."})
    pg_variant: Optional[str] = field(
        default=None,
        metadata={"help": "The variant of the policy gradient."},
    )
    model_args: ModelArguments = field(
        default_factory=ModelArguments,
        metadata={"help": "The arguments for the model, encapsulated in a ModelArguments object."},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={"help": "Training-related arguments."},
    )
    data_args: DataArguments = field(
        default=None,
        metadata={"help": "Data-related arguments; optional and can be None."},
    )
    generating_args: GeneratingArguments = field(
        default=None,
        metadata={"help": "Arguments for generating output; optional and can be None."},
    )
    strategy_args: StrategyArguments = field(
        default=None,
        metadata={"help": "The strategy configuration, encapsulated in a StrategyArguments object."},
    )
    world_size: int = field(default=None, metadata={"help": "The number of role clusters."})
    device_mapping: Union[List[int], str] = field(
        default=None,
        metadata={
            "help": "The list of device ids to use when training. "
            "Configure it as a string that can be evaluated as List[int], such as 'list(range(0, 8))'."
            "If device_mapping is None, the worker uses cpu only."
        },
    )
    num_gpus_per_worker: int = field(default=1, metadata={"help": "The number of gpu per worker."})
    model_update_frequency: int = field(default=1, metadata={"help": "Frequency of model updates."})
    infer_batch_size: int = field(default=16, metadata={"help": "Batch size for inference."})
    backend_timeout: int = field(
        default=30,
        metadata={"help": "minutes for dist backend communicating."},
    )
    system_envs: dict = field(
        default_factory=dict,
        metadata={"help": "system environment variables for this worker."},
    )
    topr_positive_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for positive samples in TOPR loss."},
    )
    topr_negative_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for negative samples in TOPR loss."},
    )
    max_concurrency: int = field(default=1, metadata={"help": "max_concurrency of this Ray Actor"})

    use_dynamic_batching_in_train: bool = field(
        default=False,
        metadata={
            "help": "Dynamic batching is a feature designed to group sequences of similar lengths into batches, "
            "minimizing padding and improving computational and memory efficiency."
        },
    )
    max_tokens_per_microbatch_in_train: int = field(
        default=0,
        metadata={
            "help": (
                "Set the maximum number of tokens for each micro-batch during training. "
                "This config must be set when using dynamic batching. "
                "Recommended value: sequence_length × 2 × micro_batch_size."
            )
        },
    )
    sequence_length_round_in_train: int = field(
        default=4,
        metadata={
            "help": "The value to round up to when truncating the sequence length."
            "Note: This config must be set when using dynamic batching."
        },
    )
    use_dynamic_batching_in_infer: bool = field(
        default=False,
        metadata={
            "help": "Dynamic batching is a feature designed to group sequences of similar lengths into batches, "
            "minimizing padding and improving computational and memory efficiency."
        },
    )
    max_tokens_per_microbatch_in_infer: int = field(
        default=None,
        metadata={
            "help": "Set the maximum number of tokens for each micro-batch. "
            "Note: This config must be set when using dynamic batching."
        },
    )
    sequence_length_round_in_infer: int = field(
        default=4,
        metadata={
            "help": "The value to round up to when truncating the sequence length."
            "Note: This config must be set when using dynamic batching."
        },
    )
    offload_nccl: bool = field(
        default=False,
        metadata={"help": "Whether offload nccl buffer to save gpu memory."},
    )

    # sequence packing
    use_sequence_packing: bool = field(
        default=False,
        metadata={
            "help": "Concatenates multiple sequences into a single “packed” sequence, eliminating most padding. "
            "Only supported in the megatron strategy"
        },
    )

    sequence_packing_args: SequencePackingConfig = field(
        default_factory= SequencePackingConfig,
        metadata={
            "help": "Sequence packing related arguments "
        }
    )


    logits_in_fp32: bool = field(
        default=True,
        metadata={
            "help": "Force logits dtype to Float"
        }
    )

    apply_loss_scale: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to multiply the aggregated loss by the global loss_scale (typically the total number of "
                "micro-batches in a global step, i.e., DP×GA) to cancel the backend’s default gradient-mean behavior "
                "under Data Parallel + Gradient Accumulation. This restores a sum-over-microbatches semantics so the "
                "resulting gradients are equivalent to computing the loss on the full global batch at once with the "
                "global denominator (especially important with variable-length inputs/sequence packing). Disable only "
                "if you already apply an equivalent scaling elsewhere or your backend does not average across DP/GA."
            )
        }
    )

    def __post_init__(self):

        if self.strategy_args is not None:
            if self.strategy_args.strategy_name not in ["hf_infer", "vllm", "sglang"] and self.num_gpus_per_worker > 1:
                logger.info(
                    f"strategy_name={self.strategy_args.strategy_name}, force set num_gpus_per_worker={self.num_gpus_per_worker} to 1."
                )
                self.num_gpus_per_worker = 1
            if self.strategy_args.strategy_name == "vllm":
                strategy_config = self.strategy_args.strategy_config
                tensor_parallel_size = strategy_config.get("tensor_parallel_size", 1)
                pipeline_parallel_size = strategy_config.get("pipeline_parallel_size", 1)
                self.num_gpus_per_worker = tensor_parallel_size * pipeline_parallel_size
                logger.info(
                    f"set vllm num_gpus_per_worker to {self.num_gpus_per_worker}, "
                    f"tensor_parallel_size: {tensor_parallel_size}, "
                    f"pipeline_parallel_size: {pipeline_parallel_size}"
                )

        if self.device_mapping is not None:
            self.device_mapping = eval(self.device_mapping)
            assert (
                len(self.device_mapping) % self.num_gpus_per_worker == 0
            ), f"len(device_mapping)={len(self.device_mapping)} must be divisible by num_gpus_per_worker={self.num_gpus_per_worker}."
            self.world_size = len(self.device_mapping) // self.num_gpus_per_worker
        else:
            self.num_gpus_per_worker = 0

        self.resource_placement_groups: Optional[List[Dict]] = None
        self.checkpoint_config: Optional[Dict] = None

        if hasattr(self, "model_args"):
            if self.model_args.dtype == "bf16":
                self.training_args.bf16 = True
            elif self.model_args.dtype == "fp16":
                self.training_args.fp16 = True



def is_actor_infer_overlapping_with_any_cluster(actor_infer: WorkerConfig, actor_train: WorkerConfig = None, reference: WorkerConfig = None, critic: WorkerConfig = None) -> bool:
    """
    Check if actor_infer overlaps with ANY of the provided clusters.

    Args:
        actor_infer: The actor_infer WorkerConfig
        actor_train: The actor_train WorkerConfig (optional)
        reference: The reference WorkerConfig (optional)
        critic: The critic WorkerConfig (optional)

    Returns:
        True if actor_infer overlaps with any provided cluster, False otherwise
    """
    infer_devices = set(actor_infer.device_mapping or [])

    clusters = {
        'actor_train': actor_train,
        'reference': reference,
        'critic': critic
    }

    for cluster_name, cluster_config in clusters.items():
        if cluster_config is not None:
            cluster_devices = set(cluster_config.device_mapping or [])
            if infer_devices.intersection(cluster_devices):
                return True

    return False

