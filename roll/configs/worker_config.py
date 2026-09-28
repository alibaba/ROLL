from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from roll.configs import DataArguments, GeneratingArguments, ModelArguments
from roll.configs.training_args import TrainingArguments
from roll.utils.logging import get_logger
logger = get_logger()


@dataclass
class RouterReplayConfig:
    """Configuration for router replay functionality."""
    mode: Literal["disable", "R2", "R3"] = field(
        default="disable",
        metadata={
            "help": "Router replay mode. Options: 'disable' (no replay), 'R2' (Megatron infer + Megatron train), 'R3' (sglang infer + Megatron train)."
        },
    )


@dataclass
class StrategyArguments:
    strategy_name: Literal[
        "hf_infer",
        "vllm",
        "vllm_omni",
        "sglang",
        "megatron_infer",
        "megatron_train",
        "fsdp2_train",
        "fsdp2_infer",
        "fsdp2_diffusion_train",
        "fsdp2_diffusion_infer",
        "veomni_train",
        "veomni_infer",
    ] = field(
        default="fsdp2_train",
        metadata={
            "help": "The name of the strategy. Options: 'hf_infer', 'vllm', 'sglang', "
            "'vllm_omni', 'megatron_infer', 'megatron_train', 'fsdp2_train', 'fsdp2_infer', "
            "'fsdp2_diffusion_train', 'fsdp2_diffusion_infer', 'veomni_train', 'veomni_infer'."
        },
    )
    strategy_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration dictionary for the strategy."},
    )

    def __post_init__(self):
        # Ensure strategy_config is always a dict, even when YAML sets it to null (~)
        if self.strategy_config is None:
            self.strategy_config = {}

@dataclass
class SequencePackingConfig:
    algorithm: str = field(
        default="load_balance",
        metadata={"help": "Sequence packing algorithm: 'none' (simple in-order partitioning) or 'load_balance' "
                          "(default; redistribute sequences across microbatches for better load balancing)."}
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
        metadata={
            "help": "Release NCCL communicator dynamic GPU memory while states are offloaded. "
            "Requires NCCL >= 2.29.7 (ncclCommSuspend/Resume)."
        },
    )

    # sequence packing
    use_sequence_packing: bool = field(
        default=True,
        metadata={
            "help": "Concatenates multiple sequences into a single “packed” sequence, eliminating most padding. "
            "Only supported in the megatron strategy. Uses the 'load_balance' algorithm by default. "
            "max_packed_sequence_length_forward/train auto-compute as sequence_length * batch_size when None."
        },
    )

    sequence_packing_args: SequencePackingConfig = field(
        default_factory= SequencePackingConfig,
        metadata={
            "help": "Sequence packing related arguments "
        }
    )


    logits_in_fp32: bool = field(
        default=False,
        metadata={
            "help": "Force logits dtype to Float"
        }
    )

    use_logits_chunking: bool = field(
        default=False,
        metadata={
            "help": "Chunk log_probs_from_logits/entropy_from_logits along the sequence dim to reduce peak "
                    "memory on long sequences. Disabled by default (single-pass path); enable for long-sequence "
                    "training that would otherwise OOM on the fp32 [B, T, V] intermediates."
        }
    )
    logits_chunk_size: int = field(
        default=2048,
        metadata={
            "help": "Sequence-dim chunk size for log_probs_from_logits/entropy_from_logits when chunking is "
                    "enabled. Larger values use more peak memory but fewer iterations; only takes effect "
                    "when seq_len exceeds it."
        }
    )

    # Router Replay Configuration
    router_replay: RouterReplayConfig = field(
        default_factory=RouterReplayConfig,
        metadata={
            "help": "Configuration for router replay in training. Only supported for Megatron strategy."
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

    # MTP training configuration
    mtp_training_mode: Optional[Literal["disabled", "standalone", "joint", "mtp_only"]] = field(
        default="disabled",
        metadata={
            "help": "MTP training mode for this worker. "
            "'disabled': MTP is loaded but not trained (default). "
            "'standalone': MTP is trained independently with truncated gradients (no gradient flow to main model). "
            "'joint': MTP participates in main model updates with full gradient flow. "
            "'mtp_only': the main model is frozen and only MTP parameters are trained "
            "(requires mtp_num_layers > 0, Megatron strategy only)."
        },
    )

    # OPD (On-Policy Distillation) KL coefficient for teacher workers
    opd_kl_coef: Optional[float] = field(
        default=None,
        metadata={
            "help": "KL coefficient for this teacher in OPD mode. "
            "Defaults to 1.0 if not set."
        },
    )
    # Tags this teacher handles for multi-teacher OPD routing
    tag_included: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Tags this teacher handles in multi-teacher OPD mode. "
            "Empty list means this teacher handles all tags (default). "
            "Used to route data to specific teachers based on domain/tag."
        },
    )

    def __post_init__(self):
        if self.use_sequence_packing and (self.use_dynamic_batching_in_train or self.use_dynamic_batching_in_infer):
            # Dynamic batching and sequence packing are mutually exclusive: the
            # strategy picks one for micro-batch layout, but leftover packing flags
            # still trigger packing-only post-processing (restore_results_order,
            # cu_seqlens handling), which breaks under dynamic batching.
            logger.warning(
                "use_sequence_packing=True conflicts with dynamic batching, forcing use_sequence_packing=False."
            )
            self.use_sequence_packing = False
        if (
            self.offload_nccl
            and self.strategy_args is not None
            and self.strategy_args.strategy_name in {"vllm", "vllm_omni"}
        ):
            raise ValueError("offload_nccl is not supported for vLLM; set offload_nccl=False for this worker")
        if self.use_sequence_packing and self.strategy_args is not None:
            if "megatron" not in self.strategy_args.strategy_name:
                logger.warning(
                    f"use_sequence_packing=True but strategy_name={self.strategy_args.strategy_name}; "
                    "packing is only implemented for megatron strategies, forcing use_sequence_packing=False."
                )
                self.use_sequence_packing = False

        if self.strategy_args is not None:
            if self.strategy_args.strategy_name not in ["hf_infer", "vllm", "vllm_omni", "sglang"] and self.num_gpus_per_worker > 1:
                logger.info(
                    f"strategy_name={self.strategy_args.strategy_name}, force set num_gpus_per_worker={self.num_gpus_per_worker} to 1."
                )
                self.num_gpus_per_worker = 1
            if self.strategy_args.strategy_name in ["vllm", "vllm_omni"]:
                strategy_config = self.strategy_args.strategy_config
                tensor_parallel_size = strategy_config.get("tensor_parallel_size", 1)
                pipeline_parallel_size = strategy_config.get("pipeline_parallel_size", 1)
                self.num_gpus_per_worker = tensor_parallel_size * pipeline_parallel_size
                logger.info(
                    f"set {self.strategy_args.strategy_name} num_gpus_per_worker to {self.num_gpus_per_worker}, "
                    f"tensor_parallel_size: {tensor_parallel_size}, "
                    f"pipeline_parallel_size: {pipeline_parallel_size}"
                )

            # Validate router_replay configuration
            if self.router_replay.mode == "R2":
                if self.strategy_args.strategy_name not in ["megatron_train", "megatron_infer"]:
                    logger.warning(
                        f"router_replay [R2] is only supported for megatron_train and megatron_infer strategy, "
                        f"but current strategy is {self.strategy_args.strategy_name}. "
                        f"router_replay will be ignored."
                    )
            elif self.router_replay.mode == "R3":
                if self.strategy_args.strategy_name not in ["megatron_train", "sglang", "vllm"]:
                    logger.warning(
                        f"router_replay [R3] is only supported for megatron_train, sglang and vllm strategy, "
                        f"but current strategy is {self.strategy_args.strategy_name}. "
                        f"router_replay will be ignored."
                    )

        if self.device_mapping is not None:
            if isinstance(self.device_mapping, str):
                self.device_mapping = eval(self.device_mapping)
            assert (
                len(self.device_mapping) % self.num_gpus_per_worker == 0
            ), f"len(device_mapping)={len(self.device_mapping)} must be divisible by num_gpus_per_worker={self.num_gpus_per_worker}."
            self.world_size = len(self.device_mapping) // self.num_gpus_per_worker
        else:
            self.num_gpus_per_worker = 0

        self.resource_placement_groups: Optional[List[Dict]] = None
        self.checkpoint_config: Optional[Dict] = None

        # Flag to indicate if this worker is configured (has GPU or model path)
        has_gpu = bool(self.device_mapping)
        has_model = self.model_args is not None and self.model_args.model_name_or_path is not None
        self.is_configured: bool = has_gpu or has_model

        if hasattr(self, "model_args"):
            if self.model_args.dtype == "bf16":
                self.training_args.bf16 = True
            elif self.model_args.dtype == "fp16":
                self.training_args.fp16 = True

    def _auto_fill_packing_lengths(self, sequence_length: int):
        """Auto-compute max_packed_sequence_length_* when None.

        Called from BaseConfig.__post_init__ after sequence_length is computed.
        forward: sequence_length * infer_batch_size
        train: sequence_length * per_device_train_batch_size
        """
        if self.use_sequence_packing:
            sp = self.sequence_packing_args
            if sp.max_packed_sequence_length_forward is None:
                sp.max_packed_sequence_length_forward = sequence_length * self.infer_batch_size
            if sp.max_packed_sequence_length_train is None:
                sp.max_packed_sequence_length_train = sequence_length * self.training_args.per_device_train_batch_size
            logger.info(
                f"[{self.name}] sequence packing enabled: algorithm={sp.algorithm}, "
                f"max_packed_sequence_length_forward={sp.max_packed_sequence_length_forward}, "
                f"max_packed_sequence_length_train={sp.max_packed_sequence_length_train}"
            )


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
