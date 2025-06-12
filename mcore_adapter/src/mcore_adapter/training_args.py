import json
from dataclasses import dataclass, field, fields
from typing import Literal, Optional, Union

from transformers import Seq2SeqTrainingArguments as HFSeq2SeqTrainingArguments
from transformers import TrainingArguments as HFTrainingArguments

from .utils import get_logger


logger = get_logger(__name__)


@dataclass
class DistributingParallelArguments:
    tensor_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of tensor model parallelism."},
    )
    pipeline_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of pipeline model parallelism."},
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={
            "help": "Makes tensor parallelism more memory efficient for LLMs (20B+) by parallelizing layer norms"
            "and dropout sequentially."
        },
    )
    virtual_pipeline_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Num of virtual pipeline in a pipeline."},
    )
    context_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of context parallelism."},
    )
    expert_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of expert model parallelism."},
    )
    account_for_embedding_in_pipeline_split: Optional[bool] = field(
        default=None,
        metadata={
            "help": "If set, the embedding layer will be treated as a standard transformer"
            "layer in the context of partition and placement for pipeline parallelism."
        },
    )
    account_for_loss_in_pipeline_split: Optional[bool] = field(
        default=None,
        metadata={
            "help": "If set, the loss layer will be treated as a standard transformer"
            "layer in the context of partition and placement for pipeline parallelism."
        },
    )
    overlap_p2p_comm: bool = field(
        default=True,
        metadata={
            "help": "Overlap pipeline parallel communication with forward and backward chunks. Only works with virtual pipeline."
        },
    )
    variable_seq_lengths: bool = field(
        default=False,
        metadata={
            "help": "Support for variable sequence lengths across microbatches. Setting this communicates the size"
            "of tensors during pipeline parallelism communication, because of this extra overhead it"
            "should only be set if the sequence length varies by microbatch within a global batch."
        },
    )
    # recompute
    distribute_saved_activations: Optional[bool] = field(
        default=None,
        metadata={"help": "If True, distribute recomputed activations across the model parallel group."},
    )
    recompute_granularity: Optional[Literal["full", "selective"]] = field(
        default=None,
        metadata={
            "help": "Checkpoint activations to allow for training with larger models, sequences, and batch sizes. "
            "It is supported at two granularities 1) full: whole transformer layer is recomputed, "
            "2) selective: core attention part of the transformer layer is recomputed.",
            "choices": ["full", "selective"],
        },
    )
    recompute_method: Optional[Literal["uniform", "recompute"]] = field(
        default=None,
        metadata={
            "help": "1) uniform: uniformly divide the total number of Transformer layers and "
            "recompute the input activation of each divided chunk at specified granularity, "
            "2) recompute the input activations of only a set number of individual Transformer layers "
            "per pipeline stage and do the rest without any recomputing at specified granularity. "
            "If None and recompute_granularity is full, all layers will do recomputation.",
            "choices": ["uniform", "recompute"],
        },
    )
    recompute_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "A comma-separated list of modules to recompute. Only effective when recompute_granularity "
            "is set to 'selective'. Choices: core_attn, moe_act, layernorm, mla_up_proj, mlp, moe. Default: core_attn"
        },
    )
    recompute_num_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "1) uniform: the number of Transformer layers in each uniformly divided recompute unit, "
            "2) block: the number of individual Transformer layers to recompute within each pipeline stage."
        },
    )
    # fusion
    bias_activation_fusion: bool = field(
        default=False,
        metadata={"help": "Fuse bias addition and the activation function when possible."},
    )
    apply_rope_fusion: bool = field(
        default=False,
        metadata={"help": "Use fused RoPE kernel."},
    )
    # moe
    moe_layer_recompute: bool = field(
        default=False,
        metadata={"help": "Memory optimization: checkpointing moe_layer to save activation memory."},
    )
    moe_token_dispatcher_type: Literal["allgather", "alltoall"] = field(
        default="allgather",
        metadata={
            "help": "The type of token dispatcher to use. Options are 'allgather' and 'alltoall'",
            "choices": ["allgather", "alltoall"],
        },
    )
    moe_aux_loss_coeff: Optional[float] = field(
        default=None,
        metadata={"help": "Scaling coefficient for the aux loss."},
    )
    moe_grouped_gemm: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When there are multiple experts per rank, compress multiple local (potentially small) gemms"
            "in a single kernel launch to improve the utilization and performance by leveraging the Grouped"
            "GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm)."
        },
    )
    moe_expert_capacity_factor: Optional[float] = field(
        default=None,
        metadata={
            "help": "The capacity factor for each expert, None means no token will be dropped. The default is None."
        },
    )
    moe_pad_expert_input_to_capacity: Optional[bool] = field(
        default=None,
        metadata={
            "help": "If True, pads the input for each expert to match the expert capacity length, "
            "effective only after the moe_expert_capacity_factor is set. The default setting is False."
        },
    )
    moe_token_drop_policy: Optional[Literal["probs", "position"]] = field(
        default=None,
        metadata={
            "help": "The policy to drop tokens. Can be either `probs` or `position`. "
            "If `probs`, the tokens with the lowest probabilities will be dropped. "
            "If `position`, tokens at the end of each batch will be dropped",
            "choices": ["probs", "position"],
        },
    )
    moe_shared_expert_overlap: bool = field(
        default=False,
        metadata={
            "help": "Enable overlapping between shared expert computations and dispatcher communications."
            " Without this, the shared epxerts execute after the routed experts."
        },
    )
    moe_router_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Data type for routing and expert output weighted averaging. Using fp32 or fp64 can "
            "improve stability especially when the number of experts is large. None means no changes for dtype.",
        },
    )
    # train options
    calculate_per_token_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether cross entropy loss is calculated over the actual number of non-padded tokens in the"
            "global batch, versus the default behavior of assuming all tokens are non-padded."
        },
    )
    transformer_impl: Optional[Literal["local", "transformer_engine"]] = field(
        default=None,
        metadata={
            "help": "Which Transformer implementation to use.",
            "choices": ["local", "transformer_engine"],
        },
    )
    additional_configs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={
            "help": "Dictionary or Path to a JSON file containing additional configuration parameters for the model.",
        },
    )

    def __post_init__(self):
        if self.additional_configs is not None and isinstance(self.additional_configs, str):
            try:
                with open(self.additional_configs, "r", encoding="utf-8") as f:
                    self.additional_configs = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load additional configs from {self.additional_configs}: {e}")
                raise e

        if self.recompute_modules is not None and isinstance(self.recompute_modules, str):
            self.recompute_modules = self.recompute_modules.split(",")

    def get_config_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}


@dataclass
class MegatronArguments(DistributingParallelArguments):
    accumulate_allreduce_grads_in_fp32: bool = field(
        default=False,
        metadata={"help": "Gradient accumulation and all-reduce in fp32."},
    )
    use_distributed_optimizer: bool = field(
        default=False,
        metadata={"help": "Use distributed optimizer."},
    )
    overlap_grad_reduce: bool = field(
        default=False,
        metadata={"help": "If true, overlap grad reduce-scatter with backward compute in distributed optimizer."},
    )
    delay_grad_reduce: bool = field(
        default=True,
        metadata={"help": "If true, delay / synchronize grad reductions in all but first PP stage."},
    )
    overlap_param_gather: bool = field(
        default=False,
        metadata={"help": "If true, overlap param all-gather with forward compute in distributed optimizer."},
    )
    check_for_nan_in_loss_and_grad: bool = field(
        default=True,
        metadata={"help": "Check for nan in loss and grad."},
    )
    ddp_average_in_collective: bool = field(
        default=False,
        metadata={
            "help": "If true, compute average in collective directly, as opposed to dividing by the"
            "dp_size first and then computing sum in the collective."
        },
    )
    ddp_bucket_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of parameters in each bucket. If unspecified, MCore uses a default"
            "value of max(40000000, 1000000 * dp_size) parameters (larger DP sizes need larger buckets"
            "to ensure collectives do not become latency-bound)."
        },
    )

    optimizer: str = field(default="adam", metadata={"help": "Optimizer function: [adam, sgd]"})
    save_hf_model: bool = field(default=False, metadata={"help": "Save model as hf format."})

    sequence_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.overlap_param_gather:
            assert self.use_distributed_optimizer, "--overlap_param_gather only supported with distributed optimizer"
            assert (
                self.overlap_grad_reduce
            ), "--overlap_grad_reduce should be turned on when using --overlap_param_gather"

    @classmethod
    def from_json_file(cls, json_file_path) -> "MegatronArguments":
        with open(json_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls(**json.loads(text))

    def allow_variable_seq_lengths(self):
        return self.variable_seq_lengths or self.pipeline_model_parallel_size <= 1


@dataclass
class TrainingArguments(MegatronArguments, HFTrainingArguments):
    def __post_init__(self):
        if self.bf16:
            self.accumulate_allreduce_grads_in_fp32 = True

        self.deepspeed = None
        MegatronArguments.__post_init__(self)
        HFTrainingArguments.__post_init__(self)
        if self.report_to is not None:
            self.report_to = [k for k in self.report_to if k != "wandb"]


@dataclass
class Seq2SeqTrainingArguments(MegatronArguments, HFSeq2SeqTrainingArguments):
    def __post_init__(self):
        if self.bf16:
            self.accumulate_allreduce_grads_in_fp32 = True

        self.deepspeed = None
        MegatronArguments.__post_init__(self)
        HFSeq2SeqTrainingArguments.__post_init__(self)
        if self.report_to is not None:
            self.report_to = [k for k in self.report_to if k != "wandb"]
