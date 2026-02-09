import copy
import dataclasses
import enum
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.nn.functional as F
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from transformers import AutoConfig
from transformers.configuration_utils import CONFIG_NAME as HF_CONFIG_NAME

from ..constants import HUGGINGFACE_AUTOMAP_CACHE, MCA_CONFIG_NAME
from ..initialize import initialize_megatron
from ..training_args import DistributingParallelArguments, TrainingArguments
from ..utils import get_logger
from .converter.template import get_template
from .model_utils import check_and_get_attention_backend_by_env


if TYPE_CHECKING:
    from .converter.template import Template

logger = get_logger(__name__)


@dataclass
class PretrainedConfig:
    name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier."},
    )
    hf_model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Corresponding HuggingFace transformers model type."},
    )
    hf_config_json: Optional[str] = field(
        default=None,
        metadata={"help": "Corresponding HuggingFace transformers config json."},
    )

    def post_init(self):
        self.__post_init__()

    def to_dict(self):
        output = {}
        for k, v in self.__dict__.items():
            if callable(v):
                output[k] = None
            elif isinstance(v, list) and callable(v[0]):
                output[k] = None
            elif isinstance(v, PipelineParallelLayerLayout):
                output[k] = str(v)
            else:
                output[k] = copy.deepcopy(v)

        return output

    def to_json_string(self):
        save_dict = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, list) and callable(v[0]):
                continue
            if callable(v) or isinstance(v, (torch.dtype, enum.Enum)):
                continue
            if isinstance(v, PipelineParallelLayerLayout):
                v = str(v)
            save_dict[f.name] = v
        return json.dumps(save_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_json_file(cls, json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
            config_dict = json.loads(text)

            valid_field_names = {f.name for f in fields(cls)}

            filtered_config = {}
            removed_keys = []
            for k, v in config_dict.items():
                if k in valid_field_names:
                    filtered_config[k] = v
                else:
                    removed_keys.append(k)

            if removed_keys:
                logger.warning(
                    f"Config loading from {json_file_path}: "
                    f"Ignoring deprecated/unknown properties: {removed_keys}. "
                    "This might be due to a Megatron version upgrade."
                )

            return cls(**filtered_config)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, MCA_CONFIG_NAME)
        self.to_json_file(output_config_file)
        config_dict = json.loads(self.hf_config_json) if self.hf_config_json else {}
        if "auto_map" in config_dict:
            self.save_hf_auto_map_files(save_directory)

    def save_hf_auto_map_files(self, save_directory: str):
        # name_or_path denotes the path of the from_pretrained model, i.e., where auto map files are located
        # should archive the auto map files in a cache path avoiding the remote name_or_path path has been cleaned
        automap_cache_path = self.get_automap_cache()
        read_cache = os.path.isdir(automap_cache_path) and any(
            f.endswith(".py") for f in os.listdir(automap_cache_path)
        )
        hf_files_path = automap_cache_path if read_cache else self.name_or_path
        if not (hf_files_path and os.path.isdir(hf_files_path)):
            return
        for dirpath, _, files in os.walk(hf_files_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(dirpath, file)
                    rel_path = os.path.relpath(full_path, hf_files_path)
                    dest_path = os.path.join(save_directory, rel_path)
                    shutil.copyfile(full_path, dest_path)

    def update_with_args(self, args: "DistributingParallelArguments", verbose: bool = True):
        if args.additional_configs is not None:
            for k, v in args.additional_configs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                else:
                    logger.warning(f"Config {k} is not found in model config, will not update it.")

        for f in dataclasses.fields(DistributingParallelArguments):
            name = f.name
            if name in ["additional_configs"] or not hasattr(self, name):
                continue
            # args config has higher priority
            if getattr(args, name) is None:
                setattr(args, name, getattr(self, name))
            else:
                if verbose and getattr(args, name) != getattr(self, name):
                    logger.info(
                        f"Argument {name} value: {getattr(args, name)} is not same as "
                        f"model_config {getattr(self, name)}."
                    )
                setattr(self, name, getattr(args, name))
        self.bf16 = getattr(args, "bf16", self.bf16)
        self.fp16 = getattr(args, "fp16", self.fp16)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, args: Optional["TrainingArguments"] = None):
        config_file = os.path.join(model_name_or_path, MCA_CONFIG_NAME)
        config = None
        from_mca_ckpt = False
        post_init_func = getattr(cls, "__post_init__", None)
        if post_init_func is not None:  # call __post_init__ after config is loaded
            setattr(cls, "__post_init__", lambda self: None)

        if os.path.isfile(config_file):
            config = cls.from_json_file(config_file)
            from_mca_ckpt = True
        elif os.path.isfile(os.path.join(model_name_or_path, HF_CONFIG_NAME)):
            # from hf ckpt
            logger.info(f"Did not find {config_file}, loading HuggingFace config from {model_name_or_path}")
            hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            template: "Template" = get_template(hf_config.model_type)
            config = cls(**template.convert_hf_to_mca_config_kws(hf_config))

        if args is not None:
            config.update_with_args(args, verbose=from_mca_ckpt)
            initialize_megatron(args)

        if post_init_func is not None:
            setattr(cls, "__post_init__", post_init_func)
        config.post_init()

        config.name_or_path = model_name_or_path
        # set automap cache in local disk
        if int(os.getenv("RANK", "0")) == 0:
            automap_cache_path = config.get_automap_cache()
            if os.path.isdir(automap_cache_path):
                shutil.rmtree(automap_cache_path)
            os.makedirs(automap_cache_path)
            config.save_hf_auto_map_files(save_directory=automap_cache_path)
        return config

    def distribute_config_match(self, other):
        "check the config corresponding ckpt can be used for current config training"
        raise NotImplementedError("distribute_config_match not implemented")

    def get_automap_cache(self):
        return os.path.join(
            os.getenv("HUGGINGFACE_AUTOMAP_CACHE", HUGGINGFACE_AUTOMAP_CACHE),
            hashlib.sha256(self.name_or_path.encode()).hexdigest(),
        )


@dataclass
class McaModelConfig(TransformerConfig, PretrainedConfig):
    position_embedding_type: Literal["learned_absolute", "rope", "mrope", "yarn", "none"] = field(
        default="rope",
        metadata={
            "help": "Position embedding type.",
            "choices": ["learned_absolute", "rope", "mrope", "yarn", "none"],
        },
    )
    padded_vocab_size: Optional[int] = field(
        default=None,
        metadata={"help": "Padded vocab size."},
    )
    squared_relu: bool = field(
        default=False,
        metadata={"help": "Use squared relu activation instead of default gelu"},
    )
    swiglu: bool = field(
        default=False,
        metadata={"help": "Use gated linear units and SiLU activation instead of default gelu"},
    )
    tie_embeddings_and_output_weights: bool = field(
        default=False,
        metadata={"help": "Untie embeddings and output weights."},
    )
    max_sequence_length: int = field(
        default=0,
        metadata={"help": "Maximum size of sequence. This is used for positional embedding"},
    )
    moe_use_shared_expert_gate: bool = field(
        default=False,
        metadata={"help": "Use shared expert use sigmoid gate to control shared outputs."},
    )
    rotary_base: int = field(
        default=10000,
        metadata={"help": "Base period for rotary position embeddings. Defaults to 10000."},
    )
    rotary_percent: float = field(
        default=1,
        metadata={"help": "Percent of rotary dimension to use, default 1.0"},
    )
    rotary_scaling: bool = field(
        default=False,
        metadata={"help": "Apply rope scaling as used in llama 3.x."},
    )
    rotary_scaling_factor: float = field(
        default=8.0,
        metadata={
            "help": "The scaling factor applied to the inverse frequencies when "
            "1) the wavelength is greater than `low_freq_wavelen` prior to smoothing, "
            "2) to all inverse frequencies during smoothing."
        },
    )
    transformer_impl: Literal["local", "transformer_engine"] = field(
        default="transformer_engine",
        metadata={
            "help": "Which Transformer implementation to use.",
            "choices": ["local", "transformer_engine"],
        },
    )

    def __post_init__(self):
        if self.virtual_pipeline_model_parallel_size is None and self.overlap_p2p_comm:
            self.overlap_p2p_comm = False
            logger.warning("Non-interleaved pipeline parallelism does not support overlapping p2p communication!")

        self.deallocate_pipeline_outputs = True
        if self.swiglu:
            self.activation_func = F.silu
            self.gated_linear_unit = True

        if self.squared_relu:

            def squared_relu(x):
                return torch.pow(F.relu(x), 2)

            self.activation_func = squared_relu

        if self.fp16:
            self.params_dtype = torch.float16
        if self.bf16:
            self.params_dtype = torch.bfloat16
        self.pipeline_dtype = self.params_dtype
        self.batch_p2p_comm = not self.overlap_p2p_comm

        # Initialize Yarn RoPE parameters when position_embedding_type is "yarn"
        if self.position_embedding_type == "yarn":
            # Dynamically add Yarn config attributes only when using yarn
            if not hasattr(self, "yarn_beta_fast"):
                self.yarn_beta_fast = 32
            if not hasattr(self, "yarn_beta_slow"):
                self.yarn_beta_slow = 1
            if not hasattr(self, "yarn_rotary_scaling_factor"):
                self.yarn_rotary_scaling_factor = 4
            if not hasattr(self, "yarn_original_max_position_embeddings"):
                self.yarn_original_max_position_embeddings = 32768
            if not hasattr(self, "yarn_mscale"):
                self.yarn_mscale = 1
            if not hasattr(self, "yarn_mscale_all_dim"):
                self.yarn_mscale_all_dim = 0
            if not hasattr(self, "yarn_correction_range_round_to_int"):
                self.yarn_correction_range_round_to_int = True

        if (
            self.recompute_granularity == "full"
            and self.recompute_method is None
            and self.recompute_num_layers is None
        ):
            # default: all layers will do recomputation
            self.recompute_method = "uniform"
            self.recompute_num_layers = 1

        if self.tensor_model_parallel_size > 1 and self.expert_model_parallel_size > 1 and not self.sequence_parallel:
            logger.warning("When using expert parallelism and tensor parallelism, sequence parallelism must be used!")
            self.sequence_parallel = True
        if self.sequence_parallel and not self.tensor_model_parallel_size > 1:
            logger.warning("When tensor parallelism is not used, cannot use sequence parallelism!")
            self.sequence_parallel = False
        self.attention_backend = check_and_get_attention_backend_by_env(self.attention_backend)
        if self.num_moe_experts is not None and self.num_moe_experts >= 32 and self.moe_router_dtype is None:
            self.moe_router_dtype = "fp32"
            logger.warning(
                f"Using {self.moe_router_dtype} for moe_router_dtype, "
                "since num_moe_experts is large and moe_router_dtype not set."
            )
        if self.variable_seq_lengths and self.moe_token_dispatcher_type in ["allgather"]:
            if self.num_moe_experts is not None:
                logger.warning(
                    f"Token dispatcher type: {self.moe_token_dispatcher_type} does not support "
                    f"variable sequence length, use alltoall dispatcher instead."
                )
            self.moe_token_dispatcher_type = "alltoall"
        if isinstance(self.pipeline_model_parallel_layout, str) and not torch.distributed.is_initialized():
            # when pipeline_model_parallel_layout is str, dist.get_rank would be called
            self.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
                layout=self.pipeline_model_parallel_layout,
                pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            )

        super().__post_init__()
        pipeline_size = self.pipeline_model_parallel_size
        if self.virtual_pipeline_model_parallel_size is not None:
            pipeline_size *= self.virtual_pipeline_model_parallel_size
        num_layers = self.num_layers
        if self.account_for_embedding_in_pipeline_split:
            num_layers += 1
        if self.account_for_loss_in_pipeline_split:
            num_layers += 1
        if self.pipeline_model_parallel_layout is None and num_layers % pipeline_size != 0:
            raise ValueError(
                f"The number of layers ({num_layers}) must be a multiple of the pipeline_model_parallel_size"
                f" ({self.pipeline_model_parallel_size}) and virtual_pipeline_model_parallel_size "
                f"({self.virtual_pipeline_model_parallel_size})."
            )

    def distribute_config_match(self, other: "McaModelConfig"):
        if not isinstance(other, McaModelConfig):
            return False
        return all(
            [
                self.tensor_model_parallel_size == other.tensor_model_parallel_size,
                self.pipeline_model_parallel_size == other.pipeline_model_parallel_size,
                self.virtual_pipeline_model_parallel_size == other.virtual_pipeline_model_parallel_size,
                self.expert_model_parallel_size == other.expert_model_parallel_size,
                self.expert_tensor_parallel_size == other.expert_tensor_parallel_size,
                self.transformer_impl == other.transformer_impl,
                self.account_for_embedding_in_pipeline_split == other.account_for_embedding_in_pipeline_split,
                self.account_for_loss_in_pipeline_split == other.account_for_loss_in_pipeline_split,
            ]
        )


@dataclass
class MLAMcaModelConfig(McaModelConfig, MLATransformerConfig):
    multi_latent_attention: Optional[bool] = field(default=True, metadata={"help": "Whether use mla"})

    def __post_init__(self):
        super().__post_init__()
