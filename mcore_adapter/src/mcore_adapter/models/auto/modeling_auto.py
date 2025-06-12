import json
import os

from transformers import AutoConfig as HfAutoConfig
from transformers.configuration_utils import CONFIG_NAME as HF_CONFIG_NAME

from ...constants import MCA_CONFIG_NAME
from ...utils import get_logger
from ..model_factory import McaGPTModel, VirtualModels
from ..qwen2_5_vl import Qwen2_5_VLModel
from ..qwen2_vl import Qwen2VLModel


logger = get_logger(__name__)


MODEL_MAPPING_NAMES = {
    "llama": McaGPTModel,
    "qwen2": McaGPTModel,
    "qwen3": McaGPTModel,
    "qwen2_moe": McaGPTModel,
    "qwen3_moe": McaGPTModel,
    "qwen2_vl": Qwen2VLModel,
    "qwen2_5_vl": Qwen2_5_VLModel,
}


def get_model_cls(model_type) -> McaGPTModel:
    cls = MODEL_MAPPING_NAMES.get(model_type, None)
    if cls is None:
        logger.warning(f"No model found for model type {model_type}, use McaGPTModel!")
        cls = McaGPTModel
    return cls


class AutoModel:
    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        config_file = os.path.join(model_name_or_path, MCA_CONFIG_NAME)
        model_type = None
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            config_values = json.loads(text)
            model_type = config_values.get("hf_model_type")
        elif os.path.isfile(os.path.join(model_name_or_path, HF_CONFIG_NAME)):
            # from hf ckpt
            logger.info(f"Did not find {config_file}, loading HuggingFace config from {model_name_or_path}")
            hf_config = HfAutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model_type = hf_config.model_type

        if model_type is None:
            raise ValueError(f"No valid config found in {model_name_or_path}")
        model_class = get_model_cls(model_type)
        return model_class.from_pretrained(model_name_or_path, *args, **kwargs)

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        model_type = config.hf_model_type
        model_class = get_model_cls(model_type)
        return VirtualModels(model_class, config=config, *args, **kwargs)
