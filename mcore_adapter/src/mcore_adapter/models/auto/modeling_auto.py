import json
import os
from collections import OrderedDict

from transformers import AutoConfig as HfAutoConfig
from transformers.configuration_utils import CONFIG_NAME as HF_CONFIG_NAME

from ...constants import MCA_CONFIG_NAME
from ...utils import get_logger
from ..model_factory import McaGPTModel, VirtualModels


logger = get_logger(__name__)


MODEL_MAPPING = OrderedDict()


def register_model(model_type, cls=None):
    def decorator(cls):
        if model_type in MODEL_MAPPING:
            logger.warning(f"Model for model type {model_type} already registered, overriding!")
        MODEL_MAPPING[model_type] = cls
        return cls

    if cls is not None:
        return decorator(cls)
    return decorator


def get_model_cls(model_type) -> "McaGPTModel":
    cls = MODEL_MAPPING.get(model_type)
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
