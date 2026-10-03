from dataclasses import field
import inspect
from typing import List
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from roll.third_party.vllm.compat import import_attribute, patch_moe_model_weight_loaders


LoRAModel = import_attribute(("vllm.lora.lora_model", "vllm.lora.models"), "LoRAModel")


# TODO: remove this patch once vllm 0.8.4 is deprecated
# Patch weight loader for moe models
# borrow from https://github.com/volcengine/verl/blob/main/verl/utils/vllm_utils.py
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)):
        return
    patch_moe_model_weight_loaders(model)


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


def _lora_loading_kwargs(loader, manager, request):
    """Select native loader options by signature, including development builds."""
    parameters = inspect.signature(loader).parameters
    kwargs = {}
    if "model_vocab_size" in parameters:
        kwargs["model_vocab_size"] = manager.vocab_size
    if "embeddings" in parameters:
        kwargs["embeddings"] = None
    if "target_embedding_padding" in parameters:
        kwargs["target_embedding_padding"] = manager.vocab_size + manager.lora_config.lora_extra_vocab_size
    if "embedding_modules" in parameters:
        kwargs["embedding_modules"] = manager.embedding_modules
    if "embedding_padding_modules" in parameters:
        kwargs["embedding_padding_modules"] = manager.embedding_padding_modules
    if "tensorizer_config_dict" in parameters:
        kwargs["tensorizer_config_dict"] = getattr(request, "tensorizer_config_dict", None)
    if "skip_prefixes" in parameters:
        kwargs["skip_prefixes"] = getattr(manager._adapter_manager.model, "lora_skip_prefixes", None)
    if "moe_ep_spec" in parameters:
        kwargs["moe_ep_spec"] = getattr(manager._adapter_manager, "moe_ep_load_spec", None)
    return kwargs


def patch_vllm_lora_manager():
    def load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
        """Load ROLL's in-memory adapters with the native vLLM loader contract."""
        from vllm.lora.peft_helper import PEFTHelper

        tensor_request = isinstance(lora_request, TensorLoRARequest)
        if tensor_request:
            peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
        else:
            lora_path = get_adapter_absolute_path(lora_request.lora_path)
            helper_kwargs = {}
            if "tensorizer_config_dict" in inspect.signature(PEFTHelper.from_local_dir).parameters:
                helper_kwargs["tensorizer_config_dict"] = getattr(lora_request, "tensorizer_config_dict", None)
            peft_helper = PEFTHelper.from_local_dir(
                lora_path, self.max_position_embeddings, **helper_kwargs
            )
        peft_helper.validate_legal(self.lora_config)

        model = self._adapter_manager.model
        weights_mapper = getattr(model, "hf_to_vllm_mapper", None)
        # Native LoRA packing needs constituent names to survive the mapper;
        # retain genuine prefix/name conversions but omit QKV/MLP fusion maps.
        unstack = getattr(weights_mapper, "get_unstacked_mapper", None)
        if callable(unstack):
            weights_mapper = unstack()

        loader = (self._lora_model_cls.from_lora_tensors if tensor_request
                  else self._lora_model_cls.from_local_checkpoint)
        kwargs = _lora_loading_kwargs(loader, self, lora_request)
        kwargs.update(lora_model_id=lora_request.lora_int_id, peft_helper=peft_helper,
                      device="cpu", dtype=self.lora_config.lora_dtype, weights_mapper=weights_mapper)
        if tensor_request:
            lora = loader(tensors=lora_request.lora_tensors, **kwargs)
        else:
            expected_modules = set()
            packed = self._adapter_manager.packed_modules_mapping
            for module in self._adapter_manager.supported_lora_modules:
                expected_modules.update(packed.get(module, [module]))
                if module == "experts":
                    expected_modules.add(module)
            lora = loader(lora_path, list(expected_modules), **kwargs)
        if hasattr(lora_request, "is_3d_lora_weight"):
            lora.is_3d_lora_weight = lora_request.is_3d_lora_weight
        return lora

    setattr(LRUCacheWorkerLoRAManager, "_load_adapter", load_adapter)
