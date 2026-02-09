from dataclasses import field
from typing import List

from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager


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

    for layer in model.model.layers:
        mlp = getattr(layer, "mlp")
        param_dict = dict(mlp.named_parameters())
        for name, param in param_dict.items():
            skip_patch = getattr(param, "roll_skip_patch_moe", False)
            if ("w13_weight" in name or "w2_weight" in name) and not skip_patch:
                param.weight_loader = mlp.experts.weight_loader


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


def patch_vllm_lora_manager():
    def load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
        """
        based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

        Reason:
        VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
        To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to load memory-based LoRA tensors.
        """
        try:
            from packaging.version import Version
            from vllm import __version__ as vllm_version

            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_lst: list[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_lst.extend(packed_modules_mapping[module])
                else:
                    expected_lora_lst.append(module)
                if module == "experts":
                    expected_lora_lst.append(module)

            expected_lora_modules = list(set(expected_lora_lst))

            lora_tensors = None
            from vllm.lora.peft_helper import PEFTHelper

            if isinstance(lora_request, TensorLoRARequest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                kwargs = {}
                if Version(vllm_version) > Version("0.8.4"):
                    kwargs["tensorizer_config_dict"] = lora_request.tensorizer_config_dict
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                peft_helper = PEFTHelper.from_local_dir(
                    lora_path,
                    self.max_position_embeddings,
                    **kwargs,
                )

            # Validates the LoRA configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.lora_config)

            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
            # to ensure correct loading of lora weights.
            model = self._adapter_manager.model
            hf_to_vllm_mapper = None
            if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                hf_to_vllm_mapper = model.hf_to_vllm_mapper

            if isinstance(lora_request, TensorLoRARequest):
                kwargs = {}
                if Version(vllm_version) >= Version("0.12.0"):
                    kwargs["model_vocab_size"] = self.vocab_size
                else:
                    kwargs["embeddings"] = None
                    kwargs["target_embedding_padding"] = self.vocab_size + self.lora_config.lora_extra_vocab_size
                    kwargs["embedding_modules"] = self.embedding_modules
                    kwargs["embedding_padding_modules"] = self.embedding_padding_modules
                lora = self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    weights_mapper=hf_to_vllm_mapper,
                    **kwargs,
                )
            else:
                kwargs = {}
                if Version(vllm_version) > Version("0.8.4"):
                    kwargs["tensorizer_config_dict"] = lora_request.tensorizer_config_dict
                if Version(vllm_version) >= Version("0.12.0"):
                    kwargs["model_vocab_size"] = self.vocab_size
                else:
                    kwargs["target_embedding_padding"] = self.vocab_size + self.lora_config.lora_extra_vocab_size
                    kwargs["embedding_modules"] = self.embedding_modules
                    kwargs["embedding_padding_modules"] = self.embedding_padding_modules
                lora = self._lora_model_cls.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    weights_mapper=hf_to_vllm_mapper,
                    **kwargs,
                )
        except Exception as e:
            raise e

        return lora

    setattr(LRUCacheWorkerLoRAManager, "_load_adapter", load_adapter)
