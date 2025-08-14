# patch qwen3 fp8
# https://github.com/vllm-project/vllm/issues/17327
# https://github.com/vllm-project/vllm/pull/17318

from vllm.model_executor.layers.linear import QKVParallelLinear

from typing import Optional
import torch
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           BlockQuantScaleParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)

def weight_loader_v2(self,
                     param: BasevLLMParameter,
                     loaded_weight: torch.Tensor,
                     loaded_shard_id: Optional[str] = None):
    if loaded_shard_id is None:  # special case for certain models
        if isinstance(param, PerTensorScaleParameter):
            param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
            return
        elif type(param) in (RowvLLMParameter, BasevLLMParameter):
            param.load_qkv_weight(loaded_weight=loaded_weight)
            return
        # TODO: @dsikka - move to parameter.py
        self._load_fused_module_from_checkpoint(param, loaded_weight)
        return

    assert loaded_shard_id in ["q", "k", "v"]

    shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
    shard_size = self._get_shard_size_mapping(loaded_shard_id)

    # Note(simon): This is needed for Qwen3's fp8 quantization.
    if isinstance(param, BlockQuantScaleParameter):
        assert self.quant_method is not None
        assert hasattr(self.quant_method, "quant_config")
        weight_block_size = self.quant_method.quant_config.weight_block_size
        block_n, _ = weight_block_size[0], weight_block_size[1]
        shard_offset = (shard_offset + block_n - 1) // block_n
        shard_size = (shard_size + block_n - 1) // block_n

    param.load_qkv_weight(loaded_weight=loaded_weight,
                          num_heads=self.num_kv_head_replicas,
                          shard_id=loaded_shard_id,
                          shard_offset=shard_offset,
                          shard_size=shard_size)

QKVParallelLinear.weight_loader_v2 = weight_loader_v2

__all__ = []
