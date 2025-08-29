from typing import List, Optional
from functools import partial
import weakref

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config, Fp8LinearMethod, Fp8MoEMethod)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform
from vllm.model_executor.utils import set_weight_attrs
from vllm._custom_ops import scaled_fp8_quant as per_tensor_fp8_quant
from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale

from roll.utils.logging import get_logger

logger = get_logger()

# Block quant operator
#
# Borrow from transformers
#   https://huggingface.co/docs/transformers/en/quantization/finegrained_fp8
#   https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/quantizers/quantizer_finegrained_fp8.py#L83
#
# May use op from torchao:
#   https://github.com/pytorch/ao/pull/1668
#   https://github.com/volcengine/verl/pull/3084
def per_block_fp8_quant(param_value: torch.Tensor, weight_block_size: List[int]):
    """
    Quantizes weights to FP8 format using Block-wise quantization
    """
    # Get FP8 min/max values
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    block_size_m, block_size_n = weight_block_size

    rows, cols = param_value.shape[-2:]

    if rows % block_size_m != 0 or cols % block_size_n != 0:
        raise ValueError(
            f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
        )
    param_value_orig_shape = param_value.shape

    param_value = param_value.reshape(
        -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
    ).permute(0, 1, 3, 2, 4)

    # Calculate scaling factor for each block
    max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
    scale = fp8_max / max_abs
    scale_orig_shape = scale.shape
    scale = scale.unsqueeze(-1).unsqueeze(-1)

    # Quantize the weights
    quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
    # Reshape back to matrix shape
    quantized_param = quantized_param.reshape(param_value_orig_shape)

    # Construct the final, correct shape for the scales
    num_row_blocks = rows // block_size_m
    num_col_blocks = cols // block_size_n
    # This preserves original batch dimensions, if any
    final_scale_shape = (*param_value_orig_shape[:-2], num_row_blocks, num_col_blocks)
    # Reshape directly to the correct shape and take the reciprocal
    scale = scale.reshape(final_scale_shape).reciprocal()

    # TODO: DeepGemm scales need to be transposed and aligned (said in vLLM fp8.py)?

    # TODO: On B200, DeepGemm only support E8M0 scale

    return quantized_param, scale

def update_quant_config(vllm_config):
    # Use hf_overrides arguments of LLM with weight_block_size
    # to enable block quantization.
    # e.g.
    #   strategy_args:
    #     strategy_name: vllm
    #     strategy_config:
    #       hf_overrides:
    #         quantization_config:
    #           activation_scheme: dynamic
    #           quant_method: fp8
    #           weight_block_size: [128, 128]
    if not vllm_config.quant_config:
        return
    if not isinstance(vllm_config.quant_config, Fp8Config):
        return

    assert vllm_config.quant_config.activation_scheme == "dynamic"
    vllm_config.quant_config.is_checkpoint_fp8_serialized = True
    logger.info(f"Using custom vLLM quantization, block size {vllm_config.quant_config.weight_block_size}")

def _fp8_linear_weight_loader(layer: weakref.ReferenceType, original_weight_loader, param: torch.Tensor, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
    layer = layer()
    assert param is layer.weight
    target_device = layer.weight.device
    with target_device:
        weight = ModelWeightParameter(
                            data=layer.weight.data if layer.weight_block_size else layer.weight.data.t(),
                            input_dim=1,
                            output_dim=0,
                            weight_loader=original_weight_loader,
                        )
        if loaded_weight.dtype == torch.float8_e4m3fn:
            original_weight_loader(weight, loaded_weight, *args, **kwargs)
        else:
            loaded_weight = loaded_weight.to(target_device)
            if layer.weight_block_size:
                weight_scale_inv = BlockQuantScaleParameter(
                                            data=layer.weight_scale_inv.data,
                                            input_dim=1,
                                            output_dim=0,
                                            weight_loader=original_weight_loader,
                                        )
                qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
                original_weight_loader(weight, qweight, *args, **kwargs)
                original_weight_loader(weight_scale_inv, scale, *args, **kwargs)
            else:
                qweight, scale = per_tensor_fp8_quant(loaded_weight, scale=None)
                original_weight_loader(weight, qweight, *args, **kwargs)
                original_weight_loader(layer.per_shard_scale, scale, *args, **kwargs)
                layer.shard_loaded += 1
                if layer.shard_loaded == layer.shard_num:
                    weight_scale, weight = requantize_with_max_scale(
                        weight=layer.weight.t(),
                        weight_scale=layer.per_shard_scale,
                        logical_widths=layer.logical_widths,
                    )
                    layer.weight.copy_(weight.t())
                    layer.weight_scale.copy_(weight_scale)
                    layer.shard_loaded = 0


def _fp8_linear_weight_scale_loader(layer: weakref.ReferenceType, original_weight_loader, param: torch.Tensor, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
    layer = layer()
    assert param is layer.weight_scale_inv
    target_device = layer.weight_scale_inv.device
    with target_device:
        weight_scale_inv = BlockQuantScaleParameter(
                                    data=layer.weight_scale_inv.data,
                                    input_dim=1,
                                    output_dim=0,
                                    weight_loader=original_weight_loader,
                                )
        original_weight_loader(weight_scale_inv, loaded_weight, *args, **kwargs)

def _fp8_linear_create_weights(
    self,
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: List[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    _original_fp8_linear_create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                                   input_size, output_size, params_dtype, **extra_weight_attrs)

    assert self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"
    assert not self.use_marlin # not implement yet, because lack weight loader for chanelwise weight_scale

    # TODO support ROCM
    assert not current_platform.is_rocm()
    assert not current_platform.is_fp8_fnuz()

    # store essential config in layer for custom weight loader
    layer.weight_block_size = self.quant_config.weight_block_size

    weight_loader = layer.weight.weight_loader
    weight_loader = partial(_fp8_linear_weight_loader, weakref.ref(layer), weight_loader) # patch weight loader
    layer.weight = Parameter(layer.weight.data, requires_grad=False) if layer.weight_block_size else Parameter(layer.weight.data.t(), requires_grad=False)
    layer.weight.weight_loader = weight_loader

    if layer.weight_block_size:
        weight_scale_inv_loader = layer.weight_scale_inv.weight_loader
        weight_scale_inv_loader = partial(_fp8_linear_weight_scale_loader, weakref.ref(layer), weight_scale_inv_loader)
        layer.weight_scale_inv = Parameter(layer.weight_scale_inv.data, requires_grad=False)
        layer.weight_scale_inv.weight_loader = weight_scale_inv_loader
    else:
        # does not support is_checkpoint_fp8_serialized now
        layer.per_shard_scale = layer.weight_scale
        layer.weight_scale = Parameter(torch.zeros(1, device=layer.weight.device, dtype=torch.float32), requires_grad=False)
        layer.shard_num = len(output_partition_sizes)
        layer.shard_loaded = 0

_original_fp8_linear_create_weights = Fp8LinearMethod.create_weights
Fp8LinearMethod.create_weights = _fp8_linear_create_weights

def _fp8_linear_process_weights_after_loading(self, layer: Module) -> None:
    pass

Fp8LinearMethod.process_weights_after_loading = _fp8_linear_process_weights_after_loading

def _fp8_moe_w13_weight_loader(layer: weakref.ReferenceType, original_weight_loader, param: torch.Tensor, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
    layer = layer()
    assert param is layer.w13_weight
    target_device = layer.w13_weight.device
    with target_device:
        loaded_weight = loaded_weight.to(target_device)
        qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
        original_weight_loader(layer.w13_weight, qweight, *args, **kwargs)
        original_weight_loader(layer.w13_weight_scale_inv, scale, *args, **kwargs)

def _fp8_moe_w2_weight_loader(layer: weakref.ReferenceType, original_weight_loader, param: torch.Tensor, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
    layer = layer()
    assert param is layer.w2_weight
    target_device = layer.w2_weight.device
    with target_device:
        loaded_weight = loaded_weight.to(target_device)
        qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
        original_weight_loader(layer.w2_weight, qweight, *args, **kwargs)
        original_weight_loader(layer.w2_weight_scale_inv, scale, *args, **kwargs)

def _fp8_moe_create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                   intermediate_size_per_partition: int,
                   params_dtype: torch.dtype, **extra_weight_attrs):
    _original_fp8_moe_create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                                     params_dtype, **extra_weight_attrs) 

    assert self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"
    assert self.quant_config.weight_block_size is not None

    # TODO support ROCM
    # https://github.com/vllm-project/vllm/blob/v0.8.4/vllm/model_executor/layers/quantization/fp8.py#L655
    assert not current_platform.is_rocm()
    assert not current_platform.is_fp8_fnuz()
    assert current_platform.fp8_dtype() == torch.float8_e4m3fn

    self.rocm_aiter_moe_enabled = False # set in original process_weights_after_loading

    # TODO: support ep
    assert layer.local_num_experts == num_experts

    # store essential config in layer for custom weight loader
    layer.weight_block_size = self.quant_config.weight_block_size

    w13_weight_loader = layer.w13_weight.weight_loader
    w13_weight_loader = partial(_fp8_moe_w13_weight_loader, weakref.ref(layer), w13_weight_loader)
    layer.w13_weight.weight_loader = w13_weight_loader
    set_weight_attrs(layer.w13_weight, {"roll_skip_patch_moe": True})

    w2_weight_loader = layer.w2_weight.weight_loader
    w2_weight_loader = partial(_fp8_moe_w2_weight_loader, weakref.ref(layer), w2_weight_loader)
    layer.w2_weight.weight_loader = w2_weight_loader
    set_weight_attrs(layer.w2_weight, {"roll_skip_patch_moe": True})

    # do not need patch weight loader of scale
    assert type(layer.w13_weight_scale_inv) == Parameter
    assert type(layer.w2_weight_scale_inv) == Parameter

_original_fp8_moe_create_weights = Fp8MoEMethod.create_weights
Fp8MoEMethod.create_weights = _fp8_moe_create_weights

def _fp8_moe_process_weights_after_loading(self, layer: Module) -> None:
    pass

Fp8MoEMethod.process_weights_after_loading = _fp8_moe_process_weights_after_loading
