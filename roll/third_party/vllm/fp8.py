from typing import List
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

from roll.utils.fp8 import per_block_fp8_quant
from roll.utils.logging import get_logger

logger = get_logger()

def update_quant_config(config, vllm_config):
    # Use hf_overrides arguments with weight_block_size
    # to enable block quantization.
    # e.g.
    #   strategy_args:
    #     strategy_name: vllm
    #     strategy_config:
    #       quantization: fp8
    #       hf_overrides:
    #         quantization_config:
    #           activation_scheme: dynamic
    #           fmt: e4m3
    #           quant_method: fp8
    #           weight_block_size: [128, 128]
    if "hf_overrides" not in config or "quantization_config" not in config["hf_overrides"]:
        return
    assert config["hf_overrides"]["quantization_config"]["quant_method"] == "fp8"
    assert isinstance(vllm_config.quant_config, Fp8Config)
    assert vllm_config.quant_config.activation_scheme == "dynamic"
    assert vllm_config.quant_config.is_checkpoint_fp8_serialized
    vllm_config.quant_config.skip_process_weights_after_loading = True
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
    if not getattr(self.quant_config, "skip_process_weights_after_loading", False):
        return

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

    layer.register_parameter("input_scale", None)

_original_fp8_linear_create_weights = Fp8LinearMethod.create_weights
Fp8LinearMethod.create_weights = _fp8_linear_create_weights

def _fp8_linear_process_weights_after_loading(self, layer: Module) -> None:
    if not getattr(self.quant_config, "skip_process_weights_after_loading", False):
        _original_fp8_linear_process_weights_after_loading(self, layer)

_original_fp8_linear_process_weights_after_loading = Fp8LinearMethod.process_weights_after_loading
Fp8LinearMethod.process_weights_after_loading = _fp8_linear_process_weights_after_loading

def _fp8_moe_w13_weight_loader(layer: weakref.ReferenceType, original_weight_loader, param: torch.Tensor, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
    layer = layer()
    assert param is layer.w13_weight
    target_device = layer.w13_weight.device
    with target_device:
        loaded_weight = loaded_weight.to(target_device)
        if loaded_weight.dtype == torch.float8_e4m3fn:
            original_weight_loader(layer.w13_weight, loaded_weight, *args, **kwargs)
        else:
            qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
            original_weight_loader(layer.w13_weight, qweight, *args, **kwargs)
            original_weight_loader(layer.w13_weight_scale_inv, scale, *args, **kwargs)

def _fp8_moe_w2_weight_loader(layer: weakref.ReferenceType, original_weight_loader, param: torch.Tensor, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
    layer = layer()
    assert param is layer.w2_weight
    target_device = layer.w2_weight.device
    with target_device:
        loaded_weight = loaded_weight.to(target_device)
        if loaded_weight.dtype == torch.float8_e4m3fn:
            original_weight_loader(layer.w2_weight, loaded_weight, *args, **kwargs)
        else:
            qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
            original_weight_loader(layer.w2_weight, qweight, *args, **kwargs)
            original_weight_loader(layer.w2_weight_scale_inv, scale, *args, **kwargs)

def _fp8_moe_create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                   intermediate_size_per_partition: int,
                   params_dtype: torch.dtype, **extra_weight_attrs):
    _original_fp8_moe_create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                                     params_dtype, **extra_weight_attrs) 
    if not getattr(self.quant_config, "skip_process_weights_after_loading", False):
        return

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

    if getattr(self, "_setup_kernel", None):
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
        assert self.fp8_backend not in [
            Fp8MoeBackend.AITER,
            Fp8MoeBackend.MARLIN,
            Fp8MoeBackend.FLASHINFER_CUTLASS,
            Fp8MoeBackend.FLASHINFER_TRTLLM,
            # TODO: support inflight fp8 quantization for DEEPGEMM and BATCHED_DEEPGEMM
            Fp8MoeBackend.DEEPGEMM,
            Fp8MoeBackend.BATCHED_DEEPGEMM,
        ]
        assert self.fp8_backend in [
            Fp8MoeBackend.TRITON,
            Fp8MoeBackend.BATCHED_TRITON,
            Fp8MoeBackend.VLLM_CUTLASS,
            Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
            Fp8MoeBackend.XPU,
        ]

    # store essential config in layer for custom weight loader
    layer.weight_block_size = self.quant_config.weight_block_size

    w13_weight_loader = layer.w13_weight.weight_loader
    w13_weight_loader = partial(_fp8_moe_w13_weight_loader, weakref.ref(layer), w13_weight_loader)
    layer.w13_weight.weight_loader = w13_weight_loader
    set_weight_attrs(layer.w13_weight, {"roll_skip_patch_moe": True}) # TODO: remove once vllm 0.8.4 is deprecated

    w2_weight_loader = layer.w2_weight.weight_loader
    w2_weight_loader = partial(_fp8_moe_w2_weight_loader, weakref.ref(layer), w2_weight_loader)
    layer.w2_weight.weight_loader = w2_weight_loader
    set_weight_attrs(layer.w2_weight, {"roll_skip_patch_moe": True}) # TODO: remove once vllm 0.8.4 is deprecated

    # do not need patch weight loader of scale
    assert type(layer.w13_weight_scale_inv) == Parameter
    assert type(layer.w2_weight_scale_inv) == Parameter

_original_fp8_moe_create_weights = Fp8MoEMethod.create_weights
Fp8MoEMethod.create_weights = _fp8_moe_create_weights

def _fp8_moe_process_weights_after_loading(self, layer: Module) -> None:
    if not getattr(self.quant_config, "skip_process_weights_after_loading", False):
        _original_fp8_moe_process_weights_after_loading(self, layer)
    else:
        if getattr(self, "_setup_kernel", None):
            w13 = layer.w13_weight
            w2 = layer.w2_weight
            w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
            w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
            w13_input_scale = layer.w13_input_scale
            w2_input_scale = layer.w2_input_scale
            assert w13_input_scale is None
            assert w2_input_scale is None
            # _setup_kernel will change layer.w13_weight, layer.w2_weight
            self._setup_kernel(
                layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale
            )
            assert w13.data_ptr() == layer.w13_weight.data_ptr()
            assert w2.data_ptr() == layer.w2_weight.data_ptr()
            assert w13_scale.data_ptr() == getattr(layer, f"w13_{self.weight_scale_name}").data_ptr()
            assert w2_scale.data_ptr() == getattr(layer, f"w2_{self.weight_scale_name}").data_ptr()

_original_fp8_moe_process_weights_after_loading = Fp8MoEMethod.process_weights_after_loading
Fp8MoEMethod.process_weights_after_loading = _fp8_moe_process_weights_after_loading
