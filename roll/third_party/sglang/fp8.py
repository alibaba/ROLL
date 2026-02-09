from typing import Any, Dict, List
from functools import partial
import weakref

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.fp8 import (
    Fp8Config,
    _is_fp8_fnuz,
    _is_cpu,
    _is_hip,
    _use_hip_int4,
    _use_aiter,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

from roll.utils.fp8 import per_block_fp8_quant
from roll.utils.logging import get_logger

logger = get_logger()

def from_config(cls, config: Dict[str, Any]) -> Fp8Config:
    quant_method = cls.get_from_keys_or(config, ["quant_method"], "")
    is_checkpoint_fp8_serialized = "fp8" in quant_method
    activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
    ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
    weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
    skip_process_weights_after_loading = not is_checkpoint_fp8_serialized
    config = cls(
        is_checkpoint_fp8_serialized=True,
        activation_scheme=activation_scheme,
        ignored_layers=ignored_layers,
        weight_block_size=weight_block_size,
    )
    config.skip_process_weights_after_loading = skip_process_weights_after_loading
    return config

def monkey_patch_fp8_config():
    Fp8Config.from_config = classmethod(from_config)

def per_block_fp8_quant_ue8m0(
    weight: torch.Tensor,
    weight_block_size: List[int],
):
    from sglang.srt.layers.quantization.fp8_utils import (
        quant_weight_ue8m0,
        transform_scale_ue8m0,
    )
    assert weight_block_size == [128, 128]

    out_w, out_s = quant_weight_ue8m0(
        weight_dequant=weight,
        weight_block_size=weight_block_size,
    )

    out_s = transform_scale_ue8m0(out_s, mn=out_w.shape[-2])

    return out_w, out_s

def monkey_patch_fp8_linear_method():
    def f_weight_loader(
        layer: weakref.ReferenceType,
        original_weight_loader,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        *args,
        **kwargs
    ) -> None:
        layer = layer()
        assert param is layer.weight
        target_device = layer.weight.device
        with target_device:
            loaded_weight = loaded_weight.to(target_device)
            weight = ModelWeightParameter(
                                data=layer.weight.data if layer.weight_block_size else layer.weight.data.t(),
                                input_dim=1,
                                output_dim=0,
                                weight_loader=original_weight_loader,
                            )
            if loaded_weight.dtype == torch.float8_e4m3fn:
                original_weight_loader(weight, loaded_weight, *args, **kwargs)
            else:
                if layer.format_ue8m0:
                    qweight, scale = per_block_fp8_quant_ue8m0(loaded_weight, layer.weight_block_size)
                else:
                    qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
                weight_scale_inv = BlockQuantScaleParameter(
                                            data=layer.weight_scale_inv.data,
                                            input_dim=1,
                                            output_dim=0,
                                            weight_loader=original_weight_loader,
                                        )
                weight_scale_inv.format_ue8m0 = True
                original_weight_loader(weight, qweight, *args, **kwargs)
                original_weight_loader(weight_scale_inv, scale, *args, **kwargs)

    def f_weight_scale_loader(
        layer: weakref.ReferenceType,
        original_weight_loader,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        *args,
        **kwargs
    ) -> None:
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

    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    original_create_weights = Fp8LinearMethod.create_weights
    original_process_weights_after_loading = Fp8LinearMethod.process_weights_after_loading

    def f_create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        original_create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra_weight_attrs)
        assert self.quant_config.is_checkpoint_fp8_serialized
        assert self.block_quant, "only suuport block-wise quantization"
        assert self.quant_config.weight_block_size
        assert self.quant_config.activation_scheme == "dynamic"
        assert not _is_fp8_fnuz
        assert not _is_cpu
        assert layer.input_scale is None

        if self.quant_config.skip_process_weights_after_loading:
            try:
                from sglang.srt.layers.quantization.fp8_utils import (
                    requant_weight_ue8m0_inplace,
                    deepgemm_w8a8_block_fp8_linear_with_fallback,
                )
                from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
                # For fp8 linear weights run with deepgemm, the weights and scales need be requantized to ue8m0
                if (
                    should_deepgemm_weight_requant_ue8m0(self.quant_config.weight_block_size)
                    and self.w8a8_block_fp8_linear is deepgemm_w8a8_block_fp8_linear_with_fallback
                ):
                    requant_weight_ue8m0_inplace(layer.weight, layer.weight_scale_inv, self.quant_config.weight_block_size)
                    layer.format_ue8m0 = True
                else:
                    layer.format_ue8m0 = False
            except:
                layer.format_ue8m0 = False

        layer.weight_block_size = self.quant_config.weight_block_size

        weight_loader = layer.weight.weight_loader
        weight_loader = partial(f_weight_loader, weakref.ref(layer), weight_loader)
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight.weight_loader = weight_loader

        weight_scale_inv_loader = layer.weight_scale_inv.weight_loader
        weight_scale_inv_loader = partial(f_weight_scale_loader, weakref.ref(layer), weight_scale_inv_loader)
        weight_scale_inv = layer.weight_scale_inv
        layer.weight_scale_inv = Parameter(weight_scale_inv.data, requires_grad=False)
        layer.weight_scale_inv.format_ue8m0 = self.quant_config.skip_process_weights_after_loading and layer.format_ue8m0
        layer.weight_scale_inv.weight_loader = weight_scale_inv_loader

    def f_process_weights_after_loading(self, layer: Module) -> None:
        if not self.quant_config.skip_process_weights_after_loading:
            original_process_weights_after_loading(self, layer)

    Fp8LinearMethod.create_weights = f_create_weights
    Fp8LinearMethod.process_weights_after_loading = f_process_weights_after_loading

def monkey_patch_fp8_moe_method():
    def f_w13_weight_loader(
        layer: weakref.ReferenceType,
        original_weight_loader,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        *args,
        **kwargs
    ) -> None:
        layer = layer()
        assert param is layer.w13_weight
        target_device = layer.w13_weight.device
        with target_device:
            loaded_weight = loaded_weight.to(target_device)
            if loaded_weight.dtype == torch.float8_e4m3fn:
                original_weight_loader(layer.w13_weight, loaded_weight, *args, **kwargs)
            else:
                if layer.format_ue8m0:
                    qweight, scale = per_block_fp8_quant_ue8m0(loaded_weight, layer.weight_block_size)
                else:
                    qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
                original_weight_loader(layer.w13_weight, qweight, *args, **kwargs)
                original_weight_loader(layer.w13_weight_scale_inv, scale, *args, **kwargs)

    def f_w2_weight_loader(
        layer: weakref.ReferenceType,
        original_weight_loader,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        *args,
        **kwargs
    ) -> None:
        layer = layer()
        assert param is layer.w2_weight
        target_device = layer.w2_weight.device
        with target_device:
            loaded_weight = loaded_weight.to(target_device)
            if loaded_weight.dtype == torch.float8_e4m3fn:
                original_weight_loader(layer.w2_weight, loaded_weight, *args, **kwargs)
            else:
                if layer.format_ue8m0:
                    qweight, scale = per_block_fp8_quant_ue8m0(loaded_weight, layer.weight_block_size)
                else:
                    qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
                original_weight_loader(layer.w2_weight, qweight, *args, **kwargs)
                original_weight_loader(layer.w2_weight_scale_inv, scale, *args, **kwargs)

    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
    original_create_weights = Fp8MoEMethod.create_weights
    original_process_weights_after_loading = Fp8MoEMethod.process_weights_after_loading

    def f_create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        original_create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs)
        assert self.quant_config.is_checkpoint_fp8_serialized
        assert self.block_quant, "only suuport block-wise quantization"
        assert self.quant_config.weight_block_size
        assert self.quant_config.activation_scheme == "dynamic"
        assert not _is_fp8_fnuz
        assert not _is_cpu
        assert not (_is_hip and _use_hip_int4)
        assert not _use_aiter

        if self.quant_config.skip_process_weights_after_loading:
            try:
                from sglang.srt.layers.quantization.fp8_utils import (
                    requant_weight_ue8m0_inplace,
                )
                from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
                # For fp8 moe run with deepgemm, the expert weights and scales need be requantized to ue8m0
                if (
                    should_deepgemm_weight_requant_ue8m0(self.quant_config.weight_block_size)
                    and get_moe_runner_backend().is_deep_gemm()
                ):
                    assert isinstance(
                        layer, DeepEPMoE
                    ), "DeepGemm MoE is only supported with DeepEPMoE"
                    requant_weight_ue8m0_inplace(layer.w13_weight, layer.w13_weight_scale_inv, layer.weight_block_size)
                    requant_weight_ue8m0_inplace(layer.w2_weight, layer.w2_weight_scale_inv, layer.weight_block_size)
                    layer.format_ue8m0 = True
                else:
                    layer.format_ue8m0 = False
            except:
                layer.format_ue8m0 = False

        # store essential config in layer for custom weight loader
        layer.weight_block_size = self.quant_config.weight_block_size

        w13_weight_loader = layer.w13_weight.weight_loader
        w13_weight_loader = partial(f_w13_weight_loader, weakref.ref(layer), w13_weight_loader)
        layer.w13_weight.weight_loader = w13_weight_loader

        w2_weight_loader = layer.w2_weight.weight_loader
        w2_weight_loader = partial(f_w2_weight_loader , weakref.ref(layer), w2_weight_loader)
        layer.w2_weight.weight_loader = w2_weight_loader

        # do not need patch weight loader of scale
        assert type(layer.w13_weight_scale_inv) == Parameter
        assert type(layer.w2_weight_scale_inv) == Parameter

    def f_process_weights_after_loading(self, layer: Module) -> None:
        if not self.quant_config.skip_process_weights_after_loading:
            original_process_weights_after_loading(self, layer)

    Fp8MoEMethod.create_weights = f_create_weights
    Fp8MoEMethod.process_weights_after_loading = f_process_weights_after_loading

def monkey_patch_fp8():
    monkey_patch_fp8_config()
    monkey_patch_fp8_linear_method()
    monkey_patch_fp8_moe_method()
