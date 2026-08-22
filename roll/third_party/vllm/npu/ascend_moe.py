"""Qwen3.5 MoE weight loading fixes for vLLM-Ascend.

The Ascend unquantized MoE implementation transposes its weights during
``process_weights_after_loading``.  RL weight updates happen after the first
load, so the loader must write the incoming checkpoint shards into the
already-transposed, TP/EP-local tensors and the post-processing transpose must
not be applied a second time.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Mapping

import torch


def _is_w13(name: str) -> bool:
    return "w13_weight" in name


def _is_w2(name: str) -> bool:
    return "w2_weight" in name


def _find_param(params_dict: Mapping[str, Any], name: str) -> Any:
    param = params_dict.get(name)
    if param is not None:
        return param
    for candidate_name, candidate in params_dict.items():
        if candidate_name.endswith(name) or name.endswith(candidate_name):
            return candidate
    return None


def _get_tp_context() -> tuple[int, int]:
    """Support both vLLM's old and new tensor-parallel helper names."""

    try:
        from vllm.distributed import get_tp_rank, get_tp_world_size

        return get_tp_rank(), get_tp_world_size()
    except ImportError:
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        except ImportError:
            # This is only used by lightweight unit tests without vLLM's
            # distributed runtime; production NPU workers always provide one.
            return 0, 1


def _map_global_expert_to_local(experts: Any, global_expert_id: int) -> int:
    """Return the local expert index, or ``-1`` for an EP-nonlocal expert."""

    for method_name in (
        "_map_global_expert_id_to_local_expert_id",
        "map_global_expert_id_to_local_expert_id",
    ):
        mapper = getattr(experts, method_name, None)
        if mapper is not None:
            return int(mapper(global_expert_id))

    manager = getattr(experts, "expert_map_manager", None)
    if manager is not None:
        for method_name in (
            "map_global_to_local",
            "map_global_expert_id_to_local_expert_id",
        ):
            mapper = getattr(manager, method_name, None)
            if mapper is not None:
                return int(mapper(global_expert_id))

    expert_map = getattr(experts, "expert_map", None)
    if expert_map is not None:
        try:
            return int(expert_map[global_expert_id])
        except (IndexError, KeyError, TypeError):
            return -1

    # Models without expert parallelism use global IDs directly.
    return global_expert_id


def _copy_transposed_expert_shard(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
    tp_rank: int,
    tp_world_size: int,
) -> bool:
    """Copy one checkpoint expert shard into a transposed local parameter.

    ``loaded_weight`` contains one fused expert shard per global expert.  The
    source layout is ``[experts, I, H]`` for ``w1``/``w3`` and ``[experts, H,
    I]`` for ``w2``.  Ascend stores the corresponding local parameter as
    ``[local_experts, H, I_tp]`` (fused ``w13``) or ``[local_experts, I_tp,
    H]`` (``w2``).
    """

    if local_expert_id < 0:
        return False
    if loaded_weight.ndim != 3:
        raise ValueError(
            f"Qwen3.5 fused expert weight must be rank-3, got shape {tuple(loaded_weight.shape)}"
        )
    if local_expert_id >= param.shape[0]:
        raise ValueError(
            "Qwen3.5 EP expert mapping produced local expert index "
            f"{local_expert_id}, but parameter has {param.shape[0]} local experts"
        )
    if tp_world_size <= 0:
        raise ValueError(f"Invalid TP world size {tp_world_size}")
    source_width = loaded_weight.shape[-1]
    if _is_w13(shard_id):
        source_width = loaded_weight.shape[-2]
    if source_width % tp_world_size:
        raise ValueError(
            "Qwen3.5 expert weight last dimension "
            f"{source_width} is not divisible by TP world size {tp_world_size}"
        )

    shard_width = source_width // tp_world_size
    start = tp_rank * shard_width
    end = start + shard_width
    if tp_rank < 0 or end > source_width:
        raise ValueError(f"Invalid TP rank {tp_rank} for world size {tp_world_size}")

    if loaded_weight.shape[0] != 1:
        raise ValueError(
            "_copy_transposed_expert_shard expects one global expert at a time, "
            f"got shape {tuple(loaded_weight.shape)}"
        )
    with torch.no_grad():
        if _is_w2(shard_id):
            # [H, I_tp] -> [I_tp, H]
            source = loaded_weight[0, ..., start:end]
            target = source.transpose(-1, -2).contiguous()
            expected = tuple(param.shape[1:])
            if tuple(target.shape) != expected:
                raise ValueError(
                    f"Qwen3.5 w2 shard shape {tuple(target.shape)} does not match local "
                    f"parameter shape {expected}"
                )
            param.data[local_expert_id].copy_(target)
        elif _is_w13(shard_id):
            # [I, H] -> [H, I_tp], then place into the w1/w3 half.
            source = loaded_weight[0].transpose(-1, -2).contiguous()
            target = source[..., start:end]
            half_width = param.shape[-1] // 2
            if target.shape[-2] != param.shape[-2] or target.shape[-1] != half_width:
                raise ValueError(
                    f"Qwen3.5 {shard_id} shard shape {tuple(target.shape)} does not match "
                    f"local fused parameter shape {tuple(param.shape)}"
                )
            offset = 0 if shard_id == "w1" else half_width
            param.data[local_expert_id, :, offset : offset + half_width].copy_(target)
        else:
            raise ValueError(f"Unsupported Qwen3.5 fused expert shard ID: {shard_id!r}")
    return True


def _patch_transposed_fused_expert_loader() -> None:
    """Patch Qwen3.5's fused loader once, keeping the original fallback."""

    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    if getattr(Qwen3_5Model, "_roll_transposed_expert_loader_patched", False):
        return

    original = Qwen3_5Model.load_fused_expert_weights

    def load_fused_expert_weights(
        self,
        name,
        params_dict,
        loaded_weight,
        shard_id,
        num_experts,
        *args,
        **kwargs,
    ):
        param = _find_param(params_dict, name)
        experts = getattr(param, "_roll_expert_module", None) if param is not None else None
        if param is None or experts is None or not getattr(param, "_roll_ascend_transposed", False):
            return original(self, name, params_dict, loaded_weight, shard_id, num_experts, *args, **kwargs)

        if loaded_weight.shape[0] != num_experts:
            raise ValueError(
                f"Qwen3.5 fused expert weight has {loaded_weight.shape[0]} experts, "
                f"expected {num_experts}"
            )

        tp_rank, tp_world_size = _get_tp_context()

        copied = False
        for global_expert_id in range(num_experts):
            local_expert_id = _map_global_expert_to_local(experts, global_expert_id)
            copied = (
                _copy_transposed_expert_shard(
                    param,
                    loaded_weight[global_expert_id : global_expert_id + 1],
                    shard_id,
                    local_expert_id,
                    tp_rank,
                    tp_world_size,
                )
                or copied
            )
        if copied:
            param._roll_already_transposed = True
        return True

    Qwen3_5Model.load_fused_expert_weights = load_fused_expert_weights
    Qwen3_5Model._roll_transposed_expert_loader_patched = True


def _iter_text_layers(model: Any):
    language_model = getattr(model, "language_model", None)
    root = language_model if language_model is not None else model
    nested_model = getattr(root, "model", root)
    layers = getattr(nested_model, "layers", None)
    if layers is None:
        layers = getattr(getattr(nested_model, "language_model", None), "layers", None)
    return layers or ()


def _is_ascend_unquantized(experts: Any) -> bool:
    try:
        from vllm_ascend.quantization.unquantized_fused_moe import AscendUnquantizedFusedMoEMethod
    except (ImportError, AttributeError):
        try:
            from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
        except (ImportError, AttributeError):
            return False

    quant_method = getattr(experts, "quant_method", None)
    quant_method = getattr(quant_method, "quant_method", quant_method)
    return isinstance(quant_method, AscendUnquantizedFusedMoEMethod)


def _patch_process_weights_after_loading(model: Any) -> None:
    """Skip only the duplicate transpose on patched unquantized MoE modules."""

    for module in model.modules():
        if not _is_ascend_unquantized(module):
            continue
        quant_method = getattr(module, "quant_method", None)
        quant_method = getattr(quant_method, "quant_method", quant_method)
        process = getattr(quant_method, "process_weights_after_loading", None)
        if process is None or getattr(quant_method, "_roll_process_patched", False):
            continue

        original = process

        def process_weights_after_loading(self, layer, *args, _original=original, **kwargs):
            params = dict(layer.named_parameters())
            w13 = next((p for n, p in params.items() if _is_w13(n)), None)
            w2 = next((p for n, p in params.items() if _is_w2(n)), None)
            if (
                w13 is not None
                and w2 is not None
                and getattr(w13, "_roll_already_transposed", False)
                and getattr(w2, "_roll_already_transposed", False)
            ):
                return
            return _original(layer, *args, **kwargs)

        quant_method.process_weights_after_loading = MethodType(process_weights_after_loading, quant_method)
        quant_method._roll_process_patched = True


def patch_ascend_moe_weight_loader(model: Any) -> None:
    """Install the Qwen3.5 NPU loader only for Ascend unquantized MoE layers."""

    _patch_transposed_fused_expert_loader()
    patched = False
    for layer in _iter_text_layers(model):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None or not _is_ascend_unquantized(experts):
            continue
        patched = True
        for name, param in mlp.named_parameters():
            if getattr(param, "roll_skip_patch_moe", False):
                continue
            if _is_w13(name) or _is_w2(name):
                param.is_transposed = True
                param._roll_ascend_transposed = True
                param._roll_expert_module = experts
                param.weight_loader = experts.weight_loader

    if patched:
        _patch_process_weights_after_loading(model)
