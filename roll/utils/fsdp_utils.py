import copy
import dataclasses
from abc import ABC
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import Shard

from roll.models.model_providers import _is_moe_config
from roll.platforms import current_platform

try:
    from torch.distributed.device_mesh import DeviceMesh
except ImportError:
    DeviceMesh = None

fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


def get_shard_placement_fn(fsdp_size):
    """
    Choose the dimension that can divide fsdp_size to avoid padding
    Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py

    """

    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def _clone_mp_policy(mp_policy, **overrides):
    if mp_policy is None:
        return None

    if dataclasses.is_dataclass(mp_policy):
        return dataclasses.replace(mp_policy, **overrides)

    # Try reconstructing via constructor from common attributes.
    attrs = {}
    for k in ("param_dtype", "reduce_dtype", "output_dtype", "cast_forward_inputs"):
        if hasattr(mp_policy, k):
            attrs[k] = getattr(mp_policy, k)
    attrs.update(overrides)
    return mp_policy.__class__(**attrs)


def _fsdp_kwargs_for_module(fsdp_kwargs: dict, module: nn.Module) -> dict:
    """
    Allows overriding FSDP2 kwargs per module, e.g. disabling mp_policy.cast_forward_inputs
    for specific classes like VL blocks.
    """
    mp_policy = fsdp_kwargs.get("mp_policy", None)
    if mp_policy is None or not hasattr(mp_policy, "cast_forward_inputs"):
        return fsdp_kwargs

    attr_override = getattr(module, "_fsdp2_cast_forward_inputs", None)
    if attr_override is not None:
        desired = bool(attr_override)
    else:
        desired = False

    if desired == mp_policy.cast_forward_inputs:
        return fsdp_kwargs

    new_kwargs = dict(fsdp_kwargs)
    new_kwargs["mp_policy"] = _clone_mp_policy(mp_policy, cast_forward_inputs=desired)
    return new_kwargs


def apply_fsdp2(model, fsdp_kwargs, config, is_lora=False):
    """
    model: AutoModelForCausalLM

    Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    and LoRA Patch: https://github.com/volcengine/verl/issues/3470

    """
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    model_cfg = getattr(model, "config", None)
    is_moe = _is_moe_config(model_cfg)
    apply_expert_patch = bool(config.get("apply_expert_patch", False))

    if is_moe and apply_expert_patch:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        from roll.third_party.fsdp2.qwen3_moe_patch import qwen3_moe_forward

        Qwen3MoeSparseMoeBlock.forward = qwen3_moe_forward
        print("[apply_fsdp2] Applied expert patch for Qwen3MoeSparseMoeBlock")

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap",
        default_transformer_cls_names_to_wrap,
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    wrap_embeddings = bool(config.get("wrap_policy", {}).get("wrap_embeddings", False))
    wrap_lm_output = bool(config.get("wrap_policy", {}).get("wrap_lm_output", False))

    def _get_embed_tokens(m: nn.Module):
        inner = getattr(m, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            return getattr(inner, "embed_tokens")
        if hasattr(m, "embed_tokens"):
            return getattr(m, "embed_tokens")
        if hasattr(m, "get_input_embeddings"):
            return m.get_input_embeddings()
        return None

    def _already_fully_sharded(mod: nn.Module) -> bool:
        # `fully_shard()` mutates the module into an internal FSDPModule type. If so, do not re-apply.
        return fully_shard_module is not None and isinstance(mod, fully_shard_module.FSDPModule)

    lora_modules = []
    selected = []
    moe_modules = []
    for name, module in model.named_modules():
        if is_lora and (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            lora_modules.append(module)

        # PumpkinComment:
        #  (MoE): Do NOT FSDP-wrap individual experts by default.
        # Experts are invoked conditionally per-rank (based on routing),
        # so wrapping `experts.*` as separate FSDP modules can deadlock collectives when
        # different ranks activate different experts. Therefor we only wrap experts
        # if we apply the expert patch.
        if is_moe and config.get("apply_expert_patch", False):
            moe_block = config.get("wrap_policy", {}).get("moe_experts", None)
            if isinstance(moe_block, str):
                moe_block = [moe_block]
            if moe_block is not None and module.__class__.__name__ in moe_block:
                moe_modules.append(module)
                print("[apply_fsdp2] Wrapped MoE expert module: ", name, module.__class__.__name__)

        # If `wrap_embeddings` is enabled, embeddings are handled explicitly below to avoid double wrapping.
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            (not wrap_embeddings)
            and isinstance(module, nn.Embedding)
            and (not getattr(getattr(model, "config", None), "tie_word_embeddings", True))
        ):
            selected.append((name, module))

    # PumpkinComment:
    # Avoid wrapping both a parent module and its child module with the same mesh.
    selected_names = [n for n, _ in selected]
    non_leaf = set()
    for n in selected_names:
        if not n:
            continue
        parts = n.split(".")
        for i in range(1, len(parts)):
            non_leaf.add(".".join(parts[:i]))

    modules = [m for n, m in selected if n not in non_leaf]

    wrapped_ids = set()

    def _wrap_once(mod: nn.Module, kwargs: dict):
        if mod is None:
            return
        if id(mod) in wrapped_ids:
            return
        if _already_fully_sharded(mod):
            wrapped_ids.add(id(mod))
            return
        with maybe_patch_fsdp_module(mod):
            fully_shard(mod, **kwargs)
        wrapped_ids.add(id(mod))

    # 1. Embeddings
    if wrap_embeddings:
        _wrap_once(_get_embed_tokens(model), fsdp_kwargs)

    # 2. LoRA Modules (Linear Layer)
    for idx, module in enumerate(lora_modules):
        _wrap_once(module, fsdp_kwargs)

    # 3. MoE
    for idx, module in enumerate(moe_modules):
        _wrap_once(module, fsdp_kwargs)

    # 4. Transformers Layers
    for idx, module in enumerate(modules):
        _wrap_once(module, _fsdp_kwargs_for_module(fsdp_kwargs, module))

    # 5. LM Output
    if wrap_lm_output:
        _wrap_once(getattr(model, "lm_head", None), fsdp_kwargs)

    # Root wrap last for remaining modules. (FSDP2 will not reshard_after_forward for the root module.)
    root_kwargs = dict(fsdp_kwargs)
    root_kwargs["mp_policy"] = _clone_mp_policy(root_kwargs.get("mp_policy", None), cast_forward_inputs=False)
    _wrap_once(model, root_kwargs)


def fsdp2_load_full_state_dict(
    model: torch.nn.Module,
    full_state: dict,
    device_mesh=None,
    cpu_offload=None,
):
    """
    Reference: https://github1s.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py

    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """

    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    device_id = current_platform.current_device()

    if dist.get_rank() == 0:
        model = model.to(device=device_id, non_blocking=True)
    else:
        model = model.to_empty(device=device_id)

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=cpu_offload,
        broadcast_from_rank0=True,
    )
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        # Ensure model is on CPU but buffers are on GPU for FSDP2 CPU offload
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(device_id)
