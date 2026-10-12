import ctypes
import gc
import sys
from enum import Enum
from typing import List, Tuple, Union

import torch
from torch import Tensor
from transformers import PreTrainedModel
from roll.platforms import current_platform


def clear_memory(clear_host_memory: bool = False):
    """Clear GPU and CPU memory caches.

    Synchronizes CUDA, runs Python GC, and clears the GPU memory cache.
    Optionally releases idle pinned host allocations and unused glibc heap
    pages. The latter can retain large HF conversion and optimizer setup
    temporaries even after their tensors have been freed.
    """
    current_platform.synchronize()
    gc.collect()
    current_platform.empty_cache()
    if clear_host_memory:
        memory = getattr(getattr(torch, "accelerator", None), "memory", None)
        empty_host_cache = getattr(memory, "empty_host_cache", None)
        if empty_host_cache is None:
            empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
        if empty_host_cache is not None:
            empty_host_cache()
        if sys.platform == "linux":
            # malloc_trim only releases free pages; live tensors and pinned
            # allocations retain their storage. Other allocators may omit it.
            try:
                trim = ctypes.CDLL(None).malloc_trim
            except (AttributeError, OSError):
                return
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)


class OffloadStateType(str, Enum):
    """
    offload/reload需要区分训练和推理阶段，在actor_train/critic用于计算log_probs和values时，只需要reload model_params
    """

    model_params = "model_params"
    optimizer_states = "optimizer_states"
    other_params = "other_params"


def offload_hf_model(model: PreTrainedModel):
    """
    根据 hf_device_map 将模型的各个层卸载到 CPU
    """
    from trl import AutoModelForCausalLMWithValueHead
    if isinstance(model, AutoModelForCausalLMWithValueHead):
        offload_hf_model(model=model.pretrained_model)
        offload_hf_model(model=model.v_head)
        return
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        model.to("cpu")
    else:
        [model.get_submodule(layer_name).to("cpu") for layer_name, device_id in device_map.items()]


def load_hf_model(model: PreTrainedModel):
    """
    根据 hf_device_map 将模型的各个层卸载到 对应的GPU
    """
    from trl import AutoModelForCausalLMWithValueHead
    if isinstance(model, AutoModelForCausalLMWithValueHead):
        load_hf_model(model=model.pretrained_model)
        load_hf_model(model=model.v_head)
        return
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        model.to(current_platform.device_type)
    else:
        [
            model.get_submodule(layer_name).to(
                device_id if isinstance(device_id, torch.device) else f"{current_platform.device_type}:{device_id}"
            )
            for layer_name, device_id in device_map.items()
        ]


def get_mapping_to_flat_buffer(
    tensors: List[torch.Tensor], alignment_bytes: int = 1
) -> List[Tuple[torch.Tensor, int, int]]:
    if not isinstance(alignment_bytes, int) or alignment_bytes < 1 or alignment_bytes & (alignment_bytes - 1):
        raise ValueError("buffer alignment must be a positive power of two")
    tensor_infos: List[Tuple[torch.Tensor, int, int]] = []

    offset = 0
    for tensor in tensors:
        alignment = max(1, alignment_bytes // tensor.element_size())
        offset = ((offset + alignment - 1) // alignment) * alignment
        tensor_numel = tensor.numel()
        # record some data so we can restore the device tensor later
        tensor_infos.append((tensor, offset, tensor_numel))
        offset += tensor_numel

    return tensor_infos


def move_tensors_to_device_buffer(
    tensors: List[Tensor],
    device: Union[str, torch.device] = "cpu",
    pin_memory: bool = True,
    non_blocking: bool = False,
    device_buffer: torch.Tensor = None,
    alignment_bytes: int = 1,
):
    if len(tensors) == 0:
        return None
    tensor_metas = [torch.zeros_like(tensor, device="meta") for tensor in tensors]
    mapping = get_mapping_to_flat_buffer(tensors, alignment_bytes=alignment_bytes)
    if device_buffer is None:
        device_buffer = torch.empty(
            mapping[-1][1] + mapping[-1][2], dtype=tensors[0].dtype, device=device, pin_memory=pin_memory
        )
    for (tensor, offset, tensor_numel), tensor_meta in zip(mapping, tensor_metas):
        device_buffer.narrow(0, offset, tensor_numel).copy_(tensor.view(-1), non_blocking=non_blocking)
        tensor.data = device_buffer.narrow(0, offset, tensor_numel).view(tensor_meta.shape)
    return device_buffer


def move_device_buffer_to_tensors(tensors: List[Tensor], device_buffer: torch.Tensor, alignment_bytes: int = 1):
    if len(tensors) == 0:
        return None
    for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(tensors, alignment_bytes=alignment_bytes):
        tensor.data = device_buffer.narrow(0, offset, tensor_numel).view(tensor.shape)


def offload_module(model: torch.nn.Module, device="cpu", pin_memory: bool = True, non_blocking: bool = False):
    tensors = list(model.parameters())
    if not getattr(model, "has_offloaded", False):
        setattr(
            model,
            "model_parameters_cpu_buffers",
            move_tensors_to_device_buffer(
                tensors=tensors,
                device=device,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                device_buffer=getattr(model, "model_parameters_cpu_buffers", None),
            ),
        )
        setattr(model, "has_offloaded", True)


def reload_module(model: torch.nn.Module, device=current_platform.device_type, non_blocking: bool = False):
    tensors = list(model.parameters())
    if getattr(model, "model_parameters_cpu_buffers", None) is not None and getattr(model, "has_offloaded"):
        move_device_buffer_to_tensors(
            tensors=tensors,
            device_buffer=getattr(model, "model_parameters_cpu_buffers").to(device, non_blocking=non_blocking),
        )
        setattr(model, "has_offloaded", False)
