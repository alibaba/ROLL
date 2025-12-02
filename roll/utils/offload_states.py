from enum import Enum
from typing import List, Tuple, Union

import torch
from torch import Tensor
from transformers import PreTrainedModel
from roll.platforms import current_platform


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


def get_mapping_to_flat_buffer(tensors: List[torch.Tensor]) -> List[Tuple[torch.Tensor, int, int]]:
    tensor_infos: List[Tuple[torch.Tensor, int, int]] = []

    offset = 0
    for tensor in tensors:
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
):
    if len(tensors) == 0:
        return None
    tensor_metas = [torch.zeros_like(tensor, device="meta") for tensor in tensors]
    if device_buffer is None:
        device_buffer = torch.empty(
            sum(p.numel() for p in tensors), dtype=tensors[0].dtype, device=device, pin_memory=pin_memory
        )
    for (tensor, offset, tensor_numel), tensor_meta in zip(get_mapping_to_flat_buffer(tensors), tensor_metas):
        device_buffer.narrow(0, offset, tensor_numel).copy_(tensor.view(-1), non_blocking=non_blocking)
        tensor.data = device_buffer.narrow(0, offset, tensor_numel).view(tensor_meta.shape)
    return device_buffer


def move_device_buffer_to_tensors(tensors: List[Tensor], device_buffer: torch.Tensor):
    if len(tensors) == 0:
        return None
    for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(tensors):
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
