"""
megatron offload states的实现思路：

offload
释放megatron.core.distributed.distributed_data_parallel.DistributedDataParallel中的buffer
offload optimizer中的main_weights, main_weights.to('cpu')，使用flat tensor
offload optimizer states, to('cpu')
offload model weights, to('cpu'), 使用flat tensor；释放shard_float16_groups和shard_fp32_groups


reload
"""
import gc
import types
from collections import defaultdict
from enum import Enum
from typing import Container, List, Union

import torch
from megatron.core import DistributedDataParallel
from megatron.core.distributed.param_and_grad_buffer import BufferType
from megatron.core.optimizer import MegatronOptimizer, ChainedOptimizer, FP32Optimizer, DistributedOptimizer, \
    Float16OptimizerWithFloat16Params
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher, MoEAllGatherTokenDispatcher
from megatron.core.fp8_utils import is_float8tensor
from torch import Tensor

from roll.platforms import current_platform
from roll.utils.offload_states import move_tensors_to_device_buffer, move_device_buffer_to_tensors


def bind_megatron_offload_states_func(optimizer: MegatronOptimizer):
    if isinstance(optimizer, ChainedOptimizer):
        for sub_optimizer in optimizer.chained_optimizers:
            bind_megatron_offload_states_func(sub_optimizer)
        optimizer.offload_states = types.MethodType(chained_optimizers_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(chained_optimizers_reload_states, optimizer)
    elif isinstance(optimizer, Float16OptimizerWithFloat16Params):
        optimizer.offload_states = types.MethodType(float16_optimizer_with_float16_params_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(float16_optimizer_with_float16_params_reload_states, optimizer)
    elif isinstance(optimizer, DistributedOptimizer):
        optimizer.offload_states = types.MethodType(distributed_optimizer_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(distributed_optimizer_reload_states, optimizer)
    elif isinstance(optimizer, FP32Optimizer):
        optimizer.offload_states = types.MethodType(fp32_optimizer_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(fp32_optimizer_reload_states, optimizer)
    else:
        raise RuntimeError(f'optimizer {optimizer} does not support offload_states func')


class MegatronOffloadStateType(str, Enum):
    """
    """
    model_params = "model_params"
    optimizer_states = "optimizer_states"
    other_params = "other_params"


def chained_optimizers_offload_states(self: ChainedOptimizer,
                                      include: Container[MegatronOffloadStateType] = None,
                                      pin_memory: bool = True,
                                      non_blocking: bool = False
                                      ):
    for sub_optimizer in self.chained_optimizers:
        sub_optimizer.offload_states(include=include, pin_memory=pin_memory, non_blocking=non_blocking)


def chained_optimizers_reload_states(self: ChainedOptimizer,
                                     include: Container[MegatronOffloadStateType] = None,
                                     non_blocking: bool = False
                                     ):
    for sub_optimizer in self.chained_optimizers[:-1]:
        sub_optimizer.reload_states(include=include, non_blocking=non_blocking)
    self.chained_optimizers[-1].reload_states(
        include=include, non_blocking=non_blocking, skip_grad_hook_register=True
    )


def float16_optimizer_with_float16_params_offload_states(self: Float16OptimizerWithFloat16Params,
                                                         include: Container[MegatronOffloadStateType] = None,
                                                         pin_memory: bool = True,
                                                         non_blocking: bool = False
                                                         ):
    device = torch.device('cpu')
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_offload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        float16_weights: List[Tensor] = [param for sub_group in self.float16_groups for param in sub_group]
        setattr(self, "float16_groups_cpu_buffer", move_tensors_to_device_buffer(tensors=float16_weights,
                                                                                 device=device,
                                                                                 pin_memory=pin_memory,
                                                                                 non_blocking=non_blocking,
                                                                                 device_buffer=getattr(self, "float16_groups_cpu_buffer", None)))

        fp32_weights: List[Tensor] = [param for sub_group in self.fp32_from_fp32_groups for param in sub_group]
        setattr(self, "float32_groups_cpu_buffer", move_tensors_to_device_buffer(tensors=fp32_weights,
                                                                                 device=device,
                                                                                 pin_memory=pin_memory,
                                                                                 non_blocking=non_blocking,
                                                                                 device_buffer=getattr(self, "float32_groups_cpu_buffer", None)))

        self.offloaded_states.add(MegatronOffloadStateType.model_params)

    if needs_offload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # offload grad
        self.zero_grad()
        move_grad_data_to_device(optimizer=self, device=device, pin_memory=pin_memory, non_blocking=non_blocking)

        # offload optimizer main param
        fp32_from_float16_weights: List[Tensor] = [param for sub_group in self.fp32_from_float16_groups for param in
                                                   sub_group]
        setattr(self, "fp32_from_float16_groups_cpu_buffer",
                move_tensors_to_device_buffer(tensors=fp32_from_float16_weights,
                                              device=device,
                                              pin_memory=pin_memory,
                                              non_blocking=non_blocking,
                                              device_buffer=getattr(self, "fp32_from_float16_groups_cpu_buffer", None)))

        self.offloaded_states.add(MegatronOffloadStateType.other_params)

    if needs_offload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # offload optimizer states
        offload_adam_states(self.optimizer, device, pin_memory=pin_memory, non_blocking=non_blocking)
        self.offloaded_states.add(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()
    gc.collect()
    current_platform.empty_cache()


def float16_optimizer_with_float16_params_reload_states(self: Float16OptimizerWithFloat16Params,
                                                        include: Container[MegatronOffloadStateType] = None,
                                                        non_blocking: bool = False,
                                                        skip_grad_hook_register: bool = False,
                                                        ):
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_reload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        float16_weights: List[Tensor] = [param for sub_group in self.float16_groups for param in sub_group]
        if getattr(self, "float16_groups_cpu_buffer") is not None:
            move_device_buffer_to_tensors(tensors=float16_weights,
                                          device_buffer=getattr(self, "float16_groups_cpu_buffer").to(device,
                                                                                                      non_blocking=non_blocking))
        fp32_weights: List[Tensor] = [param for sub_group in self.fp32_from_fp32_groups for param in sub_group]

        if getattr(self, "float32_groups_cpu_buffer") is not None:
            move_device_buffer_to_tensors(tensors=fp32_weights,
                                          device_buffer=getattr(self, "float32_groups_cpu_buffer").to(device,
                                                                                                      non_blocking=non_blocking))
        self.offloaded_states.remove(MegatronOffloadStateType.model_params)

    if needs_reload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # reload grad
        move_grad_data_to_device(optimizer=self, device=device, non_blocking=non_blocking)

        # reload optimizer main param
        fp32_from_float16_weights: List[Tensor] = [param for sub_group in self.fp32_from_float16_groups for param in
                                                   sub_group]
        if getattr(self, "fp32_from_float16_groups_cpu_buffer") is not None:
            move_device_buffer_to_tensors(tensors=fp32_from_float16_weights,
                                          device_buffer=getattr(self, "fp32_from_float16_groups_cpu_buffer").to(device,
                                                                                                                non_blocking=non_blocking))

            self.offloaded_states.remove(MegatronOffloadStateType.other_params)

    if needs_reload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # reload optimizer states
        reload_adam_states(self.optimizer, device, non_blocking=non_blocking)
        self.offloaded_states.remove(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()


def move_ddp_model_params_tensor_to_device(optimizer: DistributedOptimizer,
                                           device: Union[torch.device, str],
                                           pin_memory: bool = True,
                                           non_blocking: bool = False
                                           ):
    for buffer in optimizer.buffers:
        assert buffer.param_data is not None

        if device == torch.device('cpu') and pin_memory:
            pin_buffer = torch.empty_like(buffer.param_data.data, device=device).pin_memory()
            pin_buffer.copy_(buffer.param_data.data, non_blocking=non_blocking)
            buffer.param_data.data = pin_buffer
        else:
            buffer.param_data.data = buffer.param_data.data.to(device, non_blocking=non_blocking)

        for param in buffer.params[::-1]:
            param_start_index, param_end_index, bucket_id = buffer.param_index_map[param]
            new_param_data = buffer._get(
                param.data.shape, param_start_index, buffer_type=BufferType.PARAM
            )
            if is_float8tensor(param):
                param._data = new_param_data
            else:
                param.data = new_param_data

        for bucket in buffer.buckets:
            start_index, end_index = buffer.bucket_indices[bucket.bucket_id]
            bucket.param_data.data = buffer._get(torch.Size([end_index - start_index]), start_index,
                                                 buffer_type=BufferType.PARAM)

    if hasattr(optimizer, "shard_float16_groups") and (
            len(optimizer.shard_float16_groups[0]) > 0 or len(optimizer.shard_fp32_groups[0]) > 0):
        # offload optimizer model group
        param_gbuf_map = optimizer.model_param_gbuf_map
        gbuf_ranges = optimizer.gbuf_ranges
        for group_index, group_range in enumerate(optimizer.opt_group_ranges):
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            for model_param in group_range["params"]:
                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in [f'torch.{current_platform.device_type}.HalfTensor', f'torch.{current_platform.device_type}.BFloat16Tensor',
                                          'torch.BFloat16Tensor', 'torch.HalfTensor'] or \
                   (current_platform.device_type == "cuda" and model_param.type() in ['torch.HalfTensor', 'torch.BFloat16Tensor']):
                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1)[param_range.start: param_range.end]

                    # shard_float16_groups 不属于optimizer state的key，可以直接替换param
                    # 这种方式: optimizer.shard_float16_groups[
                    # group_index][len(shard_float16_params_this_group)].data = shard_model_param
                    # 不能实现显存释放，定位到是model_param.detach()的影响，下面的fp32能正常释放
                    optimizer.shard_float16_groups[group_index][
                        len(shard_float16_params_this_group)] = shard_model_param
                    shard_float16_params_this_group.append(shard_model_param)
                # fp32 params.
                elif model_param.type() in [f'torch.{current_platform.device_type}.FloatTensor', 'torch.FloatTensor']:
                    shard_model_param = model_param.view(-1)[param_range.start: param_range.end]
                    optimizer.shard_fp32_groups[group_index][
                        len(shard_fp32_params_this_group)].data = shard_model_param.data
                    shard_fp32_params_this_group.append(shard_model_param)


def move_grad_data_to_device(optimizer,
                             device: Union[torch.device, str],
                             pin_memory: bool = True,
                             non_blocking: bool = False,
                             skip_grad_hook_register: bool = False,
                             ):
    assert hasattr(optimizer, "buffers"), "optimizer has no buffers"
    device = torch.device(device)
    for buffer in optimizer.buffers:
        # if device == torch.device('cpu') and pin_memory:
        #     pin_buffer = torch.empty_like(buffer.grad_data.data, device=device).pin_memory()
        #     pin_buffer.copy_(buffer.grad_data.data, non_blocking=non_blocking)
        #     buffer.grad_data.data = pin_buffer
        # else:
        #     buffer.grad_data.data = buffer.grad_data.data.to(device, non_blocking=non_blocking)

        # 释放grad, 节省cpu memory
        if device == torch.device('cpu'):
            buffer.grad_data.data = torch.tensor(1, dtype=buffer.grad_data.data.dtype, device=device, pin_memory=pin_memory)
            for param in buffer.params[::-1]:
                param.main_grad = torch.tensor(1, dtype=buffer.grad_data.data.dtype, device=device, pin_memory=pin_memory)
            for bucket in buffer.buckets:
                bucket.grad_data.data = torch.tensor(1, dtype=buffer.grad_data.data.dtype, device=device, pin_memory=pin_memory)
        else:
            buffer.grad_data.data = torch.zeros(buffer.numel,
                                                dtype=buffer.grad_dtype,
                                                device=device,
                                                requires_grad=False)
            for param in buffer.params[::-1]:
                param_start_index, param_end_index, bucket_id = buffer.param_index_map[param]
                param.main_grad = buffer._get(
                    param.data.shape, param_start_index, buffer_type=BufferType.GRAD
                )
            for bucket in buffer.buckets:
                start_index, end_index = buffer.bucket_indices[bucket.bucket_id]
                bucket.grad_data.data = buffer._get(
                    torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
                )

    if device == torch.device(f'{current_platform.device_type}:{current_platform.current_device()}') and not skip_grad_hook_register:
        for model_chunk in optimizer.model_chunks:
            for param in model_chunk.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator function.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(model_chunk._make_backward_post_hook(param))
                    model_chunk.grad_accs.append(grad_acc)
    elif device == torch.device("cpu"):
        for model_chunk in optimizer.model_chunks:
            model_chunk.grad_accs.clear()


def distributed_optimizer_offload_states(self: DistributedOptimizer,
                                         include: Container[MegatronOffloadStateType] = None,
                                         pin_memory: bool = True,
                                         non_blocking: bool = False
                                         ):
    device = torch.device('cpu')
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_offload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        move_ddp_model_params_tensor_to_device(optimizer=self, device=device, pin_memory=pin_memory,
                                               non_blocking=non_blocking)
        self.offloaded_states.add(MegatronOffloadStateType.model_params)

    if needs_offload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # offload grad/optimizer related
        self.zero_grad()
        move_grad_data_to_device(optimizer=self, device=device, pin_memory=pin_memory, non_blocking=non_blocking)

        # offload main_weights
        shard_fp32_from_float16_weights: List[Tensor] = [param for sub_group in self.shard_fp32_from_float16_groups for
                                                         param in sub_group]
        setattr(self, "shard_fp32_from_float16_groups_cpu_buffer",
                move_tensors_to_device_buffer(tensors=shard_fp32_from_float16_weights,
                                              device=device,
                                              pin_memory=pin_memory,
                                              non_blocking=non_blocking,
                                              device_buffer=getattr(self, "shard_fp32_from_float16_groups_cpu_buffer", None),
                                              ))
        self.offloaded_states.add(MegatronOffloadStateType.other_params)

    if needs_offload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # offload optimizer states
        offload_adam_states(self.optimizer, device, pin_memory=pin_memory, non_blocking=non_blocking)
        self.offloaded_states.add(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()
    gc.collect()
    current_platform.empty_cache()


def distributed_optimizer_reload_states(self: DistributedOptimizer,
                                        include: Container[MegatronOffloadStateType] = None,
                                        non_blocking: bool = False,
                                        skip_grad_hook_register: bool = False,
                                        ):
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')

    self.offloaded_states = getattr(self, "offloaded_states", set())

    if needs_reload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        move_ddp_model_params_tensor_to_device(optimizer=self, device=device, non_blocking=non_blocking)
        self.offloaded_states.remove(MegatronOffloadStateType.model_params)

    if needs_reload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # reload grad/optimizer related
        move_grad_data_to_device(optimizer=self, device=device, skip_grad_hook_register=skip_grad_hook_register)

        # reload main_weights
        shard_fp32_from_float16_weights: List[Tensor] = [param for sub_group in self.shard_fp32_from_float16_groups for
                                                         param in sub_group]
        if getattr(self, "shard_fp32_from_float16_groups_cpu_buffer") is not None:
            move_device_buffer_to_tensors(tensors=shard_fp32_from_float16_weights,
                                          device_buffer=getattr(self, "shard_fp32_from_float16_groups_cpu_buffer").to(device,
                                                                                                                non_blocking=non_blocking), )

        self.offloaded_states.remove(MegatronOffloadStateType.other_params)

    if needs_reload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # reload optimizer states
        reload_adam_states(self.optimizer, device, non_blocking=non_blocking)
        self.offloaded_states.remove(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()


def fp32_optimizer_offload_states(self: FP32Optimizer,
                                  include: Container[MegatronOffloadStateType] = None,
                                  pin_memory: bool = True,
                                  non_blocking: bool = False
                                  ):
    device = torch.device('cpu')
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_offload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        float32_weights: List[Tensor] = [param for sub_group in self.optimizer.param_groups for param in
                                         sub_group['params']]
        setattr(self, "optimizer_param_groups_cpu_buffer", move_tensors_to_device_buffer(tensors=float32_weights,
                                                                                         device=device,
                                                                                         pin_memory=pin_memory,
                                                                                         non_blocking=non_blocking,
                                                                                         device_buffer=getattr(self, "optimizer_param_groups_cpu_buffer", None),
                                                                                         ))

        self.offloaded_states.add(MegatronOffloadStateType.model_params)

    if needs_offload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # offload grad
        self.zero_grad()
        move_grad_data_to_device(optimizer=self, device=device, pin_memory=pin_memory, non_blocking=non_blocking)

        # offload optimizer main param, no
        self.offloaded_states.add(MegatronOffloadStateType.other_params)

    if needs_offload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # offload optimizer states
        offload_adam_states(self.optimizer, device, pin_memory=pin_memory, non_blocking=non_blocking)
        self.offloaded_states.add(MegatronOffloadStateType.optimizer_states)
    current_platform.synchronize()
    gc.collect()
    current_platform.empty_cache()


def fp32_optimizer_reload_states(self: FP32Optimizer,
                                 include: Container[MegatronOffloadStateType] = None,
                                 non_blocking: bool = False,
                                 skip_grad_hook_register: bool = False,
                                 ):
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_reload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        float32_weights: List[Tensor] = [param for sub_group in self.optimizer.param_groups for param in
                                         sub_group['params']]
        if getattr(self, "optimizer_param_groups_cpu_buffer") is not None:
            move_device_buffer_to_tensors(tensors=float32_weights,
                                          device_buffer=getattr(self, "optimizer_param_groups_cpu_buffer").to(device,
                                                                                                              non_blocking=non_blocking), )

        self.offloaded_states.remove(MegatronOffloadStateType.model_params)

    if needs_reload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # reload grad
        move_grad_data_to_device(
            optimizer=self, device=device, non_blocking=non_blocking, skip_grad_hook_register=skip_grad_hook_register
        )

        self.offloaded_states.remove(MegatronOffloadStateType.other_params)

    if needs_reload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # reload optimizer states
        reload_adam_states(self.optimizer, device, non_blocking=non_blocking)
        self.offloaded_states.remove(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()


def offload_megatron_no_grad_module(model_chunks: List[Union[DistributedDataParallel, MegatronModule]],
                                    pin_memory: bool = True,
                                    non_blocking: bool = False
                                    ):
    """
        需要offload一下 grad=False的参数
    """

    device = torch.device('cpu')
    for model_chunk in model_chunks:
        if isinstance(model_chunk, DistributedDataParallel):
            model_chunk = model_chunk.module
        model_chunk.offloaded_states = getattr(model_chunk, "offloaded_states", set())
        if needs_offload(MegatronOffloadStateType.model_params, include=[MegatronOffloadStateType.model_params],
                         offloaded_states=model_chunk.offloaded_states):
            model_chunk.param_dtype_to_params = getattr(model_chunk, "param_dtype_to_params", defaultdict(list))
            if not model_chunk.param_dtype_to_params:
                for param in model_chunk.parameters():
                    if not param.requires_grad:
                        param_dtype = param.dtype
                        if is_float8tensor(param):
                            param_dtype = torch.uint8
                        model_chunk.param_dtype_to_params[param_dtype].append(param)
            for param_dtype, params in model_chunk.param_dtype_to_params.items():
                setattr(model_chunk, f"{param_dtype}_ddp_no_grad_groups_cpu_buffer",
                        move_tensors_to_device_buffer(tensors=params,
                                                      device=device,
                                                      pin_memory=pin_memory,
                                                      non_blocking=non_blocking,
                                                      device_buffer=getattr(model_chunk, f"{param_dtype}_ddp_no_grad_groups_cpu_buffer", None),
                                                      ))

            if hasattr(model_chunk, "decoder"):
                setattr(model_chunk.decoder, "input_tensor", None)
                for layer in model_chunk.decoder.layers:
                    if isinstance(layer.mlp, MoELayer):
                        if isinstance(layer.mlp.token_dispatcher, MoEAlltoAllTokenDispatcher):
                            layer.mlp.token_dispatcher.probs = None
                            layer.mlp.token_dispatcher.routing_map = None
                            layer.mlp.token_dispatcher.hidden_shape = None
                            layer.mlp.token_dispatcher.reversed_local_input_permutation_mapping = None
                            layer.mlp.token_dispatcher.input_splits = None
                            layer.mlp.token_dispatcher.output_splits = None
                            layer.mlp.token_dispatcher.output_splits_tp = None
                            layer.mlp.token_dispatcher.num_global_tokens_per_local_expert_cpu = None
                            layer.mlp.token_dispatcher.num_out_tokens = None
                            layer.mlp.token_dispatcher.capacity = None
                        elif isinstance(layer.mlp.token_dispatcher, MoEAllGatherTokenDispatcher):
                            layer.mlp.token_dispatcher.hidden_shape = None
                            layer.mlp.token_dispatcher.local_map = None
                            layer.mlp.token_dispatcher.local_probs = None
                            layer.mlp.token_dispatcher.reversed_local_input_permutation_mapping = None


            model_chunk.offloaded_states.add(MegatronOffloadStateType.model_params)



def reload_megatron_no_grad_module(model_chunks: List[Union[DistributedDataParallel, MegatronModule]],
                                   non_blocking: bool = False):
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')

    for model_chunk in model_chunks:
        if isinstance(model_chunk, DistributedDataParallel):
            model_chunk = model_chunk.module

        model_chunk.offloaded_states = getattr(model_chunk, "offloaded_states", set())
        if needs_reload(MegatronOffloadStateType.model_params, include=[MegatronOffloadStateType.model_params],
                        offloaded_states=model_chunk.offloaded_states):
            param_dtype_to_params = getattr(model_chunk, "param_dtype_to_params", {})
            for param_dtype, params in param_dtype_to_params.items():
                if getattr(model_chunk, f"{param_dtype}_ddp_no_grad_groups_cpu_buffer") is not None:
                    move_device_buffer_to_tensors(tensors=params,
                                                  device_buffer=getattr(model_chunk,
                                                                        f"{param_dtype}_ddp_no_grad_groups_cpu_buffer").to(device, non_blocking=non_blocking))

            model_chunk.offloaded_states.remove(MegatronOffloadStateType.model_params)


def needs_offload(target, include, offloaded_states):
    # return True
    return target not in offloaded_states and (include is None or target in include)


def needs_reload(target, include, offloaded_states):
    return (include == None or target in include) and (target in offloaded_states)


def offload_adam_states(optimizer, device, pin_memory: bool = False, non_blocking: bool = False):
    """Move optimizer states to device. Note that this assumes the state structure of DeepSpeed Adam."""
    state_tensors = []
    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            state_tensors.append(state["exp_avg"])
        if "exp_avg_sq" in state:
            state_tensors.append(state["exp_avg_sq"])
    setattr(optimizer, "optimizer_states_cpu_buffers",
            move_tensors_to_device_buffer(tensors=state_tensors, device=device, pin_memory=pin_memory, non_blocking=non_blocking,
                                          device_buffer=getattr(optimizer, "optimizer_states_cpu_buffers", None)))


def reload_adam_states(optimizer, device, non_blocking: bool = False):
    """Move optimizer states to device. Note that this assumes the state structure of DeepSpeed Adam."""
    state_tensors = []
    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            state_tensors.append(state["exp_avg"])
        if "exp_avg_sq" in state:
            state_tensors.append(state["exp_avg_sq"])
    if getattr(optimizer, "optimizer_states_cpu_buffers", None) is not None:
        move_device_buffer_to_tensors(tensors=state_tensors,
                                      device_buffer=getattr(optimizer, "optimizer_states_cpu_buffers").to(device, non_blocking=non_blocking),)
