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
import traceback
import types
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import Container, List, Union

import torch
from megatron.core import DistributedDataParallel
from megatron.core.distributed.param_and_grad_buffer import BufferType
from megatron.core.optimizer import MegatronOptimizer, ChainedOptimizer, FP32Optimizer, DistributedOptimizer, \
    Float16OptimizerWithFloat16Params
try:
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
except ImportError:  # Older Megatron versions have no CPU optimizer.
    HybridDeviceOptimizer = ()
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher, MoEAllGatherTokenDispatcher
from megatron.core.fp8_utils import is_float8tensor
from torch import Tensor

from roll.platforms import current_platform
from roll.utils.offload_states import move_tensors_to_device_buffer, move_device_buffer_to_tensors, clear_memory
from roll.third_party.megatron.optimizer_lifecycle import (
    cpu_optimizer_offload_states,
    cpu_optimizer_reload_states,
)
from roll.third_party.megatron.cpu_master_params import (
    cpu_master_shard_pairs, empty_cpu_parameter_buffer, restore_cpu_master_shards,
)


def bind_megatron_offload_states_func(optimizer: MegatronOptimizer):
    if not isinstance(optimizer, ChainedOptimizer) and _uses_cpu_master_model_offload(optimizer):
        if not isinstance(optimizer, DistributedOptimizer) or not isinstance(optimizer.optimizer, HybridDeviceOptimizer):
            raise ValueError("offload_model_from_cpu_master requires DistributedOptimizer with HybridDeviceOptimizer")
        if optimizer.ddp_config.overlap_param_gather or optimizer.ddp_config.overlap_grad_reduce:
            raise ValueError("offload_model_from_cpu_master requires synchronous parameter and gradient collectives")
        cpu_master_shard_pairs(optimizer)
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
    elif isinstance(optimizer, HybridDeviceOptimizer):
        optimizer.offload_states = types.MethodType(cpu_optimizer_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(cpu_optimizer_reload_states, optimizer)
    else:
        raise RuntimeError(f'optimizer {optimizer} does not support offload_states func')


class MegatronOffloadStateType(str, Enum):
    """
    """
    model_params = "model_params"
    optimizer_states = "optimizer_states"
    other_params = "other_params"


def _clear_checkpoint_exception_frames(error):
    """Release failed loader locals while retaining the exception traceback."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))


@contextmanager
def checkpoint_grad_buffer_offload(optimizer):
    """Free disposable training gradients while model factories merge weights.

    CPU optimizer masters and moments stay owned by the optimizer. The existing
    other_params transition replaces gradients with scalar CPU placeholders;
    it does not park a full gradient image in host memory.
    """
    leaves = getattr(optimizer, "chained_optimizers", [optimizer])
    suspended = []
    try:
        for leaf in leaves:
            if MegatronOffloadStateType.other_params not in getattr(leaf, "offloaded_states", set()):
                leaf.offload_states(include=[MegatronOffloadStateType.other_params], pin_memory=False)
                suspended.append(leaf)
        yield
    except BaseException as error:
        # A failed loader or installer can retain merged tensors in traceback
        # frame locals even after the caller drops its model_state reference.
        _clear_checkpoint_exception_frames(error)
        raise
    finally:
        # The caller drops loaded model tensors before leaving this context.
        # Release their cached allocations before rebuilding training buffers.
        clear_memory(clear_host_memory=True)
        registered_models = set()
        restore_error = None
        for leaf in suspended:
            models = {id(model) for model in leaf.model_chunks}
            try:
                leaf.reload_states(include=[MegatronOffloadStateType.other_params],
                                   skip_grad_hook_register=models.issubset(registered_models))
                registered_models.update(models)
            except Exception as error:
                _clear_checkpoint_exception_frames(error)
                if restore_error is None:
                    restore_error = error
                elif hasattr(restore_error, "add_note"):
                    restore_error.add_note(f"Another gradient restoration failed: {error!r}")
        if restore_error is not None:
            raise restore_error


def chained_optimizers_offload_states(self: ChainedOptimizer,
                                      include: Container[MegatronOffloadStateType] = None,
                                      pin_memory: bool = True,
                                      non_blocking: bool = False
                                      ):
    # Validate all children before destructive model offload of the first one.
    for sub_optimizer in self.chained_optimizers:
        if _uses_cpu_master_model_offload(sub_optimizer) and needs_offload(
                MegatronOffloadStateType.model_params, include,
                getattr(sub_optimizer, "offloaded_states", set())):
            cpu_master_shard_pairs(sub_optimizer)
            _cpu_master_bucket_groups(sub_optimizer)
    for sub_optimizer in self.chained_optimizers:
        sub_optimizer.offload_states(include=include, pin_memory=pin_memory, non_blocking=non_blocking)


def chained_optimizers_reload_states(self: ChainedOptimizer,
                                     include: Container[MegatronOffloadStateType] = None,
                                     non_blocking: bool = False
                                     ):
    models_to_register = {}
    for sub_optimizer in self.chained_optimizers:
        if needs_reload(MegatronOffloadStateType.other_params, include,
                        getattr(sub_optimizer, "offloaded_states", set())):
            for model in getattr(sub_optimizer, "model_chunks", []):
                models_to_register[id(model)] = model
        sub_optimizer.reload_states(
            include=include, non_blocking=non_blocking, skip_grad_hook_register=True
        )
    # Dense and expert optimizers can own different parameters of the same
    # model. A later CPU -> CUDA .data move invalidates that parameter's CPU
    # AccumulateGrad, so register only after every optimizer has restored it.
    _register_megatron_grad_hooks(models_to_register.values())


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

    clear_memory()


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
            self.float16_groups_cpu_buffer = None

        fp32_weights: List[Tensor] = [param for sub_group in self.fp32_from_fp32_groups for param in sub_group]

        if getattr(self, "float32_groups_cpu_buffer") is not None:
            move_device_buffer_to_tensors(tensors=fp32_weights,
                                          device_buffer=getattr(self, "float32_groups_cpu_buffer").to(device,
                                                                                                      non_blocking=non_blocking))
            self.float32_groups_cpu_buffer = None

        self.offloaded_states.remove(MegatronOffloadStateType.model_params)

    if needs_reload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # reload grad
        move_grad_data_to_device(optimizer=self, device=device, non_blocking=non_blocking,
                                 skip_grad_hook_register=skip_grad_hook_register)

        # reload optimizer main param
        fp32_from_float16_weights: List[Tensor] = [param for sub_group in self.fp32_from_float16_groups for param in
                                                   sub_group]
        if getattr(self, "fp32_from_float16_groups_cpu_buffer") is not None:
            move_device_buffer_to_tensors(tensors=fp32_from_float16_weights,
                                          device_buffer=getattr(self, "fp32_from_float16_groups_cpu_buffer").to(device,
                                                                                                                non_blocking=non_blocking))
            self.fp32_from_float16_groups_cpu_buffer = None

            self.offloaded_states.remove(MegatronOffloadStateType.other_params)

    if needs_reload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # reload optimizer states
        reload_adam_states(self.optimizer, device, non_blocking=non_blocking)
        self.offloaded_states.remove(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()


def _clear_hybrid_bucket_shard_cache(optimizer, cache_name):
    """Discard collective slice views when their underlying DDP buffer moves."""
    if not isinstance(optimizer.optimizer, HybridDeviceOptimizer):
        return
    moved_buckets = {id(bucket) for buffer in optimizer.buffers for bucket in buffer.buckets}
    for model_chunk in optimizer.model_chunks:
        for group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            cache = getattr(group, cache_name, None)
            if cache is not None:
                for index, bucket in enumerate(group.buckets):
                    if id(bucket) in moved_buckets:
                        cache[index] = None


def _rebind_hybrid_model_shards(optimizer, replacements):
    """Replace Hybrid originals without moving their authoritative inner state.

    A detached shard's _base can retain the old DDP allocation even after .data
    is rebound. Replace the view, then rekey every Hybrid owner of that view;
    CPU master tensors and Adam state dictionaries remain the same objects.
    """
    hybrid = optimizer.optimizer
    if not replacements or not isinstance(hybrid, HybridDeviceOptimizer):
        return

    def replace(value):
        return replacements.get(value, value) if isinstance(value, Tensor) else value

    def rekey(mapping):
        entries = [(replace(key), replace(value)) for key, value in mapping.items()]
        mapping.clear()
        mapping.update(entries)

    for groups in (hybrid.param_groups, hybrid.cpu_param_groups, hybrid.gpu_param_groups,
                   [group["orig_group"] for group in optimizer.opt_group_ranges]):
        for group in groups:
            group["params"][:] = [replace(param) for param in group["params"]]
    for name in ("state", "gpu_params_map_cpu_copy", "cpu_copys_map_gpu_param",
                 "param_to_fp32_param", "fp32_param_to_orig_param",
                 "param_to_inner_param", "inner_param_to_orig_param"):
        rekey(getattr(hybrid, name))
    # Covers any directly optimized originals as well as the usual FP32 inner
    # parameters. The latter are not keys in replacements and stay untouched.
    for sub_optimizer in hybrid.sub_optimizers:
        for group in sub_optimizer.param_groups:
            group["params"][:] = [replace(param) for param in group["params"]]
        rekey(sub_optimizer.state)


def _uses_cpu_master_model_offload(optimizer):
    return getattr(getattr(optimizer, "config", None), "offload_model_from_cpu_master", False)


def _cpu_master_bucket_groups(optimizer):
    owned = {id(bucket) for buffer in optimizer.buffers for bucket in buffer.buckets}
    groups, covered = [], set()
    for model in optimizer.model_chunks:
        for group in model.bucket_groups + model.expert_parallel_bucket_groups:
            bucket_ids = {id(bucket) for bucket in group.buckets}
            if bucket_ids & owned:
                if not bucket_ids <= owned:
                    raise ValueError("CPU master restore cannot split a parameter collective across optimizers")
                if getattr(group, "param_gather_handle", None) is not None:
                    raise ValueError("CPU master offload requires completed parameter collectives")
                groups.append(group)
                covered.update(bucket_ids)
    if covered != owned:
        raise ValueError("CPU master restore requires a collective for every parameter buffer")
    return groups


def move_ddp_model_params_tensor_to_device(optimizer: DistributedOptimizer,
                                           device: Union[torch.device, str],
                                           pin_memory: bool = True,
                                           non_blocking: bool = False
                                           ):
    from_master = _uses_cpu_master_model_offload(optimizer)
    if from_master:
        cpu_master_shard_pairs(optimizer)
        bucket_groups = _cpu_master_bucket_groups(optimizer)
    _clear_hybrid_bucket_shard_cache(optimizer, "cached_param_buffer_shard_list")
    replacements = {}
    for buffer in optimizer.buffers:
        assert buffer.param_data is not None

        if from_master:
            if torch.device(device).type == "cpu":
                buffer.param_data.data = empty_cpu_parameter_buffer(buffer.numel, buffer.param_data.dtype)
            else:
                buffer.param_data.data = torch.zeros(buffer.numel, dtype=buffer.param_data.dtype, device=device)
        elif device == torch.device('cpu') and pin_memory:
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

                    old_shard = optimizer.shard_float16_groups[group_index][
                        len(shard_float16_params_this_group)]
                    if isinstance(optimizer.optimizer, HybridDeviceOptimizer):
                        # Preserve model-parallel metadata and decoupled gradients;
                        # full offload clears gradients through the new owners next.
                        shard_model_param.__dict__.update(old_shard.__dict__)
                        replacements[old_shard] = shard_model_param
                    optimizer.shard_float16_groups[group_index][
                        len(shard_float16_params_this_group)] = shard_model_param
                    shard_float16_params_this_group.append(shard_model_param)
                # fp32 params.
                elif model_param.type() in [f'torch.{current_platform.device_type}.FloatTensor', 'torch.FloatTensor']:
                    shard_model_param = model_param.view(-1)[param_range.start: param_range.end]
                    optimizer.shard_fp32_groups[group_index][
                        len(shard_fp32_params_this_group)].data = shard_model_param.data
                    shard_fp32_params_this_group.append(shard_model_param)

    _rebind_hybrid_model_shards(optimizer, replacements)
    if from_master and torch.device(device).type != "cpu":
        restore_cpu_master_shards(cpu_master_shard_pairs(optimizer))
        # Each child owns either dense or expert buckets. Gather only these
        # buckets, using their native DP group, after all local shards are ready.
        for group in bucket_groups:
            group.start_param_sync(force_sync=True)


def _register_megatron_grad_hooks(model_chunks):
    for model_chunk in model_chunks:
        for param in model_chunk.module.parameters():
            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(model_chunk._make_backward_post_hook(param))
                model_chunk.grad_accs.append(grad_acc)


def move_grad_data_to_device(optimizer,
                             device: Union[torch.device, str],
                             pin_memory: bool = True,
                             non_blocking: bool = False,
                             skip_grad_hook_register: bool = False,
                             ):
    assert hasattr(optimizer, "buffers"), "optimizer has no buffers"
    _clear_hybrid_bucket_shard_cache(optimizer, "cached_grad_buffer_shard_list")
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
        _register_megatron_grad_hooks(optimizer.model_chunks)
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
                                                         param in sub_group if param is not None]
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

    clear_memory()


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
                                                         param in sub_group if param is not None]
        if getattr(self, "shard_fp32_from_float16_groups_cpu_buffer") is not None:
            move_device_buffer_to_tensors(tensors=shard_fp32_from_float16_weights,
                                          device_buffer=getattr(self, "shard_fp32_from_float16_groups_cpu_buffer").to(device,
                                                                                                                non_blocking=non_blocking), )
            self.shard_fp32_from_float16_groups_cpu_buffer = None

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
    clear_memory()


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
            self.optimizer_param_groups_cpu_buffer = None

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
                # Preserve CUDA allocation alignment after packing ragged
                # frozen tensors; unaligned GEMMs can change backward rounding.
                setattr(model_chunk, f"{param_dtype}_ddp_no_grad_groups_cpu_buffer",
                        move_tensors_to_device_buffer(tensors=params,
                                                      device=device,
                                                      pin_memory=pin_memory,
                                                      non_blocking=non_blocking,
                                                      alignment_bytes=256,
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
                buffer_attr = f"{param_dtype}_ddp_no_grad_groups_cpu_buffer"
                if getattr(model_chunk, buffer_attr, None) is not None:
                    move_device_buffer_to_tensors(tensors=params,
                                                  device_buffer=getattr(model_chunk, buffer_attr).to(device, non_blocking=non_blocking),
                                                  alignment_bytes=256)
                    setattr(model_chunk, buffer_attr, None)

            model_chunk.offloaded_states.remove(MegatronOffloadStateType.model_params)


def needs_offload(target, include, offloaded_states):
    # return True
    return target not in offloaded_states and (include is None or target in include)


def needs_reload(target, include, offloaded_states):
    return (include == None or target in include) and (target in offloaded_states)


def offload_adam_states(optimizer, device, pin_memory: bool = False, non_blocking: bool = False):
    """Move optimizer states to device."""
    if isinstance(optimizer, HybridDeviceOptimizer):
        # Its CPU sub-optimizers own persistent FP32 masters and Adam moments.
        # Flattening those moments with GPU state breaks both their ownership
        # and the next CPU update. Only the GPU-owned partition changes phase.
        if optimizer.gpu_optimizer is not None:
            offload_adam_states(optimizer.gpu_optimizer, device, pin_memory, non_blocking)
        return
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
    """Move optimizer states to device."""
    if isinstance(optimizer, HybridDeviceOptimizer):
        if optimizer.gpu_optimizer is not None:
            reload_adam_states(optimizer.gpu_optimizer, device, non_blocking)
        return
    state_tensors = []
    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            state_tensors.append(state["exp_avg"])
        if "exp_avg_sq" in state:
            state_tensors.append(state["exp_avg_sq"])
    if getattr(optimizer, "optimizer_states_cpu_buffers", None) is not None:
        move_device_buffer_to_tensors(tensors=state_tensors,
                                      device_buffer=getattr(optimizer, "optimizer_states_cpu_buffers").to(device, non_blocking=non_blocking),)
        optimizer.optimizer_states_cpu_buffers = None
