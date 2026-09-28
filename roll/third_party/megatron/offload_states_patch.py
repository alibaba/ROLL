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
from megatron.core import DistributedDataParallel, parallel_state
from megatron.core.distributed.param_and_grad_buffer import BufferType
from megatron.core.optimizer import MegatronOptimizer, ChainedOptimizer, FP32Optimizer, DistributedOptimizer, \
    Float16OptimizerWithFloat16Params
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher, MoEAllGatherTokenDispatcher
from megatron.core.fp8_utils import is_float8tensor
from torch import Tensor

from roll.platforms import current_platform
from roll.utils.context_managers import log_offload_debug

def bind_megatron_offload_states_func(optimizer: MegatronOptimizer):
    if isinstance(optimizer, ChainedOptimizer):
        for sub_optimizer in optimizer.chained_optimizers:
            bind_megatron_offload_states_func(sub_optimizer)
        optimizer.offload_states = types.MethodType(chained_optimizers_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(chained_optimizers_reload_states, optimizer)
    elif isinstance(optimizer, Float16OptimizerWithFloat16Params):
        _capture_buffer_param_shapes(optimizer)
        optimizer.offload_states = types.MethodType(float16_optimizer_with_float16_params_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(float16_optimizer_with_float16_params_reload_states, optimizer)
    elif isinstance(optimizer, DistributedOptimizer):
        _capture_buffer_param_shapes(optimizer)
        optimizer.offload_states = types.MethodType(distributed_optimizer_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(distributed_optimizer_reload_states, optimizer)
    elif isinstance(optimizer, FP32Optimizer):
        _capture_buffer_param_shapes(optimizer)
        optimizer.offload_states = types.MethodType(fp32_optimizer_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(fp32_optimizer_reload_states, optimizer)
    else:
        raise RuntimeError(f'optimizer {optimizer} does not support offload_states func')


def _capture_buffer_param_shapes(optimizer) -> None:
    """Record every buffer param's view shape at bind time. Offload empties
    param.data, so reload cannot derive shapes from the live tensors."""
    for buffer in getattr(optimizer, "buffers", []):
        buffer._roll_param_shapes = {param: param.data.shape for param in buffer.params}


def _param_view_shape(buffer, param) -> torch.Size:
    shapes = getattr(buffer, "_roll_param_shapes", None)
    if shapes is not None and param in shapes:
        return shapes[param]
    return param.data.shape


def is_model_params_offloaded(optimizer: MegatronOptimizer) -> bool:
    """Whether any (sub-)optimizer currently has its model params offloaded."""
    if isinstance(optimizer, ChainedOptimizer):
        return any(is_model_params_offloaded(sub) for sub in optimizer.chained_optimizers)
    return MegatronOffloadStateType.model_params in getattr(optimizer, "offloaded_states", set())


class MegatronOffloadStateType(str, Enum):
    """
    """
    model_params = "model_params"
    optimizer_states = "optimizer_states"
    other_params = "other_params"


def _replication_group(is_expert: bool):
    """The process group over which a param's full value is replicated.

    Dense params are replicated across DP (CP ranks hold identical params);
    expert params only across the expert-DP group (different EP ranks hold
    different experts, and with ETP != TP the expert-DP group may span TP
    ranks where dense shards differ) — so the two must dedup separately.
    """
    if is_expert:
        return parallel_state.get_expert_data_parallel_group()
    return parallel_state.get_data_parallel_group(with_context_parallel=True)


def _optimizer_replication_group(optimizer):
    """Replication group for an optimizer's params. get_megatron_optimizer
    builds at most one dense and one expert sub-optimizer, so param_groups
    within one optimizer are uniformly dense or uniformly expert."""
    is_expert = any(
        g.get("is_expert_parallel", False) for g in optimizer.optimizer.param_groups
    )
    return _replication_group(is_expert)


def chained_optimizers_offload_states(self: ChainedOptimizer,
                                      include: Container[MegatronOffloadStateType] = None,
                                      ):
    for idx, sub_optimizer in enumerate(self.chained_optimizers):
        sub_optimizer._offload_backend = self._offload_backend
        sub_optimizer._offload_key_prefix = f"{self._offload_key_prefix}_chain{idx}"
        sub_optimizer.offload_states(include=include)


def chained_optimizers_reload_states(self: ChainedOptimizer,
                                     include: Container[MegatronOffloadStateType] = None,
                                     ):
    key_prefix = getattr(self, '_offload_key_prefix', '')
    for idx, sub_optimizer in enumerate(self.chained_optimizers[:-1]):
        sub_optimizer._offload_backend = self._offload_backend
        sub_optimizer._offload_key_prefix = f"{key_prefix}_chain{idx}"
        sub_optimizer.reload_states(include=include)
    last = self.chained_optimizers[-1]
    last._offload_backend = self._offload_backend
    last._offload_key_prefix = f"{key_prefix}_chain{len(self.chained_optimizers) - 1}"
    last.reload_states(include=include, skip_grad_hook_register=True)


def float16_optimizer_with_float16_params_offload_states(self: Float16OptimizerWithFloat16Params,
                                                         include: Container[MegatronOffloadStateType] = None,
                                                         ):
    backend = self._offload_backend
    key_prefix = self._offload_key_prefix
    group = _optimizer_replication_group(self)

    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_offload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        float16_weights: List[Tensor] = [param for sub_group in self.float16_groups for param in sub_group]
        key = f"{key_prefix}_float16_model_weights"
        self._offload_key_float16_model = backend.put_tensors(key, float16_weights, replicated_over=group)

        fp32_weights: List[Tensor] = [param for sub_group in self.fp32_from_fp32_groups for param in sub_group]
        key = f"{key_prefix}_fp32_model_weights"
        self._offload_key_fp32_model = backend.put_tensors(key, fp32_weights, replicated_over=group)

        self.offloaded_states.add(MegatronOffloadStateType.model_params)

    if needs_offload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # offload grad
        self.zero_grad()
        _release_ddp_grad_buffers(self)

        # offload optimizer main param
        fp32_from_float16_weights: List[Tensor] = [param for sub_group in self.fp32_from_float16_groups for param in
                                                   sub_group]
        key = f"{key_prefix}_fp32_master_weights"
        self._offload_key_fp32_master = backend.put_tensors(key, fp32_from_float16_weights, replicated_over=group)

        self.offloaded_states.add(MegatronOffloadStateType.other_params)

    if needs_offload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # offload optimizer states
        adam_key = f"{key_prefix}_adam_states"
        offload_adam_states(self.optimizer, backend=backend, key=adam_key)
        self.offloaded_states.add(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()
    gc.collect()
    current_platform.empty_cache()


def float16_optimizer_with_float16_params_reload_states(self: Float16OptimizerWithFloat16Params,
                                                        include: Container[MegatronOffloadStateType] = None,
                                                        skip_grad_hook_register: bool = False,
                                                        ):
    backend = self._offload_backend
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_reload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        float16_weights: List[Tensor] = [param for sub_group in self.float16_groups for param in sub_group]
        key = getattr(self, "_offload_key_float16_model", None)
        if key is not None:
            backend.get_tensors(key, float16_weights, device)
            backend.delete_tensors(key)
            self._offload_key_float16_model = None

        fp32_weights: List[Tensor] = [param for sub_group in self.fp32_from_fp32_groups for param in sub_group]
        key = getattr(self, "_offload_key_fp32_model", None)
        if key is not None:
            backend.get_tensors(key, fp32_weights, device)
            backend.delete_tensors(key)
            self._offload_key_fp32_model = None

        self.offloaded_states.remove(MegatronOffloadStateType.model_params)

    if needs_reload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # reload grad
        _restore_ddp_grad_buffers(self, register_hooks=not skip_grad_hook_register)

        # reload optimizer main param
        fp32_from_float16_weights: List[Tensor] = [param for sub_group in self.fp32_from_float16_groups for param in
                                                   sub_group]
        key = getattr(self, "_offload_key_fp32_master", None)
        if key is not None:
            backend.get_tensors(key, fp32_from_float16_weights, device)
            backend.delete_tensors(key)
            self._offload_key_fp32_master = None

        self.offloaded_states.remove(MegatronOffloadStateType.other_params)

    if needs_reload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # reload optimizer states
        reload_adam_states(self.optimizer, device, backend=backend)
        self.offloaded_states.remove(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()


def _release_ddp_grad_buffers(optimizer) -> None:
    """Offload: discard DDP grad buffers + clear backward hooks (no backend involved)."""
    assert hasattr(optimizer, "buffers"), "optimizer has no buffers"
    for buffer in optimizer.buffers:
        dtype = buffer.grad_data.data.dtype
        buffer.grad_data.data = torch.tensor(1, dtype=dtype, device='cpu')
        for param in buffer.params[::-1]:
            param.main_grad = torch.tensor(1, dtype=dtype, device='cpu')
        for bucket in buffer.buckets:
            bucket.grad_data.data = torch.tensor(1, dtype=dtype, device='cpu')
    for model_chunk in optimizer.model_chunks:
        model_chunk.grad_accs.clear()


def _restore_ddp_grad_buffers(optimizer, register_hooks: bool = True) -> None:
    """Reload: rebuild grad buffers on GPU + re-view + re-register hooks."""
    assert hasattr(optimizer, "buffers"), "optimizer has no buffers"
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')
    for buffer in optimizer.buffers:
        buffer.grad_data.data = torch.zeros(buffer.numel, dtype=buffer.grad_dtype,
                                            device=device, requires_grad=False)
        for param in buffer.params[::-1]:
            param_start_index, param_end_index, bucket_id = buffer.param_index_map[param]
            param.main_grad = buffer._get(
                _param_view_shape(buffer, param), param_start_index, buffer_type=BufferType.GRAD
            )
        for bucket in buffer.buckets:
            start_index, end_index = buffer.bucket_indices[bucket.bucket_id]
            bucket.grad_data.data = buffer._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
            )

    if register_hooks:
        for model_chunk in optimizer.model_chunks:
            for param in model_chunk.module.parameters():
                if param.requires_grad:
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(model_chunk._make_backward_post_hook(param))
                    model_chunk.grad_accs.append(grad_acc)


def distributed_optimizer_offload_states(self: DistributedOptimizer,
                                         include: Container[MegatronOffloadStateType] = None,
                                         ):
    backend = self._offload_backend
    key_prefix = self._offload_key_prefix

    self.offloaded_states = getattr(self, "offloaded_states", set())

    # Offload cold data first (FP32 master, Adam states) so that DDP buffer —
    # the hot data accessed 4× per step — is written last and stays warmest
    # for backends with recency-based eviction.

    if needs_offload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # offload grad/optimizer related
        self.zero_grad()
        _release_ddp_grad_buffers(self)

        # offload main_weights (DistributedOptimizer shards them across DP,
        # so nothing is replicated: full per-rank copy)
        fp32_master_key = f"{key_prefix}_fp32_master_weights"
        if backend.has_key(fp32_master_key):
            backend.delete_tensors(fp32_master_key, replicated_over=False)
        shard_fp32_from_float16_weights: List[Tensor] = [
            param for sub_group in self.shard_fp32_from_float16_groups for param in sub_group
        ]
        self._offload_key_fp32_master = backend.put_tensors(
            fp32_master_key, shard_fp32_from_float16_weights, replicated_over=False
        )
        self.offloaded_states.add(MegatronOffloadStateType.other_params)

    if needs_offload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # offload optimizer states (DP-sharded, same as main weights)
        adam_key = f"{key_prefix}_adam_states"
        if backend.has_key(adam_key):
            backend.delete_tensors(adam_key, replicated_over=False)
        offload_adam_states(self.optimizer, backend=backend, key=adam_key)
        self.offloaded_states.add(MegatronOffloadStateType.optimizer_states)

    # PUT DDP buffer last — most recent in LRU → stays in DRAM during eviction.
    if needs_offload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        _offload_ddp_buffers(self)
        self.offloaded_states.add(MegatronOffloadStateType.model_params)

    log_offload_debug(
        f"optimizer.offload_states backend={type(backend).__name__}",
        store_stats=backend.get_stats(),
        once_id=f"distopt_offload_{id(self)}",
    )

    current_platform.synchronize()
    # gc/empty_cache only at phase boundaries (model/other params leaving GPU,
    # e.g. handing memory back to a colocated infer engine). The per-iteration
    # optimizer_states-only offload skips them: empty_cache gives this process
    # nothing (cached blocks are already reusable) and forces the next adam
    # reload through driver cudaMalloc.
    _phase_boundary = include is None or any(
        s in include for s in (MegatronOffloadStateType.model_params, MegatronOffloadStateType.other_params)
    )
    if _phase_boundary:
        gc.collect()
        current_platform.empty_cache()


def distributed_optimizer_reload_states(self: DistributedOptimizer,
                                        include: Container[MegatronOffloadStateType] = None,
                                        skip_grad_hook_register: bool = False,
                                        ):
    backend = self._offload_backend
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')

    self.offloaded_states = getattr(self, "offloaded_states", set())

    if needs_reload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        _reload_ddp_buffers(self)
        self.offloaded_states.remove(MegatronOffloadStateType.model_params)

    if needs_reload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # reload grad/optimizer related
        _restore_ddp_grad_buffers(self, register_hooks=not skip_grad_hook_register)

        # reload main_weights
        key = getattr(self, "_offload_key_fp32_master", None)
        if key is not None:
            shard_fp32_from_float16_weights: List[Tensor] = [
                param for sub_group in self.shard_fp32_from_float16_groups for param in sub_group
            ]
            backend.get_tensors(key, shard_fp32_from_float16_weights, device, replicated_over=False)
            backend.delete_tensors(key, replicated_over=False)
            self._offload_key_fp32_master = None

        self.offloaded_states.remove(MegatronOffloadStateType.other_params)

    if needs_reload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        # reload optimizer states
        reload_adam_states(self.optimizer, device, backend=backend)
        self.offloaded_states.remove(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()


def fp32_optimizer_offload_states(self: FP32Optimizer,
                                  include: Container[MegatronOffloadStateType] = None,
                                  ):
    backend = self._offload_backend
    key_prefix = self._offload_key_prefix

    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_offload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        group = _optimizer_replication_group(self)
        float32_weights: List[Tensor] = [param for sub_group in self.optimizer.param_groups for param in
                                         sub_group['params']]
        key = f"{key_prefix}_fp32_model_weights"
        self._offload_key_fp32_model = backend.put_tensors(key, float32_weights, replicated_over=group)

        self.offloaded_states.add(MegatronOffloadStateType.model_params)

    if needs_offload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # offload grad
        self.zero_grad()
        _release_ddp_grad_buffers(self)

        self.offloaded_states.add(MegatronOffloadStateType.other_params)

    if needs_offload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        adam_key = f"{key_prefix}_adam_states"
        offload_adam_states(self.optimizer, backend=backend, key=adam_key)
        self.offloaded_states.add(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()
    if include is None or any(
        s in include for s in (MegatronOffloadStateType.model_params, MegatronOffloadStateType.other_params)
    ):
        gc.collect()
        current_platform.empty_cache()


def fp32_optimizer_reload_states(self: FP32Optimizer,
                                 include: Container[MegatronOffloadStateType] = None,
                                 skip_grad_hook_register: bool = False,
                                 ):
    backend = self._offload_backend
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_reload(MegatronOffloadStateType.model_params, include, self.offloaded_states):
        float32_weights: List[Tensor] = [param for sub_group in self.optimizer.param_groups for param in
                                         sub_group['params']]
        key = getattr(self, "_offload_key_fp32_model", None)
        if key is not None:
            backend.get_tensors(key, float32_weights, device)
            backend.delete_tensors(key)
            self._offload_key_fp32_model = None

        self.offloaded_states.remove(MegatronOffloadStateType.model_params)

    if needs_reload(MegatronOffloadStateType.other_params, include, self.offloaded_states):
        # reload grad
        _restore_ddp_grad_buffers(self, register_hooks=not skip_grad_hook_register)

        self.offloaded_states.remove(MegatronOffloadStateType.other_params)

    if needs_reload(MegatronOffloadStateType.optimizer_states, include, self.offloaded_states):
        reload_adam_states(self.optimizer, device, backend=backend)
        self.offloaded_states.remove(MegatronOffloadStateType.optimizer_states)

    current_platform.synchronize()


def _clear_moe_dispatcher_state(model_chunk):
    """Clear MoE token dispatcher cached state to free memory."""
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


def _get_no_grad_params_by_dtype(model_chunk) -> dict:
    """Get or build the no-grad params grouped by (dtype, is_expert), cached on
    model_chunk. is_expert follows Megatron's own convention (allreduce=False),
    since dense and expert params replicate over different process groups."""
    model_chunk.param_dtype_to_params = getattr(model_chunk, "param_dtype_to_params", defaultdict(list))
    if not model_chunk.param_dtype_to_params:
        for param in model_chunk.parameters():
            if not param.requires_grad:
                param_dtype = param.dtype
                if is_float8tensor(param):
                    param_dtype = torch.uint8
                is_expert = not getattr(param, "allreduce", True)
                model_chunk.param_dtype_to_params[(param_dtype, is_expert)].append(param)
    return model_chunk.param_dtype_to_params


def offload_megatron_no_grad_module(model_chunks: List[Union[DistributedDataParallel, MegatronModule]],
                                    backend=None,
                                    key_prefix: str = "",
                                    ):
    """Offload grad=False parameters (embeddings, frozen layers, etc.)."""
    for chunk_idx, model_chunk in enumerate(model_chunks):
        if isinstance(model_chunk, DistributedDataParallel):
            model_chunk = model_chunk.module
        model_chunk.offloaded_states = getattr(model_chunk, "offloaded_states", set())
        if not needs_offload(MegatronOffloadStateType.model_params,
                             include=[MegatronOffloadStateType.model_params],
                             offloaded_states=model_chunk.offloaded_states):
            continue

        param_dtype_to_params = _get_no_grad_params_by_dtype(model_chunk)

        if getattr(model_chunk, "_offload_no_grad_keys", []):
            # write-once: no-grad params are frozen, backend keys from the first
            # offload are still valid — just free GPU memory, no re-put.
            for params in param_dtype_to_params.values():
                for t in params:
                    t.data = torch.empty(0, dtype=t.dtype, device='cpu')
        else:
            offload_keys = []
            for (param_dtype, is_expert), params in param_dtype_to_params.items():
                if not params:
                    continue
                kind = "expert" if is_expert else "dense"
                key = f"{key_prefix}_no_grad_chunk_{chunk_idx}_{param_dtype}_{kind}"
                backend.put_tensors(key, params, replicated_over=_replication_group(is_expert))
                offload_keys.append({
                    'key': key,
                    'param_dtype': (param_dtype, is_expert),
                })
            model_chunk._offload_no_grad_keys = offload_keys

        _clear_moe_dispatcher_state(model_chunk)
        model_chunk.offloaded_states.add(MegatronOffloadStateType.model_params)

    log_offload_debug(
        f"offload_no_grad_module backend={type(backend).__name__}",
        store_stats=backend.get_stats() if backend is not None else None,
        once_id="no_grad_module",
    )


def reload_megatron_no_grad_module(model_chunks: List[Union[DistributedDataParallel, MegatronModule]],
                                   backend=None):
    """Reload grad=False parameters back to GPU."""
    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')

    for model_chunk in model_chunks:
        if isinstance(model_chunk, DistributedDataParallel):
            model_chunk = model_chunk.module

        model_chunk.offloaded_states = getattr(model_chunk, "offloaded_states", set())
        if not needs_reload(MegatronOffloadStateType.model_params,
                            include=[MegatronOffloadStateType.model_params],
                            offloaded_states=model_chunk.offloaded_states):
            continue

        offload_keys = getattr(model_chunk, "_offload_no_grad_keys", [])
        param_dtype_to_params = getattr(model_chunk, "param_dtype_to_params", {})
        for key_info in offload_keys:
            params = param_dtype_to_params[key_info['param_dtype']]
            backend.get_tensors(key_info['key'], params, device)

        model_chunk.offloaded_states.remove(MegatronOffloadStateType.model_params)

    # write-once: keys are kept so the next offload skips the re-put.


# ---------------------------------------------------------------------------
# DDP buffer offload/reload/cleanup via unified OffloadBackend
# (merged from offload_ddp_buffers.py)
# ---------------------------------------------------------------------------

def _release_ddp_param_views(optimizer: DistributedOptimizer) -> None:
    for ddp_buffer in optimizer.buffers:
        if ddp_buffer.param_data is None:
            continue
        ddp_buffer.param_data.data = torch.empty(0, dtype=ddp_buffer.param_data.dtype, device='cpu')
        for param in ddp_buffer.params:
            if is_float8tensor(param):
                param._data = torch.empty(0, dtype=param._data.dtype, device='cpu')
            else:
                param.data = torch.empty(0, dtype=param.dtype, device='cpu')
        for bucket in ddp_buffer.buckets:
            bucket.param_data.data = torch.empty(0, dtype=bucket.param_data.dtype, device='cpu')


def _offload_ddp_buffers(optimizer: DistributedOptimizer) -> None:
    """Offload DDP ParamAndGradBuffer flat buffers to backend.

    param_data is the post-all-gather full bf16 params, identical across this
    (sub-)optimizer's data-parallel replicas (overlap_param_gather is
    unsupported), so dedup over the (sub-)optimizer's replication group stores
    one copy per group: DP×CP for the dense sub-optimizer, expert-DP for the
    expert one.
    """
    backend = optimizer._offload_backend
    key_prefix = optimizer._offload_key_prefix
    group = _optimizer_replication_group(optimizer)

    keys: List[dict] = []
    total_bytes = 0

    for buffer_idx, ddp_buffer in enumerate(optimizer.buffers):
        if ddp_buffer.param_data is None:
            continue
        if ddp_buffer.param_data.data.numel() == 0:
            continue

        key = f"{key_prefix}_ddp_buffer_{buffer_idx}"
        total_bytes += ddp_buffer.param_data.data.numel() * ddp_buffer.param_data.data.element_size()
        if backend.has_key(key):
            backend.delete_tensors(key, replicated_over=True)
        backend.put_tensors(key, [ddp_buffer.param_data], replicated_over=group)
        keys.append({
            'key': key,
            'buffer_idx': buffer_idx,
        })

    # put_tensors rebinds param_data and shrinks its storage; still break the
    # param/bucket views so nothing reads the emptied buffer before reload.
    _release_ddp_param_views(optimizer)

    # put_tensors(replicated_over=group) already barriers internally — no extra barrier here.
    optimizer._offload_ddp_keys = keys

    log_offload_debug(
        f"offload_ddp_buffers num_buffers={len(keys)} total_data={total_bytes / 1024**2:.1f}MB "
        f"backend={type(backend).__name__}",
        store_stats=backend.get_stats(),
        once_id=f"ddp_offload_{id(optimizer)}",
    )


def _relink_shard_param_views(optimizer: DistributedOptimizer) -> None:
    """Rebind the optimizer's shard param views into the reloaded DDP buffers.

    Megatron builds shard_fp32_groups/shard_float16_groups as views into the
    DDP param buffers at optimizer-construction time. For native fp32 params
    (e.g. LoRA adapters) the shard view IS the main param: the inner torch
    optimizer's param_groups reference these exact objects, and Adam state is
    keyed by them. Offload shrinks the old buffers' storage and reload rebinds
    param.data to a new allocation, so these views must be re-pointed in place
    — replacing the objects would orphan the Adam state keys and silently
    reset optimizer state.
    """
    for model_groups, shard_groups in (
        (optimizer.model_fp32_groups, optimizer.shard_fp32_groups),
        (optimizer.model_float16_groups, optimizer.shard_float16_groups),
    ):
        for model_group, shard_group in zip(model_groups, shard_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                if shard_param is None:  # fp8 params keep no shard view
                    continue
                gbuf_index, _, bucket_id = optimizer.model_param_gbuf_map[model_param]
                bucket_param_data = optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
                if bucket_param_data is None:
                    continue
                world_range = optimizer._get_model_param_range_map(model_param)["gbuf_world_in_bucket"]
                shard_param.data = bucket_param_data.view(-1)[world_range.start:world_range.end]


def _reload_ddp_buffers(optimizer: DistributedOptimizer) -> None:
    """Reload DDP ParamAndGradBuffer flat buffers from backend.

    No entry barrier: dp_rank=0 may start the store read as soon as it arrives
    (the data is its own put from the previous offload, ordered by program
    order), and the get's broadcast is itself the rendezvous with the other
    ranks. This overlaps the read with the wait for straggler ranks instead of
    serializing read-after-barrier.
    """
    backend = optimizer._offload_backend
    keys = getattr(optimizer, '_offload_ddp_keys', [])
    if not keys:
        return

    device = torch.device(f'{current_platform.device_type}:{current_platform.current_device()}')

    for metadata in keys:
        buffer_idx = metadata['buffer_idx']
        key = metadata['key']

        ddp_buffer = optimizer.buffers[buffer_idx]
        backend.get_tensors(key, [ddp_buffer.param_data], device, replicated_over=True)

        for param in ddp_buffer.params[::-1]:
            param_start_index, param_end_index, bucket_id = ddp_buffer.param_index_map[param]
            new_param_data = ddp_buffer._get(
                _param_view_shape(ddp_buffer, param), param_start_index, buffer_type=BufferType.PARAM
            )
            if is_float8tensor(param):
                param._data = new_param_data
            else:
                param.data = new_param_data

        for bucket in ddp_buffer.buckets:
            start_index, end_index = ddp_buffer.bucket_indices[bucket.bucket_id]
            bucket.param_data.data = ddp_buffer._get(
                torch.Size([end_index - start_index]), start_index,
                buffer_type=BufferType.PARAM
            )

    # Shard views (shard_fp32_groups/shard_float16_groups) alias the old
    # buffers' storage and were not rebound above — re-point them at the fresh
    # allocation before anything reads them.
    _relink_shard_param_views(optimizer)

    # Read-before-delete ordering matters only on shared remote stores, where
    # get_tensors(replicated_over=True) barriers per key so every rank has read
    # before cleanup deletes. Local backends' barrier() is a no-op (deletes are
    # rank-local), so the guard costs nothing there.
    _cleanup_ddp_buffers(optimizer, backend)


def _cleanup_ddp_buffers(optimizer: DistributedOptimizer, backend) -> None:
    """Delete stale offloaded DDP buffers from backend."""
    keys = getattr(optimizer, '_offload_ddp_keys', [])
    if not keys:
        return

    for metadata in keys:
        backend.delete_tensors(metadata['key'], replicated_over=True)

    optimizer._offload_ddp_keys = []


def cleanup_ddp_buffers(optimizer, backend) -> None:
    """Delete stale DDP buffer keys from backend after optimizer.step()."""
    if isinstance(optimizer, ChainedOptimizer):
        for sub_optimizer in optimizer.chained_optimizers:
            cleanup_ddp_buffers(sub_optimizer, backend)
        return

    if not isinstance(optimizer, DistributedOptimizer):
        return

    _cleanup_ddp_buffers(optimizer, backend)


def needs_offload(target, include, offloaded_states):
    return target not in offloaded_states and (include is None or target in include)


def needs_reload(target, include, offloaded_states):
    return (include == None or target in include) and (target in offloaded_states)


def offload_adam_states(optimizer, backend=None, key: str = None):
    """Move optimizer states to backend."""
    state_tensors = []
    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            state_tensors.append(state["exp_avg"])
        if "exp_avg_sq" in state:
            state_tensors.append(state["exp_avg_sq"])
    optimizer._adam_offload_key = backend.put_tensors(key, state_tensors, replicated_over=False)


def reload_adam_states(optimizer, device, backend=None):
    """Move optimizer states from backend to device."""
    key = getattr(optimizer, '_adam_offload_key', None)
    if key is None:
        return
    state_tensors = []
    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            state_tensors.append(state["exp_avg"])
        if "exp_avg_sq" in state:
            state_tensors.append(state["exp_avg_sq"])

    backend.get_tensors(key, state_tensors, device, replicated_over=False)
    backend.delete_tensors(key, replicated_over=False)
    optimizer._adam_offload_key = None
