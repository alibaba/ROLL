import gc
import itertools
import types
from typing import Callable, Dict, Union, Iterable, Container, List

from deepspeed import DeepSpeedEngine
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum, OffloadDeviceEnum
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.utils import get_mapping_to_flat_buffer
from deepspeed.utils import logger

import torch
from torch import Tensor

from roll.third_party.deepspeed.offload_states import needs_offload, needs_reload, offload_adam_states, \
    reload_adam_states


def bind_deepspeed_offload_states_func(engine: DeepSpeedEngine):
    if hasattr(engine, 'offload_states') and hasattr(engine, 'reload_states'):
        setattr(engine, 'old_offload_states', engine.offload_states)
        setattr(engine, 'old_reload_states', engine.reload_states)
    engine.offload_states = types.MethodType(engine_offload_states, engine)
    engine.reload_states = types.MethodType(engine_reload_states, engine)
    engine.offload_states = types.MethodType(engine_offload_states, engine)
    engine.offload_non_trainable_parameters = types.MethodType(engine_offload_non_trainable_parameters, engine)
    engine.reload_non_trainable_parameters = types.MethodType(engine_reload_non_trainable_parameters, engine)

    assert isinstance(engine.optimizer, (DeepSpeedZeroOptimizer, DeepSpeedZeRoOffload, DeepSpeedZeroOptimizer_Stage3)), \
        f"offload_states only supports optimizer [DeepSpeedZeroOptimizer, DeepSpeedZeRoOffload, DeepSpeedZeroOptimizer_Stage3], but get {type(engine.optimizer)}"

    optimizer = engine.optimizer
    if hasattr(optimizer, 'offload_states') and hasattr(optimizer, 'reload_states'):
        setattr(optimizer, 'old_offload_states', optimizer.offload_states)
        setattr(optimizer, 'old_reload_states', optimizer.reload_states)

    if isinstance(optimizer, DeepSpeedZeroOptimizer):
        optimizer.offload_states = types.MethodType(stage_1_and_2_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(stage_1_and_2_reload_states, optimizer)
    elif isinstance(optimizer, DeepSpeedZeRoOffload):
        optimizer.offload_states = types.MethodType(parameter_offload_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(parameter_offload_reload_states, optimizer)
    elif isinstance(optimizer, DeepSpeedZeroOptimizer_Stage3):
        optimizer.offload_states = types.MethodType(stage_3_offload_states, optimizer)
        optimizer.reload_states = types.MethodType(stage_3_reload_states, optimizer)
    else:
        raise RuntimeError(f'engine {engine} does not support offload_states func')


def engine_offload_non_trainable_parameters(self: DeepSpeedEngine,
                                            include: Container[OffloadStateTypeEnum] = None,
                                            device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                                            pin_memory: bool = True,
                                            non_blocking: bool = False):
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_offload(OffloadStateTypeEnum.lp_params, include, self.offloaded_states):
        if not hasattr(self, "non_trainable_params"):
            self.non_trainable_params: List[Tensor] = []
            self.non_trainable_params_meta: List[Tensor] = []
            for param in self.module.parameters():
                if not param.requires_grad:
                    if hasattr(param, "ds_id"):
                        self.non_trainable_params.append(param.ds_tensor)
                    else:
                        self.non_trainable_params.append(param)
                    self.non_trainable_params_meta.append(
                        torch.zeros_like(self.non_trainable_params[-1], device="meta"))
            if len(self.non_trainable_params) > 0:
                self.non_trainable_lp_param_buffer = self.flatten(self.non_trainable_params)

        if len(self.non_trainable_params) == 0:
            return

        if pin_memory:
            if not hasattr(self, "non_trainable_lp_param_contiguous_pin_buffer"):
                self.non_trainable_lp_param_contiguous_pin_buffer = get_accelerator().pin_memory(
                    torch.empty_like(self.non_trainable_lp_param_buffer, device=device))
            self.non_trainable_lp_param_contiguous_pin_buffer.copy_(self.non_trainable_lp_param_buffer,
                                                                    non_blocking=non_blocking)
            self.non_trainable_lp_param_buffer.data = self.non_trainable_lp_param_contiguous_pin_buffer
        else:
            self.non_trainable_lp_param_buffer.data = self.non_trainable_lp_param_buffer.to(device,
                                                                                            non_blocking=non_blocking)
        unflatten_tensor = self.unflatten(self.non_trainable_lp_param_buffer, self.non_trainable_params_meta)
        for src_tensor, dist_tensor in zip(unflatten_tensor, self.non_trainable_params):
            dist_tensor.data = src_tensor.data

        self.offloaded_states.add(OffloadStateTypeEnum.lp_params)


def engine_reload_non_trainable_parameters(self: DeepSpeedEngine, include=None, non_blocking: bool = False):
    device = get_accelerator().current_device_name()
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_reload(OffloadStateTypeEnum.lp_params, include, self.offloaded_states) and len(
            self.non_trainable_params) > 0:
        lp_param_cpu_buffer = self.non_trainable_lp_param_contiguous_pin_buffer if hasattr(
            self, "non_trainable_lp_param_contiguous_pin_buffer") else self.non_trainable_lp_param_buffer
        self.non_trainable_lp_param_buffer.data = lp_param_cpu_buffer.data.to(device, non_blocking=non_blocking)

        unflatten_tensor = self.unflatten(self.non_trainable_lp_param_buffer, self.non_trainable_params_meta)
        for src_tensor, dist_tensor in zip(unflatten_tensor, self.non_trainable_params):
            dist_tensor.data = src_tensor.data
        self.offloaded_states.remove(OffloadStateTypeEnum.lp_params)


def engine_offload_states(self: DeepSpeedEngine,
                          include: Container[OffloadStateTypeEnum] = None,
                          device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                          pin_memory: bool = True,
                          non_blocking: bool = False) -> None:
    """Offload the engine's states to the specified device.

    Arguments:
        include: Optional. The set of states to offload. If not provided, all states are offloaded.
        device: Optional. The device to move the ZeRO optimizer buffers to. Currently only `OffloadDeviceEnum.cpu` is supported.
        pin_memory: Optional. Whether to pin the memory of the offloaded states.
        non_blocking: Optional. Whether to offload the states asynchronously.
    """
    param_offload_config = self.zero_offload_param()
    assert param_offload_config is None or param_offload_config.device == OffloadDeviceEnum.none, "Moving states across devices is not supported for offloaded parameters."

    assert isinstance(self.optimizer, (DeepSpeedZeroOptimizer, DeepSpeedZeRoOffload, DeepSpeedZeroOptimizer_Stage3)), \
        f"offload_states only supports optimizer [DeepSpeedZeroOptimizer, DeepSpeedZeRoOffload, DeepSpeedZeroOptimizer_Stage3], but get {type(self.optimizer)}"

    assert self.zero_optimization_stage() > 0, "offload_states only supports optimizer stage > 0."

    if device == OffloadDeviceEnum.none:
        logger.warning("No device specified for offloading states.")
        return

    if device == OffloadDeviceEnum.nvme:
        raise ValueError("NVMe offload is not supported for offloading states.")

    self.offload_non_trainable_parameters(include=include, device=device, pin_memory=pin_memory,
                                          non_blocking=non_blocking)
    self.optimizer.offload_states(include=include, device=device, pin_memory=pin_memory, non_blocking=non_blocking)

    gc.collect()
    get_accelerator().empty_cache()


def engine_reload_states(self, include: Container[OffloadStateTypeEnum] = None, non_blocking: bool = False) -> None:
    """Reload the engine states to the original device.

    Arguments:
        include: Optional. The set of states to offload. If not provided, all states are reloaded.
        non_blocking: Optional. Whether to offload the states asynchronously.
    """
    self.reload_non_trainable_parameters(include=include, non_blocking=non_blocking)
    self.optimizer.reload_states(include=include, non_blocking=non_blocking)


def stage_1_and_2_offload_states(self: DeepSpeedZeroOptimizer,
                                 include: Container[OffloadStateTypeEnum] = None,
                                 device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                                 pin_memory: bool = True,
                                 non_blocking: bool = False):
    device = device.value
    self.offloaded_states = getattr(self, "offloaded_states", set())
    # HP param
    if needs_offload(OffloadStateTypeEnum.hp_params, include, self.offloaded_states) and not \
            self.single_partition_of_fp32_groups[0].is_cpu:
        if pin_memory:
            if not hasattr(self, "hp_params_pin_buffers"):
                self.hp_params_pin_buffers = [
                    get_accelerator().pin_memory(torch.empty_like(t, device=device))
                    for t in self.single_partition_of_fp32_groups
                ]
            for src_tensor, dest_buf in zip(self.single_partition_of_fp32_groups, self.hp_params_pin_buffers):
                dest_buf.copy_(src_tensor, non_blocking=non_blocking)
                src_tensor.data = dest_buf
        else:
            for buf in self.single_partition_of_fp32_groups:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
        self.offloaded_states.add(OffloadStateTypeEnum.hp_params)

    # LP param
    if needs_offload(OffloadStateTypeEnum.lp_params, include, self.offloaded_states) and not self.bit16_groups_flat[
        0].is_cpu:
        # NOTE: Only supports offloading parameters in optimizer here
        if pin_memory:
            if not hasattr(self, "lp_params_pin_buffers"):
                self.lp_params_pin_buffers = [
                    get_accelerator().pin_memory(torch.empty_like(t, device=device))
                    for t in self.bit16_groups_flat
                ]
            for src_tensor, dest_buf in zip(self.bit16_groups_flat, self.lp_params_pin_buffers):
                dest_buf.copy_(src_tensor, non_blocking=non_blocking)
                src_tensor.data = dest_buf.data
        else:
            for buf in self.bit16_groups_flat:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.parallel_partitioned_bit16_groups = []
        self.offloaded_states.add(OffloadStateTypeEnum.lp_params)

    # LP grad
    # NOTE: There seems to be no grad cache here
    if needs_offload(OffloadStateTypeEnum.lp_grads, include, self.offloaded_states):
        pass

    # contiguous bucket
    if needs_offload(OffloadStateTypeEnum.contiguous_grad_buffer, include, self.offloaded_states):
        self.ipg_buffer_meta = []
        if hasattr(self, "ipg_buffer") and self.ipg_buffer is not None:
            # Record properties like shape, strides, etc. as a meta tensor
            for buffer in self.ipg_buffer:
                self.ipg_buffer_meta.append(buffer.to("meta"))
            self.ipg_buffer = None
        if hasattr(self, "temp_grad_buffer_for_gpu_offload"):
            if pin_memory:
                if not hasattr(self, "temp_grad_buffer_for_gpu_offload_pin_buffer"):
                    self.temp_grad_buffer_for_gpu_offload_pin_buffer = get_accelerator().pin_memory(
                        torch.empty_like(self.temp_grad_buffer_for_gpu_offload, device=device))
                self.temp_grad_buffer_for_gpu_offload_pin_buffer.copy_(self.temp_grad_buffer_for_gpu_offload,
                                                                       non_blocking=non_blocking)
                self.temp_grad_buffer_for_gpu_offload.data = self.temp_grad_buffer_for_gpu_offload_pin_buffer
            else:
                self.temp_grad_buffer_for_gpu_offload.data = self.temp_grad_buffer_for_gpu_offload.data.to(device,
                                                                                                           non_blocking=non_blocking)
        self.averaged_gradients = {}
        self.offloaded_states.add(OffloadStateTypeEnum.contiguous_grad_buffer)

    # Adam
    if needs_offload(OffloadStateTypeEnum.optim_states, include, self.offloaded_states):
        offload_adam_states(self.optimizer, device, pin_memory=pin_memory, non_blocking=non_blocking)
        self.offloaded_states.add(OffloadStateTypeEnum.optim_states)

    # NOTE: Clean up extra references, hp_mapping contains a reference to all flat tensors
    for group in self.bit16_groups:
        for param in group:
            param._hp_mapping = None
    self._link_all_hp_params()


def stage_1_and_2_reload_states(self: DeepSpeedZeroOptimizer, include=None, non_blocking: bool = False):
    device = get_accelerator().current_device_name()
    self.offloaded_states = getattr(self, "offloaded_states", set())
    # HP param
    if needs_reload(OffloadStateTypeEnum.hp_params, include, self.offloaded_states):
        if hasattr(self, "hp_params_pin_buffers"):
            for src, dest in zip(self.hp_params_pin_buffers, self.single_partition_of_fp32_groups):
                dest.data = src.to(device, non_blocking=non_blocking)
        else:
            for buf in self.single_partition_of_fp32_groups:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
        self.offloaded_states.remove(OffloadStateTypeEnum.hp_params)

    # LP Param
    if needs_reload(OffloadStateTypeEnum.lp_params, include, self.offloaded_states):
        if hasattr(self, "lp_params_pin_buffers"):
            for src, dest in zip(self.lp_params_pin_buffers, self.bit16_groups_flat):
                dest.data = src.to(device, non_blocking=non_blocking)
        else:
            for buf in self.bit16_groups_flat:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)
        self.offloaded_states.remove(OffloadStateTypeEnum.lp_params)

    # LP grad
    if needs_reload(OffloadStateTypeEnum.lp_grads, include, self.offloaded_states):
        pass

    # contiguous bucket
    if needs_reload(OffloadStateTypeEnum.contiguous_grad_buffer, include, self.offloaded_states):
        self.ipg_buffer = []
        for buffer in self.ipg_buffer_meta:
            self.ipg_buffer.append(torch.empty_like(buffer, device=device))
        if hasattr(self, "temp_grad_buffer_for_gpu_offload"):
            cpu_buffer = self.temp_grad_buffer_for_gpu_offload_pin_buffer if hasattr(
                self, "temp_grad_buffer_for_gpu_offload_pin_buffer") else self.temp_grad_buffer_for_gpu_offload
            self.temp_grad_buffer_for_gpu_offload.data = cpu_buffer.data.to(device, non_blocking=non_blocking)
        self.averaged_gradients = {}
        self.offloaded_states.remove(OffloadStateTypeEnum.contiguous_grad_buffer)

    # Adam
    if needs_reload(OffloadStateTypeEnum.optim_states, include, self.offloaded_states):
        reload_adam_states(self.optimizer, device, non_blocking=non_blocking)
        self.offloaded_states.remove(OffloadStateTypeEnum.optim_states)

    # NOTE: Restore link
    for group in self.bit16_groups:
        for param in group:
            param._hp_mapping = None
    self._link_all_hp_params()

    if non_blocking:
        get_accelerator().synchronize()


def stage_3_offload_states(self: DeepSpeedZeroOptimizer_Stage3,
                           include: Container[OffloadStateTypeEnum] = None,
                           device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                           pin_memory: bool = True,
                           non_blocking: bool = False):
    device = device.value
    self.offloaded_states = getattr(self, "offloaded_states", set())
    self.empty_partition_cache()

    # HP param
    if needs_offload(OffloadStateTypeEnum.hp_params, include, self.offloaded_states) and not \
            self.fp32_partitioned_groups_flat[0].is_cpu:
        if pin_memory:
            if not hasattr(self, "hp_params_pin_buffers"):
                self.hp_params_pin_buffers = [
                    get_accelerator().pin_memory(torch.empty_like(t, device=device))
                    for t in self.fp32_partitioned_groups_flat
                ]

            for src_tensor, dest_buf in zip(self.fp32_partitioned_groups_flat, self.hp_params_pin_buffers):
                dest_buf.copy_(src_tensor, non_blocking=non_blocking)
                src_tensor.data = dest_buf
        else:
            for buf in self.fp32_partitioned_groups_flat:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
        self.offloaded_states.add(OffloadStateTypeEnum.hp_params)

    # LP param
    if needs_offload(OffloadStateTypeEnum.lp_params, include, self.offloaded_states):
        if pin_memory:
            if not hasattr(self, "lp_param_contiguous_pin_buffer"):
                self.lp_param_contiguous_pin_buffer = get_accelerator().pin_memory(
                    torch.empty_like(self.lp_param_buffer, device=device))
            # NOTE: lp_param_buffer stores parameter order retrieved from optimizer
            #       When offloading, first move lp_param_buffer.cpu()
            #       Then copy tensor.data to model's tensor.data, this step may also have order inconsistency issues
            self.lp_param_contiguous_pin_buffer.copy_(self.lp_param_buffer, non_blocking=non_blocking)
            cpu_buffer = self.lp_param_contiguous_pin_buffer
        else:
            cpu_buffer = self.lp_param_buffer.to(device, non_blocking=non_blocking)

        self.lp_param_buffer.data = cpu_buffer
        parameter_partitions: List[Tensor] = [param.ds_tensor for sub_group in self.fp16_groups for param in
                                              sub_group]
        for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(parameter_partitions):
            tensor.data = cpu_buffer.narrow(0, offset, tensor_numel)

        self.fp16_partitioned_groups_flat.clear()
        self.offloaded_states.add(OffloadStateTypeEnum.lp_params)

    # LP grad
    if needs_offload(OffloadStateTypeEnum.lp_grads, include,
                     self.offloaded_states) and not self.grad_partitions_flat_buffer.is_cpu:
        if pin_memory:
            if not hasattr(self, "lp_grad_partitions_flat_pin_buffers"):
                self.lp_grad_partitions_flat_pin_buffers = get_accelerator().pin_memory(
                    torch.empty_like(self.grad_partitions_flat_buffer, device=device))
            self.lp_grad_partitions_flat_pin_buffers.copy_(self.grad_partitions_flat_buffer,
                                                           non_blocking=non_blocking)
            self.grad_partitions_flat_buffer.data = self.lp_grad_partitions_flat_pin_buffers
        else:
            self.grad_partitions_flat_buffer.data = self.grad_partitions_flat_buffer.data.to(device)
        self.averaged_gradients = {}
        # NOTE: self.__param_id_to_grad_partition stores a reference to grad_partitions_flat_buffer, patch modifications need to use name mangling
        setattr(self, "_DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition", {})

        self.offloaded_states.add(OffloadStateTypeEnum.lp_grads)

    # contiguous bucket
    if needs_offload(OffloadStateTypeEnum.contiguous_grad_buffer, include, self.offloaded_states):
        if hasattr(self, "_DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer"):
            # Record properties like shape, strides, etc. as a meta tensor
            self.grad_buffer_meta = getattr(self, "_DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer").to("meta")
            setattr(self, "_DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer", None)
            self.offloaded_states.add(OffloadStateTypeEnum.contiguous_grad_buffer)

    # Adam
    if needs_offload(OffloadStateTypeEnum.optim_states, include, self.offloaded_states):
        offload_adam_states(self.optimizer, device, pin_memory=pin_memory, non_blocking=non_blocking)
        self.offloaded_states.add(OffloadStateTypeEnum.optim_states)


def stage_3_reload_states(self: DeepSpeedZeroOptimizer_Stage3, include=None, non_blocking: bool = False):
    device = get_accelerator().current_device_name()
    self.offloaded_states = getattr(self, "offloaded_states", set())
    # HP param
    if needs_reload(OffloadStateTypeEnum.hp_params, include, self.offloaded_states):
        if hasattr(self, "hp_params_pin_buffers"):
            for src, dest in zip(self.hp_params_pin_buffers, self.fp32_partitioned_groups_flat):
                dest.data = src.to(device, non_blocking=non_blocking)
        else:
            for buf in self.fp32_partitioned_groups_flat:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
        self.offloaded_states.remove(OffloadStateTypeEnum.hp_params)

    # LP Param
    if needs_reload(OffloadStateTypeEnum.lp_params, include, self.offloaded_states):
        cpu_buffer = self.lp_param_contiguous_pin_buffer if hasattr(
            self, "lp_param_contiguous_pin_buffer") else self.lp_param_buffer
        self.lp_param_buffer.data = cpu_buffer.data.to(device, non_blocking=non_blocking)
        self._set_fp16_partitioned_groups_flat()

        # NOTE: Here we iterate over self.module.parameters(), while lp_param_buffer contains parameters from fp16 groups, the parameter order is inconsistent here
        #       Here [p.ds_tensor for p in self.module.parameters()] needs to be reordered according to self.fp16_groups order
        parameter_partitions: List[Tensor] = [param.ds_tensor for sub_group in self.fp16_groups for param in sub_group]
        for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(parameter_partitions):
            tensor.data = self.lp_param_buffer.data.narrow(0, offset, tensor_numel)
        self.offloaded_states.remove(OffloadStateTypeEnum.lp_params)

    # LP grad
    if needs_reload(OffloadStateTypeEnum.lp_grads, include, self.offloaded_states):
        if hasattr(self, "lp_grad_partitions_flat_pin_buffers"):
            self.grad_partitions_flat_buffer.data = self.lp_grad_partitions_flat_pin_buffers.to(
                device, non_blocking=non_blocking)
        else:
            self.grad_partitions_flat_buffer.data = self.grad_partitions_flat_buffer.data.to(
                device, non_blocking=non_blocking)
        self.averaged_gradients = {}

        offset = 0
        all_params = list(itertools.chain.from_iterable(self.fp16_groups))

        param_id_to_grad_partition = getattr(self, "_DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition")
        for param in all_params:
            param_id_to_grad_partition[param.ds_id] = self.grad_partitions_flat_buffer.narrow(
                0, offset, param.partition_numel())
            offset += param.partition_numel()

        self.offloaded_states.remove(OffloadStateTypeEnum.lp_grads)

    # contiguous bucket
    if needs_reload(OffloadStateTypeEnum.contiguous_grad_buffer, include, self.offloaded_states):
        setattr(self, "_DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer", torch.empty_like(self.grad_buffer_meta, device=device))
        self.offloaded_states.remove(OffloadStateTypeEnum.contiguous_grad_buffer)

    # Adam
    if needs_reload(OffloadStateTypeEnum.optim_states, include, self.offloaded_states):
        reload_adam_states(self.optimizer, device, non_blocking=non_blocking)
        self.offloaded_states.remove(OffloadStateTypeEnum.optim_states)

    if non_blocking:
        get_accelerator().synchronize()


def parameter_offload_offload_states(self: DeepSpeedZeRoOffload,
                                     include: Container[OffloadStateTypeEnum] = None,
                                     device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                                     pin_memory: bool = True,
                                     non_blocking: bool = False):
    device = device.value
    self.empty_partition_cache()
    self.offloaded_states = getattr(self, "offloaded_states", set())

    if needs_offload(OffloadStateTypeEnum.lp_params, include, self.offloaded_states):
        # NOTE: This won't execute since non_trainable_params are all handled in the engine
        if not hasattr(self, "trainable_params"):
            self.trainable_params = [param.ds_tensor for param in self.module.parameters() if param.requires_grad]
        if len(self.trainable_params) == 0:
            return

        if not hasattr(self, "lp_param_buffer"):
            from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
            self.lp_param_buffer = DeepSpeedZeroOptimizer_Stage3.defragment(self.trainable_params)

        if pin_memory:
            if not hasattr(self, "lp_param_contiguous_pin_buffer"):
                self.lp_param_contiguous_pin_buffer = get_accelerator().pin_memory(
                    torch.empty_like(self.lp_param_buffer, device=device))
            self.lp_param_contiguous_pin_buffer.copy_(self.lp_param_buffer, non_blocking=non_blocking)
            lp_param_cpu_buffer = self.lp_param_contiguous_pin_buffer
        else:
            lp_param_cpu_buffer = self.lp_param_buffer.to(device, non_blocking=non_blocking)
        self.lp_param_buffer.data = lp_param_cpu_buffer
        for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(
                [p.ds_tensor for p in self.module.parameters() if p.requires_grad]):
            tensor.data = self.lp_param_buffer.data.narrow(0, offset, tensor_numel)

        self.offloaded_states.add(OffloadStateTypeEnum.lp_params)


def parameter_offload_reload_states(self: DeepSpeedZeRoOffload, include=None, non_blocking: bool = False):
    device = get_accelerator().current_device_name()
    self.offloaded_states = getattr(self, "offloaded_states", set())
    if needs_reload(OffloadStateTypeEnum.lp_params, include, self.offloaded_states):
        lp_param_cpu_buffer = self.lp_param_contiguous_pin_buffer if hasattr(
            self, "lp_param_contiguous_pin_buffer") else self.lp_param_buffer
        self.lp_param_buffer.data = lp_param_cpu_buffer.data.to(device, non_blocking=non_blocking)

        for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(
                [p.ds_tensor for p in self.module.parameters() if p.requires_grad]):
            tensor.data = self.lp_param_buffer.narrow(0, offset, tensor_numel)

        self.offloaded_states.remove(OffloadStateTypeEnum.lp_params)
