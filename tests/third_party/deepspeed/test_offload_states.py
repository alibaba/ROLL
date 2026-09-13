# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import time

# DeepSpeed Team

import pytest

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
import torch

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from roll.third_party.deepspeed.offload_states import get_state_devices
from roll.third_party.deepspeed.offload_states_patch import bind_deepspeed_offload_states_func

import deepspeed
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum
from deepspeed.utils import safe_get_local_fp32_param, safe_get_local_optimizer_state

from tests.third_party.deepspeed.common import DistributedTest
from tests.third_party.deepspeed.simple_model import random_dataloader, SimpleModel


def validate_device(model, device: torch.device, include) -> None:

    def compare_device(state) -> bool:
        devices = get_state_devices(model, state)
        return len(devices) == 1 and device in devices

    for state in OffloadStateTypeEnum:
        if include is None or state in include:
            if state == OffloadStateTypeEnum.contiguous_grad_buffer and device == torch.device("cpu"):
                assert (
                    len(get_state_devices(model, state)) == 0
                ), f"State {state} must be removed after offload_states()"
            else:
                assert compare_device(state), f"State {state} is not on device {device}"


def run_model(model, config_dict, hidden_dim, dtype, include, pin_memory, non_blocking, optimizer=None):
    # Currently we only support OffloadDeviceEnum.cpu
    offload_device = OffloadDeviceEnum.cpu

    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=config_dict, optimizer=optimizer
    )
    bind_deepspeed_offload_states_func(model)
    data_loader = random_dataloader(
        model=model, total_samples=10, hidden_dim=hidden_dim, device=model.device, dtype=dtype
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    dist.barrier()
    for batch in data_loader:
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()

        hp_params_expected = [safe_get_local_fp32_param(p).clone() for p in trainable_params]
        lp_params_expected = [p.ds_tensor.clone() for p in model.parameters()]
        lp_grads_expected = model.optimizer.grad_partitions_flat_buffer.clone()
        adam_exp_avg_expected = [safe_get_local_optimizer_state(p, "exp_avg").clone() for p in trainable_params]
        adam_exp_avg_sq = [safe_get_local_optimizer_state(p, "exp_avg_sq").clone() for p in trainable_params]

        # Start offloading
        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=include, device=offload_device, pin_memory=pin_memory, non_blocking=non_blocking)
        alloc_after_offload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_before_offload
        ), f"Allocated memory should decrease after offload, ({alloc_before_offload}, {alloc_after_offload})"

        validate_device(model, torch.device(offload_device.value), include)

        # Reload states
        model.reload_states()

        alloc_after_reload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_after_reload
        ), f"Allocated memory should increase after offload back, ({alloc_after_offload}, {alloc_after_reload})"

        # Verify restored states
        hp_param_restored = [safe_get_local_fp32_param(p) for p in trainable_params]
        for hp_param_expected, hp_param_restored in zip(hp_params_expected, hp_param_restored):
            assert torch.equal(hp_param_expected, hp_param_restored)

        lp_param_restored = [p.ds_tensor for p in model.parameters()]

        for lp_param_expected, lp_param_restored in zip(lp_params_expected, lp_param_restored):
            assert torch.equal(lp_param_expected, lp_param_restored)

        assert torch.equal(lp_grads_expected, model.optimizer.grad_partitions_flat_buffer)

        adam_exp_avg_restored = [safe_get_local_optimizer_state(p, "exp_avg") for p in trainable_params]
        for adam_exp_avg_expected, adam_exp_avg_restored in zip(adam_exp_avg_expected, adam_exp_avg_restored):
            assert torch.equal(adam_exp_avg_expected, adam_exp_avg_restored)

        adam_exp_avg_sq_restored = [safe_get_local_optimizer_state(p, "exp_avg_sq") for p in trainable_params]
        for adam_exp_avg_sq_expected, adam_exp_avg_sq_restored in zip(adam_exp_avg_sq, adam_exp_avg_sq_restored):
            assert torch.equal(adam_exp_avg_sq_expected, adam_exp_avg_sq_restored)

        if model.zero_offload_optimizer() is None or model.zero_offload_optimizer().device != "cpu":
            validate_device(model, torch.device(get_accelerator().current_device_name()), include)

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


def run_model_infer(model, config_dict, hidden_dim, dtype, include, pin_memory, non_blocking):
    # Currently we only support OffloadDeviceEnum.cpu
    offload_device = OffloadDeviceEnum.cpu

    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    bind_deepspeed_offload_states_func(model)
    data_loader = random_dataloader(
        model=model, total_samples=10, hidden_dim=hidden_dim, device=model.device, dtype=dtype
    )
    dist.barrier()
    for batch in data_loader:
        with torch.no_grad():
            model(batch[0], batch[1])

        lp_params_expected = [p.ds_tensor.clone() for p in model.parameters()]
        # Start offloading
        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=include, device=offload_device, pin_memory=pin_memory, non_blocking=non_blocking)

        alloc_after_offload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_before_offload
        ), f"Allocated memory should decrease after offload, ({alloc_before_offload}, {alloc_after_offload})"

        validate_device(model, torch.device(offload_device.value), include)

        # Reload states
        model.reload_states()

        alloc_after_reload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_after_reload
        ), f"Allocated memory should increase after offload back, ({alloc_after_offload}, {alloc_after_reload})"

        # Verify restored states

        lp_param_restored = [p.ds_tensor for p in model.parameters()]

        for lp_param_expected, lp_param_restored in zip(lp_params_expected, lp_param_restored):
            assert torch.equal(lp_param_expected, lp_param_restored)

        validate_device(model, torch.device(get_accelerator().current_device_name()), include)

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


def run_model_stage_1_2(model, config_dict, hidden_dim, dtype, include, pin_memory, non_blocking, optimizer=None):
    # Currently we only support OffloadDeviceEnum.cpu
    offload_device = OffloadDeviceEnum.cpu

    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=config_dict, optimizer=optimizer
    )
    data_loader = random_dataloader(
        model=model, total_samples=10, hidden_dim=hidden_dim, device=model.device, dtype=dtype
    )
    bind_deepspeed_offload_states_func(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    dist.barrier()
    all_alloc_after_reload = None
    part_alloc_after_reload = None
    for batch in data_loader:
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()

        hp_params_expected = [p.get_full_hp_param().clone() for p in trainable_params]
        lp_params_expected = [p.clone() for p in model.parameters()]
        adam_exp_avg_expected = [
            p._hp_mapping.get_optim_state_fragment("exp_avg").clone()
            for p in trainable_params
            if p in model.optimizer.params_in_partition
        ]
        adam_exp_avg_sq = [
            p._hp_mapping.get_optim_state_fragment("exp_avg_sq").clone()
            for p in trainable_params
            if p in model.optimizer.params_in_partition
        ]

        # Start offloading
        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=include, device=offload_device, pin_memory=pin_memory, non_blocking=non_blocking)
        alloc_after_offload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_before_offload
        ), f"Allocated memory should decrease after offload, ({alloc_before_offload}, {alloc_after_offload})"

        # Reload states
        # NOTE: 测试分批reload
        if not model.zero_cpu_offload() and include is None:
            model.reload_states(include=[OffloadStateTypeEnum.lp_params])
            model.reload_states(
                include=[
                    OffloadStateTypeEnum.hp_params,
                    OffloadStateTypeEnum.optim_states,
                    OffloadStateTypeEnum.lp_grads,
                    OffloadStateTypeEnum.contiguous_grad_buffer,
                ]
            )
            part_alloc_after_reload = get_accelerator().memory_allocated()
            if all_alloc_after_reload is not None:
                assert all_alloc_after_reload == part_alloc_after_reload, (
                    f"part reload and all reload should yield the same result"
                    f"part_alloc_after_reload={part_alloc_after_reload},"
                    f"all_alloc_after_reload={all_alloc_after_reload}"
                )
        else:
            model.reload_states()
            all_alloc_after_reload = get_accelerator().memory_allocated()
            if part_alloc_after_reload is not None:
                assert all_alloc_after_reload == part_alloc_after_reload, (
                    f"part reload and all reload should yield the same result"
                    f"part_alloc_after_reload={part_alloc_after_reload},"
                    f"all_alloc_after_reload={all_alloc_after_reload}"
                )

        alloc_after_reload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_after_reload
        ), f"Allocated memory should increase after offload back, ({alloc_after_offload}, {alloc_after_reload})"

        # Verify restored states
        hp_param_restored = [p.get_full_hp_param() for p in trainable_params]
        for hp_param_expected, hp_param_restored in zip(hp_params_expected, hp_param_restored):
            assert torch.equal(hp_param_expected, hp_param_restored)

        lp_param_restored = list(model.parameters())

        for lp_param_expected, lp_param_restored in zip(lp_params_expected, lp_param_restored):
            assert torch.equal(lp_param_expected, lp_param_restored)

        adam_exp_avg_restored = [
            p._hp_mapping.get_optim_state_fragment("exp_avg")
            for p in trainable_params
            if p in model.optimizer.params_in_partition
        ]
        for adam_exp_avg_expected, adam_exp_avg_restored in zip(adam_exp_avg_expected, adam_exp_avg_restored):
            assert torch.equal(adam_exp_avg_expected, adam_exp_avg_restored)

        adam_exp_avg_sq_restored = [
            p._hp_mapping.get_optim_state_fragment("exp_avg_sq")
            for p in trainable_params
            if p in model.optimizer.params_in_partition
        ]
        for adam_exp_avg_sq_expected, adam_exp_avg_sq_restored in zip(adam_exp_avg_sq, adam_exp_avg_sq_restored):
            assert torch.equal(adam_exp_avg_sq_expected, adam_exp_avg_sq_restored)

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


def run_model_memory_dump(model, config_dict, hidden_dim, dtype, include, pin_memory, non_blocking, optimizer=None):
    # Currently we only support OffloadDeviceEnum.cpu
    offload_device = OffloadDeviceEnum.cpu

    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=config_dict, optimizer=optimizer
    )
    bind_deepspeed_offload_states_func(model)
    dist.barrier()
    for batch in range(3):
        # Start offloading
        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=include, device=offload_device, pin_memory=pin_memory, non_blocking=non_blocking)
        alloc_after_offload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_before_offload
        ), f"Allocated memory should decrease after offload, ({alloc_before_offload}, {alloc_after_offload})"

        # Reload states
        model.reload_states()

        alloc_after_reload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_after_reload
        ), f"Allocated memory should increase after offload back, ({alloc_after_offload}, {alloc_after_reload})"

    model.offload_states()
    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


# The full offload state matrix takes too long in NPU CI.
@pytest.mark.skip_on_npu
class TestOffloadStates(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2

    @pytest.mark.parametrize(
        "included_state",
        [
            OffloadStateTypeEnum.hp_params,
            OffloadStateTypeEnum.lp_params,
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.lp_grads,
            OffloadStateTypeEnum.contiguous_grad_buffer,
            None,
        ],
    )
    @pytest.mark.parametrize("pin_memory", [False, True])
    @pytest.mark.parametrize("non_blocking", [False, True])
    @pytest.mark.parametrize("stage", [1, 2, 3])
    @pytest.mark.parametrize("partial_param", [True, False])
    @pytest.mark.parametrize("optimizer_offload", [True, False])
    def test_offload_states(self, included_state, pin_memory, non_blocking, stage, optimizer_offload, partial_param):
        if optimizer_offload:
            if included_state in [
                OffloadStateTypeEnum.hp_params,
                OffloadStateTypeEnum.optim_states,
                OffloadStateTypeEnum.lp_grads,
            ]:
                return
        if stage in [1, 2]:
            if included_state == OffloadStateTypeEnum.contiguous_grad_buffer and not optimizer_offload:
                return
            if included_state in [OffloadStateTypeEnum.lp_grads]:
                return

        hidden_dim = 1024

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 1e-6}},
            "zero_optimization": {
                "stage": stage,
            },
        }
        config_dict["bf16"] = {"enabled": True}

        if optimizer_offload:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

        if stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = SimpleModel(hidden_dim, nlayers=4)
        else:
            model = SimpleModel(hidden_dim, nlayers=4)

        if partial_param:
            for param in model.linears[0].parameters():
                param.requires_grad = False

        include = None if included_state is None else [included_state]
        if stage == 3:
            run_model(model, config_dict, hidden_dim, torch.bfloat16, include, pin_memory, non_blocking)
        else:
            run_model_stage_1_2(model, config_dict, hidden_dim, torch.bfloat16, include, pin_memory, non_blocking)

    @pytest.mark.parametrize("included_state", [OffloadStateTypeEnum.lp_params])
    @pytest.mark.parametrize("pin_memory", [False, True])
    @pytest.mark.parametrize("non_blocking", [False, True])
    def test_offload_states_with_zero3_infer_only(self, included_state, pin_memory, non_blocking):
        hidden_dim = 1024

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
            },
        }
        config_dict["bf16"] = {"enabled": True}

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim, nlayers=4)

        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        include = None if included_state is None else [included_state]
        run_model_infer(model, config_dict, hidden_dim, torch.bfloat16, include, pin_memory, non_blocking)

    @pytest.mark.parametrize(
        "included_state",
        [
            OffloadStateTypeEnum.hp_params,
            OffloadStateTypeEnum.lp_params,
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.lp_grads,
            OffloadStateTypeEnum.contiguous_grad_buffer,
            None,
        ],
    )
    @pytest.mark.parametrize("pin_memory", [True, False])
    @pytest.mark.parametrize("non_blocking", [True, False])
    @pytest.mark.parametrize("stage", [1, 2, 3])
    @pytest.mark.parametrize("optimizer_offload", [True, False])
    @pytest.mark.parametrize("partial_param", [True, False])
    @pytest.mark.parametrize("with_optim_params", [True, False])
    def test_offload_states_zero(
        self, included_state, pin_memory, non_blocking, stage, optimizer_offload, partial_param, with_optim_params
    ):
        if optimizer_offload:
            if included_state in [
                OffloadStateTypeEnum.hp_params,
                OffloadStateTypeEnum.optim_states,
                OffloadStateTypeEnum.lp_grads,
            ]:
                return
        if stage in [1, 2]:
            if included_state == OffloadStateTypeEnum.contiguous_grad_buffer and not optimizer_offload:
                return
            if included_state in [OffloadStateTypeEnum.lp_grads]:
                return

        hidden_dim = 1024

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": stage,
            },
        }
        config_dict["bf16"] = {"enabled": True}

        optimizer_cls = FusedAdam
        if optimizer_offload:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            optimizer_cls = DeepSpeedCPUAdam

        if stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = SimpleModel(hidden_dim, nlayers=4)
        else:
            model = SimpleModel(hidden_dim, nlayers=4)

        if partial_param:
            for param in model.linears[0].parameters():
                param.requires_grad = False

        if with_optim_params:
            optim_params = [
                {
                    "params": [
                        param
                        for i in range(len(model.linears))
                        if i % 2 == 1
                        for param in model.linears[i].parameters()
                        if param.requires_grad
                    ],
                    "lr": 1e-3,
                },
                {
                    "params": [
                        param
                        for i in range(len(model.linears))
                        if i % 2 == 0
                        for param in model.linears[i].parameters()
                        if param.requires_grad
                    ],
                    "lr": 1e-4,
                },
            ]
        else:
            optim_params = model.parameters()

        optimizer = optimizer_cls(optim_params)

        include = None if included_state is None else [included_state]
        if stage == 3:
            run_model(model, config_dict, hidden_dim, torch.bfloat16, include, pin_memory, non_blocking, optimizer)
        else:
            run_model_stage_1_2(
                model, config_dict, hidden_dim, torch.bfloat16, include, pin_memory, non_blocking, optimizer
            )

    # NOTE: 只forward 没有[OffloadStateTypeEnum.optim_states, OffloadStateTypeEnum.lp_grads]
    @pytest.mark.parametrize(
        "included_state",
        [
            OffloadStateTypeEnum.hp_params,
            OffloadStateTypeEnum.lp_params,
            OffloadStateTypeEnum.contiguous_grad_buffer,
            None,
        ],
    )
    @pytest.mark.parametrize("pin_memory", [True, True])
    @pytest.mark.parametrize("non_blocking", [True, False])
    @pytest.mark.parametrize("stage", [1, 2, 3])
    @pytest.mark.parametrize("optimizer_offload", [True, False])
    @pytest.mark.parametrize("partial_param", [True, False])
    @pytest.mark.parametrize("with_optim_params", [True, False])
    def test_offload_states_zero_memory_dump(
        self, included_state, pin_memory, non_blocking, stage, optimizer_offload, partial_param, with_optim_params
    ):

        if optimizer_offload:
            if included_state in [
                OffloadStateTypeEnum.hp_params,
                OffloadStateTypeEnum.optim_states,
                OffloadStateTypeEnum.lp_grads,
            ]:
                return
        if stage in [1, 2]:
            if included_state == OffloadStateTypeEnum.contiguous_grad_buffer and not optimizer_offload:
                return

        hidden_dim = 1024

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": stage,
            },
        }
        config_dict["bf16"] = {"enabled": True}

        optimizer_cls = FusedAdam
        if optimizer_offload:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            optimizer_cls = DeepSpeedCPUAdam

        MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
        torch.cuda.memory._record_memory_history(
            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
        )

        if stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = SimpleModel(hidden_dim, nlayers=4)
        else:
            model = SimpleModel(hidden_dim, nlayers=4)

        if partial_param:
            for param in model.linears[0].parameters():
                param.requires_grad = False

        if with_optim_params:
            optim_params = [
                {
                    "params": [
                        param
                        for i in range(len(model.linears))
                        if i % 2 == 1
                        for param in model.linears[i].parameters()
                        if param.requires_grad
                    ],
                    "lr": 1e-3,
                },
                {
                    "params": [
                        param
                        for i in range(len(model.linears))
                        if i % 2 == 0
                        for param in model.linears[i].parameters()
                        if param.requires_grad
                    ],
                    "lr": 1e-4,
                },
            ]
        else:
            optim_params = model.parameters()

        optimizer = optimizer_cls(optim_params)
        include = None if included_state is None else [included_state]

        run_model_memory_dump(
            model, config_dict, hidden_dim, torch.bfloat16, include, pin_memory, non_blocking, optimizer
        )

        if dist.get_rank() == 0:
            t0 = torch.randint(0, 100, (1024, 1024), device="cuda")
            t1 = torch.randint(0, 100, (1024, 1024), device="cuda")
            t2 = torch.randint(0, 100, (1024, 1024), device="cuda")

            tensor_list = [t0, t1, t2]
            tensor_meta_list = [torch.zeros_like(t, device="meta") for t in tensor_list]

            from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

            flat_tensor = _flatten_dense_tensors(tensor_list)
            del tensor_list, t0, t1, t2
            unflattened_tensor = _unflatten_dense_tensors(flat_tensor, tensor_meta_list)

            t1_narrow = flat_tensor.narrow(0, 0, tensor_meta_list[0].numel())
            t2_narrow = flat_tensor.narrow(0, tensor_meta_list[0].numel(), tensor_meta_list[1].numel())
            t3_narrow = flat_tensor.narrow(
                0, tensor_meta_list[0].numel() + tensor_meta_list[1].numel(), tensor_meta_list[2].numel()
            )

            dump_file = (
                f"./memory_dump/snapshot_{included_state}_{pin_memory}_{non_blocking}_"
                f"{stage}_{optimizer_offload}_{partial_param}_{with_optim_params}_{os.environ['RANK']}.pickle"
            )
            os.makedirs(os.path.dirname(dump_file), exist_ok=True)
            torch.cuda.memory._dump_snapshot(dump_file)
            torch.cuda.memory._record_memory_history(enabled=None)
