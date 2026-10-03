"""CPU master restoration must preserve values without a second host model."""
from types import SimpleNamespace
import ast
from enum import Enum
from pathlib import Path
from typing import Container

import pytest
import torch

from roll.third_party.megatron import cpu_master_params


def owner(parameters, masters):
    return SimpleNamespace(
        shard_float16_groups=[parameters], shard_fp32_groups=[[]],
        optimizer=SimpleNamespace(
            param_groups=[{"params": parameters}],
            param_to_fp32_param=dict(zip(parameters, masters)),
            gpu_params_map_cpu_copy=dict(zip(parameters, masters)),
        ),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_restores_current_masters_with_bounded_casts_and_no_master_mutation(dtype, monkeypatch):
    # Nonrepresentable FP32 values exercise rounding; stale initial weights
    # must not be used after an optimizer update.
    master = torch.tensor([1.001, -2.019, 0.125, 3.14159, 42.03125])
    saved = master.clone()
    parameter = torch.full((5,), -99, dtype=dtype)
    optimizer = owner([parameter], [master])
    pairs = cpu_master_params.cpu_master_shard_pairs(optimizer)
    copies = []
    original_copy = torch.Tensor.copy_

    def copy_(destination, source, *args, **kwargs):
        copies.append(source.numel() * source.element_size())
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    cpu_master_params.restore_cpu_master_shards(pairs, chunk_bytes=8)
    torch.testing.assert_close(parameter, saved.to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(master, saved, atol=0, rtol=0)
    assert pairs[0][1] is master
    assert copies and max(copies) <= 8


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "private_copy", "unowned", "fp32_model"])
def test_rejects_non_authoritative_or_incomplete_master_mapping_before_mutation(fault):
    parameters = [torch.full((4,), 7, dtype=torch.bfloat16) for _ in range(2)]
    masters = [torch.arange(4, dtype=torch.float32) for _ in range(2)]
    optimizer = owner(parameters, masters)
    hybrid = optimizer.optimizer
    if fault == "missing":
        del hybrid.param_to_fp32_param[parameters[1]]
    elif fault == "shape":
        hybrid.param_to_fp32_param[parameters[1]] = torch.zeros(3)
    elif fault == "dtype":
        hybrid.param_to_fp32_param[parameters[1]] = masters[1].bfloat16()
    elif fault == "private_copy":
        hybrid.gpu_params_map_cpu_copy[parameters[1]] = masters[1].clone()
    elif fault == "unowned":
        hybrid.param_groups[0]["params"] = parameters[:1]
    else:
        optimizer.shard_fp32_groups = [[torch.zeros(4)]]
    with pytest.raises(ValueError):
        cpu_master_params.cpu_master_shard_pairs(optimizer)
    for parameter in parameters:
        assert torch.equal(parameter, torch.full_like(parameter, 7))


def test_restore_rejects_nonpositive_chunk_bound():
    with pytest.raises(ValueError, match="chunk"):
        cpu_master_params.restore_cpu_master_shards([], chunk_bytes=0)


def test_placeholder_preserves_logical_shape_without_allocating_a_model_copy():
    placeholder = cpu_master_params.empty_cpu_parameter_buffer(1_000_000, torch.bfloat16)
    assert placeholder.shape == (1_000_000,)
    assert placeholder.device.type == "cpu"
    assert placeholder.untyped_storage().nbytes() <= 2
    assert placeholder[100:300].view(10, 20).shape == (10, 20)


def _chain_functions():
    # Load the real orchestration unchanged without importing CUDA Megatron.
    path = Path(__file__).parents[3] / "roll/third_party/megatron/offload_states_patch.py"
    names = {"MegatronOffloadStateType", "chained_optimizers_offload_states",
             "_uses_cpu_master_model_offload", "_cpu_master_bucket_groups", "needs_offload"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = dict(Enum=Enum, Container=Container, ChainedOptimizer=object,
                     cpu_master_shard_pairs=cpu_master_params.cpu_master_shard_pairs)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("fault", ["pending", "missing", "mixed"])
@pytest.mark.parametrize("include", [None, ["model_params"], ["other_params"], ["optimizer_states"]])
def test_chain_rejects_invalid_later_collective_before_any_model_mutation(fault, include):
    namespace = _chain_functions()
    leaves, parameters, calls = [], [], []
    for index in range(2):
        parameter = torch.full((4,), 7, dtype=torch.bfloat16)
        leaf = owner([parameter], [torch.arange(4, dtype=torch.float32)])
        leaf.config = SimpleNamespace(offload_model_from_cpu_master=True)
        leaf.offloaded_states = set()
        bucket = SimpleNamespace()
        group = SimpleNamespace(buckets=[bucket], param_gather_handle=None)
        leaf.buffers = [SimpleNamespace(buckets=[bucket])]
        leaf.model_chunks = [SimpleNamespace(bucket_groups=[group], expert_parallel_bucket_groups=[])]
        if index == 1:
            if fault == "pending":
                group.param_gather_handle = object()
            elif fault == "missing":
                leaf.model_chunks[0].bucket_groups = []
            else:
                group.buckets.append(SimpleNamespace())

        def offload(*, include, _leaf=leaf, _parameter=parameter, **kwargs):
            if include is None or "model_params" in include:
                namespace["_cpu_master_bucket_groups"](_leaf)
                _parameter.zero_()
            calls.append(_leaf)

        leaf.offload_states = offload
        leaves.append(leaf)
        parameters.append(parameter)
    chain = SimpleNamespace(chained_optimizers=leaves, _offload_backend=None, _offload_key_prefix="chain")
    if include is None or "model_params" in include:
        with pytest.raises(ValueError, match="CPU master"):
            namespace["chained_optimizers_offload_states"](chain, include=include)
        assert not calls
    else:
        namespace["chained_optimizers_offload_states"](chain, include=include)
        assert len(calls) == 2
    assert all(torch.equal(p, torch.full_like(p, 7)) for p in parameters)
