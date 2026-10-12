"""Actual eight-rank TP4/EP8/ETP1 loss and gradient coverage.

RUN_QWEN4_TP4_EP8_TESTS=1 torchrun --nproc-per-node=8 -m pytest -q -s this_file.py
"""
import gc
import os

import pytest
import torch
import torch.distributed as dist

from test_qwen4_exp_chunked_model import (
    model_config,
    test_full_model_default_bounds_saved_logits_and_matches_ddp_gradients as _check_ddp,
    test_roll_patched_model_bounds_vocabulary_projection as _check_roll,
)
from test_qwen4_exp_model import make_model


@pytest.fixture(scope="module")
def distributed():
    if os.environ.get("RUN_QWEN4_TP4_EP8_TESTS") != "1":
        pytest.skip("requires eight real Megatron/TE CUDA ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    yield
    dist.destroy_process_group()


@pytest.fixture
def parallel(distributed):
    from megatron.core import parallel_state, tensor_parallel
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=4,
        expert_model_parallel_size=8, expert_tensor_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(351)
    torch.manual_seed(351)
    yield 4
    parallel_state.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("query_groups", [1, 2, 4])
def test_qsa_gate_matches_global_projection_and_gradients(parallel, query_groups):
    """Each gate head follows its global query head, including replicated KV."""
    from megatron.core import parallel_state

    cfg = model_config(parallel)
    cfg.sequence_parallel = False
    cfg.num_attention_heads = 8
    cfg.num_query_groups = query_groups
    attention = make_model(cfg).decoder.layers[3].self_attention
    hidden = torch.randn(4, 2, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = attention.linear_qkv.weight
    group = parallel_state.get_tensor_model_parallel_group()
    rank = dist.get_rank(group)
    shards = [torch.empty_like(weight) for _ in range(parallel)]
    dist.all_gather(shards, weight.detach(), group=group)
    full_weight = torch.cat(shards, dim=0).requires_grad_()
    projected = torch.nn.functional.linear(hidden, full_weight)
    per_group = cfg.num_attention_heads // query_groups
    grouped = projected.reshape(4, 2, query_groups, 2 * per_group + 2, cfg.kv_channels)
    global_gate = grouped[..., per_group:2 * per_group, :].reshape(4, 2, 8, cfg.kv_channels)
    expected = global_gate.chunk(parallel, dim=2)[rank]
    query, _, _, gate = attention.get_query_key_value_tensors(hidden, output_gate=True)
    assert gate.shape == query.shape == expected.shape
    torch.testing.assert_close(gate, expected, atol=0, rtol=0)
    # Sum all rank contributions: the QKV projection may assemble replicated
    # groups, but every global gate head must contribute exactly once.
    gate.float().square().sum().backward()
    expected.float().square().sum().backward()
    dist.all_reduce(full_weight.grad, group=group)
    torch.testing.assert_close(weight.grad, full_weight.grad.chunk(parallel, dim=0)[rank],
                               atol=0, rtol=0)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_roll_patched_tp4_ep8_bounds_forward(parallel, tied, training, monkeypatch):
    _check_roll(parallel, tied, training, monkeypatch)


@pytest.mark.parametrize("tied", [False, True])
def test_tp4_ep8_matches_full_gradient_reference(parallel, tied):
    _check_ddp(parallel, tied)
