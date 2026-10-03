"""Real Megatron/Transformer Engine coverage for grouped expert LoRA.

Run with two ranks so the same test covers TP1/ETP1 and TP2/ETP2::

    RUN_MEGATRON_GROUPED_LORA_TESTS=1 torchrun --nnodes=1 --nproc-per-node=2 \
        --master-addr=127.0.0.1 --master-port=29783 \
        -m pytest -q mcore_adapter/tests/test_megatron_grouped_lora.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


ATOL = 2e-2
RTOL = 2e-2
HIDDEN_SIZE = 16
FFN_SIZE = 32
LORA_RANK = 8
NUM_EXPERTS = 2
TOKENS_PER_EXPERT = (2, 3)


@pytest.fixture(scope="module")
def distributed():
    if os.environ.get("RUN_MEGATRON_GROUPED_LORA_TESTS") != "1":
        pytest.skip("requires two real Megatron/Transformer Engine CUDA ranks")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for Transformer Engine grouped GEMMs")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group("nccl")
    if dist.get_world_size() != 2:
        pytest.skip("run with torchrun --nproc-per-node=2")

    yield

    if owns_process_group:
        dist.destroy_process_group()


def _config(tp_size, expert_tp_size):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        ffn_hidden_size=FFN_SIZE,
        moe_ffn_hidden_size=FFN_SIZE,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        tensor_model_parallel_size=tp_size,
        expert_tensor_parallel_size=expert_tp_size,
        expert_model_parallel_size=1,
        add_bias_linear=False,
        sequence_parallel=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
    )


def _initialize_layout(tp_size, expert_tp_size):
    from megatron.core import parallel_state, tensor_parallel

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        expert_tensor_parallel_size=expert_tp_size,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(1979)
    torch.manual_seed(1979)


def _destroy_layout():
    from megatron.core import parallel_state

    dist.barrier()
    parallel_state.destroy_model_parallel()


def _fill_grouped_adapter(adapter):
    with torch.no_grad():
        for expert_idx in range(NUM_EXPERTS):
            weight_a = getattr(adapter.lora_A["default"], f"weight{expert_idx}")
            weight_b = getattr(adapter.lora_B["default"], f"weight{expert_idx}")
            weight_a.copy_(
                torch.linspace(-0.2, 0.2, weight_a.numel(), device=weight_a.device).reshape_as(weight_a)
                + 0.03 * (expert_idx + 1)
            )
            weight_b.copy_(
                torch.linspace(0.15, -0.1, weight_b.numel(), device=weight_b.device).reshape_as(weight_b)
                + 0.02 * (expert_idx + 1)
            )


def _grouped_delta_reference(adapter, inputs):
    outputs = []
    offset = 0
    for expert_idx, token_count in enumerate(TOKENS_PER_EXPERT):
        expert_inputs = inputs[offset : offset + token_count]
        weight_a = getattr(adapter.lora_A["default"], f"weight{expert_idx}")
        weight_b = getattr(adapter.lora_B["default"], f"weight{expert_idx}")
        outputs.append(F.linear(F.linear(expert_inputs, weight_a), weight_b))
        offset += token_count
    return torch.cat(outputs) * adapter.scaling["default"]


def _make_grouped_adapter(kind, config):
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from mcore_adapter.adapters.lora_layer import LoraColumnParallelLinear, LoraRowParallelLinear
    from mcore_adapter.adapters import set_linear_is_expert

    common = dict(
        num_gemms=NUM_EXPERTS,
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
        is_expert=True,
    )
    if kind == "column":
        base = TEColumnParallelGroupedLinear(
            input_size=HIDDEN_SIZE,
            output_size=FFN_SIZE,
            **common,
        )
        adapter_type = LoraColumnParallelLinear
    else:
        base = TERowParallelGroupedLinear(
            input_size=FFN_SIZE,
            output_size=HIDDEN_SIZE,
            **common,
        )
        adapter_type = LoraRowParallelLinear

    # Match model-provider setup before LoRA injection; MCore's constructor
    # consumes is_expert but does not retain the attribute used by MCA LoRA.
    set_linear_is_expert(base)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    adapter = adapter_type(
        base_layer=base,
        adapter_name="default",
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0.0,
        init_lora_weights=False,
        lora_bias=False,
    )
    _fill_grouped_adapter(adapter)
    return adapter


@pytest.mark.parametrize("tp_size,expert_tp_size", [(1, 1), (2, 2)])
@pytest.mark.parametrize("kind", ["column", "row"])
def test_grouped_expert_lora_forward_backward_merge_and_ownership(
    distributed, tp_size, expert_tp_size, kind
):
    """Catches wrong grouped constructor args, dimensions, ownership, or adapter math."""
    _initialize_layout(tp_size, expert_tp_size)
    try:
        config = _config(tp_size, expert_tp_size)
        adapter = _make_grouped_adapter(kind, config)
        base = adapter.get_base_layer()
        lora_a = adapter.lora_A["default"]
        lora_b = adapter.lora_B["default"]
        effective_rank = LORA_RANK // config.moe_router_topk

        assert base.num_gemms == lora_a.num_gemms == lora_b.num_gemms == NUM_EXPERTS
        assert lora_a._pg_collection is base._pg_collection
        assert lora_b._pg_collection is base._pg_collection
        assert lora_a._tp_group is base._pg_collection.expt_tp
        assert lora_b._tp_group is base._pg_collection.expt_tp
        assert dist.get_world_size(base._pg_collection.expt_tp) == expert_tp_size

        if kind == "column":
            assert base.weight0.shape == (FFN_SIZE // expert_tp_size, HIDDEN_SIZE)
            assert lora_a.weight0.shape == (effective_rank, HIDDEN_SIZE)
            assert lora_b.weight0.shape == (FFN_SIZE // expert_tp_size, effective_rank)
            nonparallel_factor = lora_a
            input_size = HIDDEN_SIZE
        else:
            assert base.weight0.shape == (HIDDEN_SIZE, FFN_SIZE // expert_tp_size)
            assert lora_a.weight0.shape == (effective_rank, FFN_SIZE // expert_tp_size)
            assert lora_b.weight0.shape == (HIDDEN_SIZE, effective_rank)
            nonparallel_factor = lora_b
            input_size = FFN_SIZE // expert_tp_size
        assert nonparallel_factor.parallel_mode is None

        inputs = torch.linspace(
            -0.5,
            0.75,
            sum(TOKENS_PER_EXPERT) * input_size,
            device="cuda",
            dtype=torch.bfloat16,
        ).reshape(sum(TOKENS_PER_EXPERT), input_size)
        splits = list(TOKENS_PER_EXPERT)
        frozen_weights = [
            getattr(base, f"weight{expert_idx}").detach().clone() for expert_idx in range(NUM_EXPERTS)
        ]

        with torch.no_grad():
            base_output, _ = base(inputs, splits)
        output, bias = adapter(inputs, splits)
        actual_delta = output - base_output
        expected_delta = _grouped_delta_reference(adapter, inputs)
        assert bias is None
        assert float(actual_delta.float().norm()) > 0
        torch.testing.assert_close(actual_delta, expected_delta, atol=ATOL, rtol=RTOL)

        output.float().square().mean().backward()
        for factor in (lora_a, lora_b):
            for expert_idx in range(NUM_EXPERTS):
                gradient = getattr(factor, f"weight{expert_idx}").grad
                assert gradient is not None
                assert float(gradient.float().norm()) > 0
        assert all(parameter.grad is None for parameter in base.parameters())

        with torch.no_grad():
            unmerged, _ = adapter(inputs, splits)
            adapter.merge()
            merged, _ = adapter(inputs, splits)
            adapter.unmerge()
            restored, _ = adapter(inputs, splits)
        torch.testing.assert_close(merged, unmerged, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(restored, unmerged, atol=ATOL, rtol=RTOL)
        for expert_idx, expected_weight in enumerate(frozen_weights):
            torch.testing.assert_close(
                getattr(base, f"weight{expert_idx}"), expected_weight, atol=ATOL, rtol=RTOL
            )
    finally:
        _destroy_layout()


@pytest.mark.parametrize("kind", ["column", "row"])
def test_dense_parallel_lora_remains_functional(distributed, kind):
    """Catches regressions from selecting grouped versus dense process-group kwargs."""
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
    from mcore_adapter.adapters.lora_layer import LoraColumnParallelLinear, LoraRowParallelLinear

    _initialize_layout(2, 2)
    try:
        config = _config(2, 2)
        common = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        if kind == "column":
            base = TEColumnParallelLinear(
                input_size=HIDDEN_SIZE,
                output_size=FFN_SIZE,
                gather_output=False,
                **common,
            )
            adapter_type = LoraColumnParallelLinear
            input_size = HIDDEN_SIZE
        else:
            base = TERowParallelLinear(
                input_size=FFN_SIZE,
                output_size=HIDDEN_SIZE,
                input_is_parallel=True,
                **common,
            )
            adapter_type = LoraRowParallelLinear
            input_size = FFN_SIZE // 2

        for parameter in base.parameters():
            parameter.requires_grad_(False)
        adapter = adapter_type(
            base_layer=base,
            adapter_name="default",
            r=LORA_RANK,
            lora_alpha=LORA_RANK,
            lora_dropout=0.0,
            init_lora_weights=False,
            lora_bias=False,
        )
        with torch.no_grad():
            adapter.lora_A["default"].weight.fill_(0.125)
            adapter.lora_B["default"].weight.fill_(0.25)
        inputs = torch.linspace(
            -0.5,
            0.75,
            3 * input_size,
            device="cuda",
            dtype=torch.bfloat16,
        ).reshape(3, input_size)
        with torch.no_grad():
            base_output, _ = base(inputs)
        output, _ = adapter(inputs)
        assert float((output - base_output).float().norm()) > 0
        output.float().square().mean().backward()
        assert float(adapter.lora_A["default"].weight.grad.float().norm()) > 0
        assert float(adapter.lora_B["default"].weight.grad.float().norm()) > 0
        assert all(parameter.grad is None for parameter in base.parameters())
    finally:
        _destroy_layout()
