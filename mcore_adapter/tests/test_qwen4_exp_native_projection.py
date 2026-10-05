"""Native GDN projection values, adapter semantics, and distributed gradients.

CUDA: NVIDIA_TF32_OVERRIDE=0 RUN_QWEN4_NATIVE_PROJECTION_TESTS=1 torchrun
--nproc_per_node=2 -m pytest -q this_file.py. The override keeps the independent
FP32 oracle and TE's adapter GEMMs in the same precision without relaxing checks.
"""
import importlib.util
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F


def implementation():
    path = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp/gated_delta_net.py"
    spec = importlib.util.spec_from_file_location("native_projection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    # Only the unused CUDA superclass is substituted for CPU helper coverage.
    # The CUDA tests below import and use the real Megatron/TE implementations.
    if os.environ.get("RUN_QWEN4_NATIVE_PROJECTION_TESTS") == "1":
        spec.loader.exec_module(module)
    else:
        stub = types.ModuleType("megatron.core.ssm.gated_delta_net")
        stub.GatedDeltaNet = torch.nn.Module
        with patch.dict(sys.modules, {stub.__name__: stub}):
            spec.loader.exec_module(module)
    return module.Qwen4ExpGatedDeltaNet._split_input_projection


class TinyAdapter(torch.nn.Module):
    """Small differentiable adapter with the same public layer attributes."""

    def __init__(self, base, dtype):
        super().__init__()
        self.base_layer = base
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(5, 3, bias=False, dtype=dtype)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 9, bias=False, dtype=dtype)})
        self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
        self.scaling = {"default": 0.25}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False

    def get_base_layer(self):
        return self.base_layer

    def merge(self):
        with torch.no_grad():
            self.base_layer.weight.add_(self.scaling["default"] *
                                       (self.lora_B["default"].weight @ self.lora_A["default"].weight))
        self.merged = True

    def unmerge(self):
        with torch.no_grad():
            self.base_layer.weight.sub_(self.scaling["default"] *
                                       (self.lora_B["default"].weight @ self.lora_A["default"].weight))
        self.merged = False


@pytest.mark.parametrize("bias", ["none", "empty", "real"])
def test_strided_input_values_and_gradients(bias):
    torch.manual_seed(19)
    base = torch.nn.Linear(5, 9, bias=bias == "real", dtype=torch.float64)
    if bias == "empty":
        base.bias = torch.nn.Parameter(torch.empty(0, dtype=torch.float64))
    hidden = torch.randn(4, 3, 7, dtype=torch.float64)[..., :5].requires_grad_()
    actual, returned_bias = implementation()(base, hidden, 7)
    expected = F.linear(hidden, base.weight, base.bias if bias == "real" else None)
    torch.testing.assert_close(actual, expected)
    assert returned_bias is None
    inputs = (hidden, base.weight) + ((base.bias,) if bias == "real" else ())
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, inputs, upstream)
    expected_grad = torch.autograd.grad(expected, inputs, upstream)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want)


def test_adapter_uses_its_own_dtype_and_preserves_gradients():
    torch.manual_seed(20)
    base = torch.nn.Linear(5, 9, bias=False, dtype=torch.float64)
    adapter = TinyAdapter(base, torch.float32)
    hidden = torch.randn(4, 2, 5, dtype=torch.float64, requires_grad=True)
    actual, _ = implementation()(adapter, hidden, 7)
    delta = adapter.lora_B["default"](adapter.lora_A["default"](hidden.float())) * 0.25
    expected = F.linear(hidden, base.weight) + delta.to(hidden.dtype)
    torch.testing.assert_close(actual, expected)
    inputs = (hidden, *adapter.parameters())
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, inputs, upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, inputs, upstream)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("adapter_count", [1, 2])
def test_fp32_adapters_round_only_after_adding_to_bf16_base(adapter_count):
    torch.manual_seed(20)
    base = torch.nn.Linear(5, 9, bias=False, dtype=torch.bfloat16)
    adapter = TinyAdapter(base, torch.float32)
    if adapter_count == 2:
        adapter.lora_A["second"] = torch.nn.Linear(5, 3, bias=False)
        adapter.lora_B["second"] = torch.nn.Linear(3, 9, bias=False)
        adapter.lora_dropout["second"] = torch.nn.Identity()
        adapter.scaling["second"] = 0.5
        adapter.active_adapters.append("second")
    hidden = torch.randn(4, 2, 5, dtype=torch.bfloat16, requires_grad=True)
    actual, _ = implementation()(adapter, hidden, 7)
    # Native Qwen uses separate qkvz/ba linears; BF16 input gradients from
    # those two GEMMs need not equal a fused projection's rounded gradient.
    # Apply the shared LoRA wrapper's promoted additions to that native base.
    expected = torch.cat([F.linear(hidden, base.weight[:7]),
                          F.linear(hidden, base.weight[7:])], dim=-1)
    for name in adapter.active_adapters:
        expected = expected + adapter.lora_B[name](adapter.lora_A[name](hidden.float())) * adapter.scaling[name]
    expected = expected.to(hidden.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (hidden, *adapter.parameters())
    actual_grad = torch.autograd.grad(actual.float().square().sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.float().square().sum(), inputs)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_disabling_a_merged_adapter_restores_base_projection():
    torch.manual_seed(21)
    base = torch.nn.Linear(5, 9, bias=False, dtype=torch.float64)
    adapter = TinyAdapter(base, torch.float64)
    hidden = torch.randn(3, 2, 5, dtype=torch.float64)
    expected = F.linear(hidden, base.weight).detach().clone()
    adapter.merge()
    adapter.disable_adapters = True
    actual, _ = implementation()(adapter, hidden, 7)
    torch.testing.assert_close(actual, expected)
    assert not adapter.merged


@pytest.fixture(scope="module")
def distributed():
    if os.environ.get("RUN_QWEN4_NATIVE_PROJECTION_TESTS") != "1":
        pytest.skip("requires two real CUDA/Megatron ranks")
    import torch.distributed as dist
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


@pytest.mark.parametrize("tp,sp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("native_split", [True, False])
def test_te_projection_matches_dense_gradient_across_two_updates(distributed, tp, sp, ddp, lora, native_split):
    """Detect absent TP dgrad, absent SP gather/RS, and lost DDP main_grad."""
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    project = implementation()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    tensor_parallel.model_parallel_cuda_manual_seed(331)
    torch.manual_seed(331)
    cfg = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4,
                            tensor_model_parallel_size=tp, sequence_parallel=sp,
                            params_dtype=torch.float32, gradient_accumulation_fusion=ddp)
    base = TEColumnParallelLinear(64, 160, config=cfg, init_method=cfg.init_method,
                                 bias=False, skip_bias_add=False, gather_output=False, is_expert=False)
    torch.manual_seed(332)
    rank = parallel_state.get_tensor_model_parallel_rank()
    full_weight = torch.randn(160, 64, device="cuda") / 8
    local_rows = slice(rank * (160 // tp), (rank + 1) * (160 // tp))
    with torch.no_grad():
        base.weight.copy_(full_weight[local_rows])
    layer = base
    full_a = full_b = None
    if lora:
        from mcore_adapter.adapters.lora_layer import LoraColumnParallelLinear

        layer = LoraColumnParallelLinear(base, "default", r=8, lora_alpha=8)
        torch.manual_seed(333)
        full_a = torch.randn(8, 64, device="cuda") / 8
        full_b = torch.randn(160, 8, device="cuda") / 8
        with torch.no_grad():
            layer.lora_A["default"].weight.copy_(full_a)
            layer.lora_B["default"].weight.copy_(full_b[local_rows])

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = layer

        def forward(self, hidden):
            return (project(self.linear, hidden, 128 // tp) if native_split else self.linear(hidden))[0]

    model = Projection()
    if ddp:
        ddp_cfg = DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=True)
        model = DistributedDataParallel(cfg, ddp_cfg, model)
    try:
        for _ in range(2):
            if ddp:
                model.zero_grad_buffer()
            else:
                model.zero_grad(set_to_none=True)
            full_hidden = torch.randn(12, 2, 64, device="cuda", requires_grad=True)
            reference_weight = full_weight.clone().requires_grad_()
            hidden = (full_hidden.chunk(tp, dim=0)[rank] if sp else full_hidden).detach().clone()
            hidden.requires_grad_()
            upstream = torch.randn(12, 2, 160, device="cuda")
            expected = F.linear(full_hidden, reference_weight)
            if lora:
                reference_a = full_a.clone().requires_grad_()
                reference_b = full_b.clone().requires_grad_()
                expected = expected + F.linear(F.linear(full_hidden, reference_a), reference_b)
            actual = model(hidden)
            torch.testing.assert_close(actual, expected[..., local_rows], atol=1e-5, rtol=1e-5)
            (actual * upstream[..., local_rows]).sum().backward()
            (expected * upstream).sum().backward()
            if ddp:
                finalize_model_grads([model])
            expected_hidden_grad = full_hidden.grad.chunk(tp)[rank] if sp else full_hidden.grad
            torch.testing.assert_close(hidden.grad, expected_hidden_grad, atol=3e-5, rtol=3e-5)
            actual_grad = base.weight.main_grad if ddp else base.weight.grad
            torch.testing.assert_close(actual_grad, reference_weight.grad[local_rows], atol=3e-5, rtol=3e-5)
            if lora:
                a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
                a_grad, b_grad = (a.main_grad, b.main_grad) if ddp else (a.grad, b.grad)
                torch.testing.assert_close(a_grad, reference_a.grad, atol=3e-5, rtol=3e-5)
                torch.testing.assert_close(b_grad, reference_b.grad[local_rows], atol=3e-5, rtol=3e-5)
            with torch.no_grad():
                full_weight.sub_(0.001 * reference_weight.grad)
                base.weight.sub_(0.001 * actual_grad)
                if lora:
                    full_a.sub_(0.001 * reference_a.grad)
                    full_b.sub_(0.001 * reference_b.grad)
                    a.sub_(0.001 * a_grad)
                    b.sub_(0.001 * b_grad)
    finally:
        parallel_state.destroy_model_parallel()
