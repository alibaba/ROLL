"""GDN row projection must retain partial sums until the TP reduction."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


def implementation():
    path = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp/precision_linear.py"
    spec = importlib.util.spec_from_file_location("precision_linear_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_strided_input_and_double_gradcheck():
    linear = implementation().linear_with_fp32_output
    torch.manual_seed(416)
    x = torch.randn(3, 2, 7, dtype=torch.float64)[..., :5].requires_grad_()
    weight = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(linear(x, weight), F.linear(x, weight))
    assert torch.autograd.gradcheck(linear, (x, weight), fast_mode=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_partial_output_and_backward_match_fp32_reference(device):
    if device == "cuda" and os.environ.get("RUN_QWEN4_OUTPUT_PROJECTION_TESTS") != "1":
        pytest.skip("requires the CUDA validation environment")
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    linear = implementation().linear_with_fp32_output
    torch.manual_seed(417)
    x = torch.randn(9, 2, 64, device=device, dtype=torch.bfloat16).requires_grad_()
    weight = torch.randn(32, 64, device=device, dtype=torch.bfloat16).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference_w = weight.detach().clone().requires_grad_()
    actual = linear(x, weight)
    expected = F.linear(reference_x.float(), reference_w.float())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    # The production projection casts only after the FP32 collective; its
    # upstream gradient is consequently representable in the input dtype.
    upstream = torch.randn_like(actual).bfloat16()
    actual.bfloat16().backward(upstream)
    expected.bfloat16().backward(upstream)
    for got, want in ((x.grad, reference_x.grad), (weight.grad, reference_w.grad)):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.fixture(scope="module")
def distributed():
    if os.environ.get("RUN_QWEN4_OUTPUT_PROJECTION_TESTS") != "1":
        pytest.skip("requires two real Megatron/TE CUDA ranks")
    import torch.distributed as dist
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


@pytest.mark.parametrize("tp,sp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_real_row_projection_and_gradients_over_two_updates(distributed, tp, sp, ddp, lora):
    from types import SimpleNamespace

    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.extensions.transformer_engine import TERowParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig
    from mcore_adapter.models.qwen4_exp.gated_delta_net import Qwen4ExpGatedDeltaNet

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    tensor_parallel.model_parallel_cuda_manual_seed(418)
    torch.manual_seed(418)
    config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4,
                              tensor_model_parallel_size=tp, sequence_parallel=sp,
                              params_dtype=torch.bfloat16, bf16=True,
                              gradient_accumulation_fusion=ddp)
    base = TERowParallelLinear(96, 64, config=config, init_method=config.init_method,
                               bias=False, skip_bias_add=False, input_is_parallel=True,
                               is_expert=False)
    rank = parallel_state.get_tensor_model_parallel_rank()
    columns = slice(rank * (96 // tp), (rank + 1) * (96 // tp))
    weight = (torch.randn(64, 96, device="cuda") / 8).bfloat16()
    with torch.no_grad():
        base.weight.copy_(weight[:, columns])
    layer = base
    if lora:
        from mcore_adapter.adapters.lora_layer import LoraRowParallelLinear
        layer = LoraRowParallelLinear(base, "default", r=8, lora_alpha=8)
        # FP32 adapters keep this check independent of base BF16 roundoff.
        layer.lora_A["default"].float()
        layer.lora_B["default"].float()
        a = torch.randn(8, 96, device="cuda") / 8
        b = torch.randn(64, 8, device="cuda") / 8
        with torch.no_grad():
            layer.lora_A["default"].weight.copy_(a[:, columns])
            layer.lora_B["default"].weight.copy_(b)

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = layer

        def forward(self, x):
            proxy = SimpleNamespace(out_proj=self.linear)
            with Qwen4ExpGatedDeltaNet._fp32_output_projection(proxy):
                return self.linear(x)[0]

    model = Projection()
    if ddp:
        model = DistributedDataParallel(config, DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=True), model)
    try:
        for step in range(2):
            if ddp:
                model.zero_grad_buffer()
            else:
                model.zero_grad(set_to_none=True)
            full = torch.randn(12, 2, 96, device="cuda").bfloat16().requires_grad_()
            x = full[..., columns].detach().clone().requires_grad_()
            rw = weight.clone().requires_grad_()
            expected = F.linear(full.float(), rw.float()).bfloat16()
            if lora:
                ra, rb = a.clone().requires_grad_(), b.clone().requires_grad_()
                expected = (expected + F.linear(F.linear(full.float(), ra), rb)).bfloat16()
            torch.testing.assert_close(base.weight, weight[:, columns], atol=0, rtol=0,
                                       msg=f"weight before update {step}")
            actual = model(x)
            shard = lambda value: value.chunk(tp, dim=0)[rank] if sp else value
            torch.testing.assert_close(actual, shard(expected), atol=2e-3, rtol=2e-3)
            upstream = torch.randn_like(expected)
            actual.backward(shard(upstream))
            expected.backward(upstream)
            if ddp:
                finalize_model_grads([model])
            grad = lambda p: p.main_grad if ddp else p.grad
            pairs = [(x.grad, full.grad[..., columns]), (grad(base.weight), rw.grad[:, columns])]
            if lora:
                pairs += [(grad(layer.lora_A["default"].weight), ra.grad[:, columns]),
                          (grad(layer.lora_B["default"].weight), rb.grad)]
            for got, want in pairs:
                relative = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)
                assert relative < .006
            with torch.no_grad():
                # DDP accumulates BF16 parameter gradients in an FP32 buffer;
                # use that same optimizer arithmetic for the dense reference.
                weight.sub_(.001 * (rw.grad.float() if ddp else rw.grad))
                base.weight.sub_(.001 * grad(base.weight))
                if lora:
                    a.sub_(.001 * ra.grad)
                    b.sub_(.001 * rb.grad)
                    layer.lora_A["default"].weight.sub_(.001 * grad(layer.lora_A["default"].weight))
                    layer.lora_B["default"].weight.sub_(.001 * grad(layer.lora_B["default"].weight))
    finally:
        parallel_state.destroy_model_parallel()
