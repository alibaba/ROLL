"""CUDA regression for repeatability and numerical accuracy of dense LoRA."""

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and native vLLM")
@pytest.mark.parametrize("count,width,rank,slices", [(210, 2560, 64, 1), (149, 10240, 6, 1),
                                                      (384, 2560, 64, 2), (1, 10240, 64, 1)])
def test_dense_lora_is_repeatable_and_matches_fp32_reference(monkeypatch, count, width, rank, slices):
    from roll.third_party.vllm.lora_shrink import patch_qwen38_lora_shrink

    module = importlib.import_module("vllm.lora.ops.triton_ops.lora_shrink_op")
    # Restore process-global native configuration after every case.
    monkeypatch.setattr(module, "get_lora_op_configs", module.get_lora_op_configs)
    patch_qwen38_lora_shrink(SimpleNamespace(model_type="qwen4_exp"), SimpleNamespace())
    torch.manual_seed(42)
    x = torch.randn(count, width, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(1, rank, width, device="cuda", dtype=torch.bfloat16) * .01 for _ in range(slices)]
    output = torch.empty(slices, count, rank, device="cuda", dtype=torch.float32)
    metadata = [torch.zeros(count, device="cuda", dtype=torch.int64),
                torch.arange(count, device="cuda", dtype=torch.int64),
                torch.tensor([count, 0], device="cuda", dtype=torch.int32),
                torch.tensor([0, count, count], device="cuda", dtype=torch.int32),
                torch.tensor([0, -1], device="cuda", dtype=torch.int32),
                torch.tensor([False]), torch.tensor([1])]
    module._lora_shrink(x, weights, output, *metadata, 1.)
    first = output.clone()
    for _ in range(32):
        module._lora_shrink(x, weights, output, *metadata, 1.)
        assert torch.equal(output, first)
    # FP64 provides an independent reference without depending on TF32 settings.
    reference = torch.stack([x.double() @ weight[0].double().T for weight in weights]).float()
    torch.testing.assert_close(output, reference, atol=1e-4, rtol=1e-4)
