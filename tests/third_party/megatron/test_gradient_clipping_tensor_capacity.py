"""Large LoRA parameter lists must preserve one global clipping coefficient."""
import math
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(os.environ.get('RUN_CPU_OPTIMIZER_TESTS') != '1', reason='requires Megatron and CUDA')
@pytest.mark.parametrize('dtype,decoupled', [(torch.float32, False), (torch.bfloat16, True)])
def test_many_gradient_tensors_use_one_global_clip_without_handle_exhaustion(dtype, decoupled):
    from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32

    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.08, 0)
    # Each tensor has norm13. This exceeds the installed TE handle capacity,
    # while the actual tensor storage is under one megabyte.
    count = 24000
    original = torch.tensor([3., -4., 0., 12.], device='cuda', dtype=dtype).repeat(count, 1)
    gradients = original.clone()
    parameters = [SimpleNamespace(**{'decoupled_grad' if decoupled else 'grad': gradient})
                  for gradient in gradients.unbind()]
    global_norm = 13. * math.sqrt(count)
    coefficient = 1. / (global_norm + 1.e-6)
    expected = original * coefficient
    for _ in range(4):
        gradients.copy_(original)
        clip_grad_by_total_norm_fp32(parameters, 1., global_norm, use_decoupled_grad=decoupled)
        torch.testing.assert_close(gradients, expected, atol=0, rtol=0)
    # No clipping preserves the same state and does not rescale independently.
    clip_grad_by_total_norm_fp32(parameters, global_norm * 2., global_norm, use_decoupled_grad=decoupled)
    torch.testing.assert_close(gradients, expected, atol=0, rtol=0)
