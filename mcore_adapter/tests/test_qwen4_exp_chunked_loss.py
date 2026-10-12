"""Real CUDA TP/SP/DDP checks for bounded vocabulary projection and loss.

RUN_QWEN4_CHUNKED_LOSS_TESTS=1 torchrun --master_addr=127.0.0.1
    --master_port=29765 --nproc-per-node=2 -m pytest -q -s this_file.py
"""
import gc
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def distributed():
    if os.environ.get("RUN_QWEN4_CHUNKED_LOSS_TESTS") != "1":
        pytest.skip("requires real Megatron CUDA ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


def implementation():
    path = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp/chunked_loss.py"
    assert path.exists(), "bounded TP vocabulary loss has not been implemented"
    spec = importlib.util.spec_from_file_location("qwen4_chunked_loss", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.chunked_vocab_parallel_cross_entropy


def config(tp, sp, dtype):
    from megatron.core.transformer.transformer_config import TransformerConfig
    return TransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4,
        tensor_model_parallel_size=tp, sequence_parallel=sp,
        params_dtype=dtype, bf16=dtype == torch.bfloat16,
        gradient_accumulation_fusion=False, use_cpu_initialization=False,
    )


def make_head(cfg, vocab):
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    return ColumnParallelLinear(cfg.hidden_size, vocab, config=cfg,
                                init_method=cfg.init_method, bias=False, gather_output=False)


def stock(head, hidden, labels):
    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
    logits, _ = head(hidden)
    return vocab_parallel_cross_entropy(logits.float(), labels)


@pytest.mark.parametrize("tp,sp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ddp", [False, True])
def test_token_losses_and_hidden_head_gradients_match_stock(distributed, tp, sp, dtype, ddp):
    """Missing TP reduction, token weights, or DDP head grads must fail this check."""
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    chunked = implementation()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    tensor_parallel.model_parallel_cuda_manual_seed(2101)
    torch.manual_seed(2101)
    cfg = config(tp, sp, dtype)
    reference_head = make_head(cfg, 256).cuda()
    actual_head = make_head(cfg, 256).cuda()
    actual_head.load_state_dict(reference_head.state_dict())
    full_hidden = torch.randn(30, 2, 64, device="cuda", dtype=dtype)
    labels = torch.arange(60, device="cuda").reshape(30, 2) * 17 % 256
    # A nonuniform, signed upstream gradient exercises PPO-like per-token weighting.
    upstream = torch.linspace(-0.3, 1.1, 60, device="cuda").reshape(30, 2) / 60
    if sp:
        full_hidden = full_hidden.chunk(tp)[parallel_state.get_tensor_model_parallel_rank()]
    expected_hidden = full_hidden.clone().detach().requires_grad_()
    actual_hidden = full_hidden.clone().detach().requires_grad_()

    class HeadLoss(torch.nn.Module):
        def __init__(self, head, bounded):
            super().__init__()
            self.head, self.bounded = head, bounded

        def forward(self, hidden):
            if self.bounded:
                return chunked(hidden, self.head.weight, labels, chunk_size=7,
                               tp_group=self.head.tp_group, sequence_parallel=sp)
            return stock(self.head, hidden, labels)

    ref, actual = HeadLoss(reference_head, False), HeadLoss(actual_head, True)
    if ddp:
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=dtype == torch.float32, overlap_grad_reduce=True,
            use_distributed_optimizer=False,
        )
        ref = DistributedDataParallel(cfg, ddp_config, ref)
        actual = DistributedDataParallel(cfg, ddp_config, actual)
    expected = ref(expected_hidden)
    observed = actual(actual_hidden)
    (expected * upstream).sum().backward()
    (observed * upstream).sum().backward()
    if ddp:
        finalize_model_grads([ref])
        finalize_model_grads([actual])
    expected_grad = reference_head.weight.main_grad if ddp else reference_head.weight.grad
    actual_grad = actual_head.weight.main_grad if ddp else actual_head.weight.grad
    assert actual_grad is not None and actual_grad.float().norm() > 0
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(observed, expected, atol=tolerance, rtol=tolerance)
    for name, observed_grad, reference_grad in (
        ("hidden", actual_hidden.grad, expected_hidden.grad),
        ("head", actual_grad, expected_grad),
    ):
        relative = (observed_grad-reference_grad).float().norm()/reference_grad.float().norm()
        assert relative < tolerance, (name, float(relative))
        print(json.dumps(dict(test="parity", tp=tp, sp=sp, dtype=str(dtype), ddp=ddp,
                              gradient=name, relative_l2=float(relative))), flush=True)
    parallel_state.destroy_model_parallel()


def test_ignored_tokens_and_frozen_head(distributed):
    """Ignored labels must have zero loss/gradient, while a frozen head propagates hidden grads."""
    from megatron.core import parallel_state
    chunked = implementation()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    hidden = torch.randn(11, 64, device="cuda", requires_grad=True)
    weight = torch.randn(256, 64, device="cuda") * 0.1
    labels = torch.arange(11, device="cuda")
    labels[::3] = -100
    expected_hidden = hidden.detach().clone().requires_grad_()
    expected = torch.nn.functional.cross_entropy(expected_hidden @ weight.T, labels, reduction="none")
    actual = chunked(hidden, weight, labels, chunk_size=3)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad, atol=1e-6, rtol=1e-5)
    assert torch.equal(actual[::3], torch.zeros_like(actual[::3]))
    parallel_state.destroy_model_parallel()


def test_shared_head_gradient_includes_other_consumers(distributed):
    """Direct main_grad writes or missing return gradients would lose this second use."""
    from megatron.core import parallel_state
    chunked = implementation()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    hidden = torch.randn(11, 64, device="cuda", requires_grad=True)
    weight = torch.randn(256, 64, device="cuda", requires_grad=True)
    expected_weight = weight.detach().clone().requires_grad_()
    labels = torch.arange(11, device="cuda")
    expected = torch.nn.functional.cross_entropy(hidden.detach() @ expected_weight.T, labels)
    actual = chunked(hidden, weight, labels, chunk_size=3).mean()
    (expected + expected_weight[:11].square().mean()).backward()
    (actual + weight[:11].square().mean()).backward()
    torch.testing.assert_close(weight.grad, expected_weight.grad, atol=1e-6, rtol=1e-5)
    parallel_state.destroy_model_parallel()


def test_chunking_bounds_saved_logits_and_peak_memory(distributed):
    """Retaining full-sequence logits in forward or backward must breach this budget."""
    from megatron.core import parallel_state, tensor_parallel
    chunked = implementation()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    tensor_parallel.model_parallel_cuda_manual_seed(2102)
    cfg = config(2, True, torch.bfloat16)
    cfg.hidden_size = 512
    head = make_head(cfg, 65536).cuda()
    labels = torch.arange(2048, device="cuda").reshape(2048, 1) % 65536
    peaks = {}
    for mode in ("stock", "chunked"):
        head.weight.grad = None
        hidden = torch.randn(1024, 1, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        if mode == "stock":
            loss = stock(head, hidden, labels)
        else:
            loss = chunked(hidden, head.weight, labels, chunk_size=64,
                           tp_group=head.tp_group, sequence_parallel=True)
        loss.mean().backward()
        torch.cuda.synchronize()
        peaks[mode] = torch.cuda.max_memory_allocated() - baseline
        del hidden, loss
    print(json.dumps(dict(test="peak_memory", rank=dist.get_rank(), bytes=peaks)), flush=True)
    assert peaks["chunked"] < peaks["stock"] * 0.35, peaks
    parallel_state.destroy_model_parallel()
