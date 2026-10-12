"""Bounded RL logprob/entropy projection: real TP/SP/DDP gradients and memory."""
import gc
import importlib.util
from pathlib import Path

import pytest
import torch

from test_qwen4_exp_chunked_loss import distributed, config, make_head


def implementation():
    path = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp/chunked_loss.py"
    spec = importlib.util.spec_from_file_location("qwen4_bounded_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "chunked_vocab_parallel_logprobs_and_entropy"), "bounded RL statistics missing"
    return module.chunked_vocab_parallel_logprobs_and_entropy


def reference_statistics(head, hidden, labels):
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    logits, _ = head(hidden)
    logits = gather_from_tensor_model_parallel_region(logits, group=head.tp_group).float()
    log_probs = logits.log_softmax(-1)
    chosen = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return torch.stack((chosen, entropy), dim=-1)


@pytest.mark.parametrize("tp,sp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ddp", [False, True])
def test_rl_statistics_match_full_vocab_values_and_signed_gradients(distributed, tp, sp, dtype, ddp):
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    bounded = implementation()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    tensor_parallel.model_parallel_cuda_manual_seed(7213)
    torch.manual_seed(7213)
    cfg = config(tp, sp, dtype)
    expected_head, actual_head = make_head(cfg, 256).cuda(), make_head(cfg, 256).cuda()
    actual_head.load_state_dict(expected_head.state_dict())
    hidden = torch.randn(30, 2, 64, device="cuda", dtype=dtype)
    if sp:
        hidden = hidden.chunk(tp)[parallel_state.get_tensor_model_parallel_rank()]
    expected_hidden = hidden.clone().requires_grad_()
    actual_hidden = hidden.clone().requires_grad_()
    labels = torch.arange(60, device="cuda").reshape(30, 2) * 17 % 256
    upstream = torch.linspace(-0.7, 1.3, 120, device="cuda").reshape(30, 2, 2) / 60

    class Projection(torch.nn.Module):
        def __init__(self, head, use_bounded):
            super().__init__()
            self.head, self.use_bounded = head, use_bounded

        def forward(self, value):
            if self.use_bounded:
                return bounded(value, self.head.weight, labels, chunk_size=7,
                               tp_group=self.head.tp_group, sequence_parallel=sp)
            return reference_statistics(self.head, value, labels)

    expected_model, actual_model = Projection(expected_head, False), Projection(actual_head, True)
    if ddp:
        cfg_ddp = DistributedDataParallelConfig(grad_reduce_in_fp32=dtype == torch.float32,
                                                overlap_grad_reduce=True, use_distributed_optimizer=False)
        expected_model = DistributedDataParallel(cfg, cfg_ddp, expected_model)
        actual_model = DistributedDataParallel(cfg, cfg_ddp, actual_model)
    expected, actual = expected_model(expected_hidden), actual_model(actual_hidden)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    if ddp:
        finalize_model_grads([expected_model])
        finalize_model_grads([actual_model])
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    reference_grad = expected_head.weight.main_grad if ddp else expected_head.weight.grad
    actual_grad = actual_head.weight.main_grad if ddp else actual_head.weight.grad
    for value, reference in ((actual_grad, reference_grad), (actual_hidden.grad, expected_hidden.grad)):
        assert value is not None and bool(value.float().norm() > 0)
        assert float((value-reference).float().norm()/reference.float().norm()) < tolerance
    parallel_state.destroy_model_parallel()


def test_ignored_tokens_and_frozen_head_have_correct_entropy_gradients(distributed):
    bounded = implementation()
    torch.manual_seed(7214)
    weight = torch.randn(256, 64, device="cuda") * 0.1
    hidden = torch.randn(11, 64, device="cuda", requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    labels = torch.arange(11, device="cuda")
    ignored = labels % 3 == 0
    labels[ignored] = -100
    lp = (reference @ weight.T).float().log_softmax(-1)
    expected = torch.stack((lp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1),
                            -(lp.exp()*lp).sum(-1)), -1).masked_fill(ignored.unsqueeze(-1), 0)
    actual = bounded(hidden, weight, labels, chunk_size=3)
    # Entropy-only backward must work when the selected logprob is unused.
    expected[:, 1].sum().backward()
    actual[:, 1].sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(hidden.grad, reference.grad, atol=1e-6, rtol=1e-5)
    assert bool((actual[ignored] == 0).all())
    assert bool((hidden.grad[ignored] == 0).all())


def test_rl_statistics_do_not_save_or_materialize_full_sequence_logits(distributed):
    from megatron.core import parallel_state, tensor_parallel
    bounded = implementation()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    tensor_parallel.model_parallel_cuda_manual_seed(7215)
    cfg = config(2, True, torch.bfloat16)
    cfg.hidden_size = 512
    head = make_head(cfg, 65536).cuda()
    labels = torch.arange(2048, device="cuda").reshape(2048, 1) % 65536
    peaks = {}
    for mode in ("full", "bounded"):
        head.weight.grad = None
        hidden = torch.randn(1024, 1, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        saved_shapes = []

        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            if mode == "full":
                values = reference_statistics(head, hidden, labels)
            else:
                values = bounded(hidden, head.weight, labels, chunk_size=64,
                                 tp_group=head.tp_group, sequence_parallel=True)
        (values[..., 0].mean() - 0.03 * values[..., 1].mean()).backward()
        torch.cuda.synchronize()
        peaks[mode] = torch.cuda.max_memory_allocated() - base
        if mode == "bounded":
            assert not any(len(shape) >= 2 and shape[-1] in (32768, 65536)
                           and torch.tensor(shape[:-1]).prod().item() == 2048 for shape in saved_shapes)
        del hidden, values
    assert peaks["bounded"] < peaks["full"] * 0.5, peaks
    print(peaks, flush=True)
    parallel_state.destroy_model_parallel()
