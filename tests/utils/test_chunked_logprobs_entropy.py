"""Correctness tests for chunked log_probs_from_logits / entropy_from_logits.

Verifies, on CPU with small shapes, that the chunked autograd.Function path
(_ChunkedLogProbsFromLogits / _ChunkedEntropyFromLogits) matches the original
single-pass implementation in BOTH forward values and input gradients, and
passes torch.autograd.gradcheck in float64.

Run directly:  python tests/utils/test_chunked_logprobs_entropy.py
Or via pytest: pytest tests/utils/test_chunked_logprobs_entropy.py -v
"""

import torch
import torch.nn.functional as F

from roll.utils.functionals import (
    _ChunkedEntropyFromLogits,
    _ChunkedLogProbsFromLogits,
    entropy_from_logits,
    log_probs_from_logits,
)

B, T, V = 2, 37, 53  # T deliberately not divisible by chunk_size
CHUNK = 8


def _ref_log_probs(logits, labels):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def _ref_entropy(logits):
    logits = logits.float()
    pd = torch.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


def test_log_probs_forward_and_grad_match():
    torch.manual_seed(0)
    base = torch.randn(B, T, V, dtype=torch.float32)
    labels = torch.randint(0, V, (B, T))
    upstream = torch.randn(B, T, dtype=torch.float32)  # simulate arbitrary dlp

    x_ref = base.clone().requires_grad_(True)
    x_chk = base.clone().requires_grad_(True)

    out_ref = _ref_log_probs(x_ref, labels)
    out_chk = log_probs_from_logits(x_chk, labels, chunk_size=CHUNK)
    assert torch.allclose(out_ref, out_chk, atol=1e-6), "forward values mismatch"

    (out_ref * upstream).sum().backward()
    (out_chk * upstream).sum().backward()
    assert torch.allclose(x_ref.grad, x_chk.grad, atol=1e-6), (
        f"grad mismatch, max abs diff={(x_ref.grad - x_chk.grad).abs().max().item()}"
    )
    print("[log_probs] forward & grad match: OK "
          f"(max grad diff={(x_ref.grad - x_chk.grad).abs().max().item():.3e})")


def test_entropy_forward_and_grad_match():
    torch.manual_seed(1)
    base = torch.randn(B, T, V, dtype=torch.float32)
    upstream = torch.randn(B, T, dtype=torch.float32)

    x_ref = base.clone().requires_grad_(True)
    x_chk = base.clone().requires_grad_(True)

    out_ref = _ref_entropy(x_ref)
    out_chk = entropy_from_logits(x_chk, chunk_size=CHUNK)
    assert torch.allclose(out_ref, out_chk, atol=1e-6), "forward values mismatch"

    (out_ref * upstream).sum().backward()
    (out_chk * upstream).sum().backward()
    assert torch.allclose(x_ref.grad, x_chk.grad, atol=1e-5), (
        f"grad mismatch, max abs diff={(x_ref.grad - x_chk.grad).abs().max().item()}"
    )
    print("[entropy] forward & grad match: OK "
          f"(max grad diff={(x_ref.grad - x_chk.grad).abs().max().item():.3e})")


def test_bf16_input_grad_dtype_and_close():
    """bf16 logits (production dtype): grad dtype must match input, values close."""
    torch.manual_seed(2)
    base = torch.randn(B, T, V, dtype=torch.bfloat16)
    labels = torch.randint(0, V, (B, T))

    x_ref = base.clone().requires_grad_(True)
    x_chk = base.clone().requires_grad_(True)

    _ref_log_probs(x_ref, labels).sum().backward()
    log_probs_from_logits(x_chk, labels, chunk_size=CHUNK).sum().backward()

    assert x_chk.grad.dtype == torch.bfloat16, "grad dtype must match input dtype"
    assert torch.allclose(x_ref.grad.float(), x_chk.grad.float(), atol=1e-3), (
        f"bf16 grad mismatch, max abs diff={(x_ref.grad.float() - x_chk.grad.float()).abs().max().item()}"
    )
    print("[bf16] grad dtype & values: OK")


def test_gradcheck_float64():
    """Strict numerical-vs-analytical gradient check in float64 (tiny shapes)."""
    b, t, v = 1, 9, 7
    chunk = 4
    torch.manual_seed(3)

    x = torch.randn(b, t, v, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, v, (b, t))
    assert torch.autograd.gradcheck(
        lambda inp: _ChunkedLogProbsFromLogits.apply(inp, labels, chunk), (x,), eps=1e-6, atol=1e-4
    ), "gradcheck failed for _ChunkedLogProbsFromLogits"
    print("[gradcheck] _ChunkedLogProbsFromLogits: OK")

    x2 = torch.randn(b, t, v, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda inp: _ChunkedEntropyFromLogits.apply(inp, chunk), (x2,), eps=1e-6, atol=1e-4
    ), "gradcheck failed for _ChunkedEntropyFromLogits"
    print("[gradcheck] _ChunkedEntropyFromLogits: OK")


def test_no_grad_path_unchanged():
    """no_grad path (old/ref log_probs scenario) must equal reference values."""
    torch.manual_seed(4)
    base = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    with torch.no_grad():
        out_chk = log_probs_from_logits(base, labels, chunk_size=CHUNK)
        out_ref = _ref_log_probs(base, labels)
        ent_chk = entropy_from_logits(base, chunk_size=CHUNK)
        ent_ref = _ref_entropy(base)
    assert torch.allclose(out_ref, out_chk, atol=1e-6)
    assert torch.allclose(ent_ref, ent_chk, atol=1e-6)
    print("[no_grad] values match: OK")


def test_input_not_modified_by_backward():
    """Regression: entropy backward used in-place math on a .to() view and
    corrupted same-dtype inputs. Inputs must be bit-identical after backward.
    fp32/fp64 are the trigger dtypes (.to() returns a view); bf16 guards the copy path."""
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        torch.manual_seed(5)
        x_lp = torch.randn(B, T, V, dtype=dtype, requires_grad=True)
        x_ent = torch.randn(B, T, V, dtype=dtype, requires_grad=True)
        labels = torch.randint(0, V, (B, T))
        snap_lp = x_lp.detach().clone()
        snap_ent = x_ent.detach().clone()

        log_probs_from_logits(x_lp, labels, chunk_size=CHUNK).sum().backward()
        entropy_from_logits(x_ent, chunk_size=CHUNK).sum().backward()

        assert torch.equal(x_lp.detach(), snap_lp), f"log_probs backward modified input ({dtype})"
        assert torch.equal(x_ent.detach(), snap_ent), f"entropy backward modified input ({dtype})"
    print("[no-mutation] inputs untouched by backward (fp32/fp64/bf16): OK")


def test_shared_logits_logprobs_plus_entropy():
    """actor_worker production pattern: the SAME logits tensor feeds both
    log_probs and entropy, gradients accumulate through one combined loss."""
    torch.manual_seed(6)
    base = torch.randn(B, T, V, dtype=torch.float32)
    labels = torch.randint(0, V, (B, T))
    u1 = torch.randn(B, T)
    u2 = torch.randn(B, T)

    x_ref = base.clone().requires_grad_(True)
    x_chk = base.clone().requires_grad_(True)

    loss_ref = (_ref_log_probs(x_ref, labels) * u1).sum() + (_ref_entropy(x_ref) * u2).sum()
    loss_chk = (log_probs_from_logits(x_chk, labels, chunk_size=CHUNK) * u1).sum() + (
        entropy_from_logits(x_chk, chunk_size=CHUNK) * u2
    ).sum()
    loss_ref.backward()
    loss_chk.backward()  # would raise a version-counter RuntimeError if either op corrupted x_chk
    assert torch.allclose(x_ref.grad, x_chk.grad, atol=1e-5), (
        f"shared-logits grad mismatch, max abs diff={(x_ref.grad - x_chk.grad).abs().max().item()}"
    )
    print("[shared-logits] combined log_probs+entropy backward: OK")


def test_chunk_disabled_equals_reference():
    """chunk_size<=0 must keep the original single-pass path bit-exact."""
    torch.manual_seed(7)
    base = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    assert torch.equal(log_probs_from_logits(base, labels, chunk_size=0), _ref_log_probs(base, labels))
    assert torch.equal(entropy_from_logits(base, chunk_size=0), _ref_entropy(base))
    print("[fallback] chunk_size=0 path bit-exact vs reference: OK")


def test_boundary_chunk_sizes():
    """Tail-width boundaries (both ops) plus an exactly-divisible case.

    With T=37: chunk=9 leaves a 1-wide tail (37=4*9+1) and chunk=36 also leaves
    a 1-wide tail (37=36+1, i.e. T=chunk+1); the exactly-divisible case is the
    separate t_div=32/chunk=8 block below.
    """
    torch.manual_seed(8)
    labels_full = torch.randint(0, V, (B, T))
    for chunk in (T // 4, T - 1):
        x = torch.randn(B, T, V, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        out = log_probs_from_logits(x, labels_full, chunk_size=chunk)
        ref = _ref_log_probs(x2, labels_full)
        assert torch.allclose(out, ref, atol=1e-6), f"log_probs forward mismatch (chunk={chunk})"
        out.sum().backward()
        ref.sum().backward()
        assert torch.allclose(x.grad, x2.grad, atol=1e-6), f"log_probs grad mismatch (chunk={chunk})"

        e = torch.randn(B, T, V, requires_grad=True)
        e2 = e.detach().clone().requires_grad_(True)
        out_e = entropy_from_logits(e, chunk_size=chunk)
        ref_e = _ref_entropy(e2)
        assert torch.allclose(out_e, ref_e, atol=1e-6), f"entropy forward mismatch (chunk={chunk})"
        out_e.sum().backward()
        ref_e.sum().backward()
        assert torch.allclose(e.grad, e2.grad, atol=1e-5), f"entropy grad mismatch (chunk={chunk})"
    # exactly divisible case
    t_div = 32
    labels_div = torch.randint(0, V, (B, t_div))
    x = torch.randn(B, t_div, V, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    out = log_probs_from_logits(x, labels_div, chunk_size=8)
    ref = _ref_log_probs(x2, labels_div)
    assert torch.allclose(out, ref, atol=1e-6), "log_probs forward mismatch (divisible)"
    out.sum().backward()
    ref.sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-6), "log_probs grad mismatch (divisible)"

    e = torch.randn(B, t_div, V, requires_grad=True)
    e2 = e.detach().clone().requires_grad_(True)
    out_e = entropy_from_logits(e, chunk_size=8)
    ref_e = _ref_entropy(e2)
    assert torch.allclose(out_e, ref_e, atol=1e-6), "entropy forward mismatch (divisible)"
    out_e.sum().backward()
    ref_e.sum().backward()
    assert torch.allclose(e.grad, e2.grad, atol=1e-5), "entropy grad mismatch (divisible)"
    print("[boundaries] 1-wide tails (x2) / divisible, both ops: OK")


def test_double_backward_raises():
    """once_differentiable must make second-order gradients fail loudly (both ops)."""
    labels = torch.randint(0, V, (B, T))

    x = torch.randn(B, T, V, requires_grad=True)
    out = log_probs_from_logits(x, labels, chunk_size=CHUNK).sum()
    (gx,) = torch.autograd.grad(out, x, create_graph=True)
    try:
        torch.autograd.grad(gx.sum(), x)
        raise AssertionError("log_probs double backward should have raised RuntimeError")
    except RuntimeError:
        pass

    e = torch.randn(B, T, V, requires_grad=True)
    out_e = entropy_from_logits(e, chunk_size=CHUNK).sum()
    (ge,) = torch.autograd.grad(out_e, e, create_graph=True)
    try:
        torch.autograd.grad(ge.sum(), e)
        raise AssertionError("entropy double backward should have raised RuntimeError")
    except RuntimeError:
        pass
    print("[once_differentiable] double backward raises (both ops): OK")


def test_entropy_bf16_grad_dtype():
    """entropy bf16 production path: grad dtype must match input, values close."""
    torch.manual_seed(9)
    base = torch.randn(B, T, V, dtype=torch.bfloat16)
    x_ref = base.clone().requires_grad_(True)
    x_chk = base.clone().requires_grad_(True)
    _ref_entropy(x_ref).sum().backward()
    entropy_from_logits(x_chk, chunk_size=CHUNK).sum().backward()
    assert x_chk.grad.dtype == torch.bfloat16, "entropy grad dtype must match input dtype"
    assert torch.allclose(x_ref.grad.float(), x_chk.grad.float(), atol=1e-2), (
        f"entropy bf16 grad mismatch, max abs diff={(x_ref.grad.float() - x_chk.grad.float()).abs().max().item()}"
    )
    print("[entropy-bf16] grad dtype & values: OK")


def test_chunk_size_one():
    """chunk_size=1 (degenerate per-token chunking) must still match reference."""
    torch.manual_seed(10)
    labels = torch.randint(0, V, (B, T))
    x = torch.randn(B, T, V, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    out = log_probs_from_logits(x, labels, chunk_size=1)
    ref = _ref_log_probs(x2, labels)
    assert torch.allclose(out, ref, atol=1e-6), "log_probs forward mismatch (chunk=1)"
    out.sum().backward(); ref.sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-6), "log_probs grad mismatch (chunk=1)"

    e = torch.randn(B, T, V, requires_grad=True)
    e2 = e.detach().clone().requires_grad_(True)
    out_e = entropy_from_logits(e, chunk_size=1)
    ref_e = _ref_entropy(e2)
    assert torch.allclose(out_e, ref_e, atol=1e-6), "entropy forward mismatch (chunk=1)"
    out_e.sum().backward(); ref_e.sum().backward()
    assert torch.allclose(e.grad, e2.grad, atol=1e-5), "entropy grad mismatch (chunk=1)"
    print("[chunk=1] per-token chunking matches reference (both ops): OK")


def test_non_3d_input_falls_back():
    """2D logits must skip chunking (guard requires dim==3) and match reference."""
    torch.manual_seed(11)
    labels = torch.randint(0, V, (T,))
    x = torch.randn(T, V, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    out = log_probs_from_logits(x, labels, chunk_size=CHUNK)  # 2D -> single-pass fallback
    ref = F.log_softmax(x2.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out, ref, atol=1e-6), "2D log_probs forward mismatch"
    out.sum().backward(); ref.sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-6), "2D log_probs grad mismatch"
    print("[dim!=3] 2D input falls back to single-pass: OK")


def test_grad_output_not_modified():
    """Regression: `g` aliases grad_output when dtypes match; backward must treat it
    read-only. Guards against a future 'optimization' like mul_(-g) -> g.neg_() that
    would corrupt the upstream node's gradient buffer. (Ordinary .sum().backward()
    can't catch this because grad_output is then a throwaway ones tensor.)"""
    labels = torch.randint(0, V, (B, T))
    for name, out in (
        ("log_probs", lambda x: log_probs_from_logits(x, labels, chunk_size=CHUNK)),
        ("entropy", lambda x: entropy_from_logits(x, chunk_size=CHUNK)),
    ):
        x = torch.randn(B, T, V, requires_grad=True)
        go = torch.randn(B, T)  # explicit, non-ones upstream grad
        snap = go.clone()
        torch.autograd.grad(out(x), x, grad_outputs=go)
        assert torch.equal(go, snap), f"{name} backward mutated grad_output (aliasing bug)"
    print("[grad_output-readonly] upstream grad untouched (both ops): OK")


def test_o_of_chunk_peak_memory_cuda():
    """CUDA-only: the O(chunk) peak-memory claim -- chunked backward must allocate
    dramatically less than the single-pass reference on a long sequence. Skipped on CPU."""
    if not torch.cuda.is_available():
        print("[peak-mem] skipped (no CUDA)")
        return
    dev = "cuda"
    b, t, v, chunk = 1, 8192, 4096, 512
    labels = torch.randint(0, v, (b, t), device=dev)

    x = torch.randn(b, t, v, device=dev, requires_grad=True)
    torch.cuda.reset_peak_memory_stats(dev)
    _ref_log_probs(x, labels).sum().backward()
    ref_peak = torch.cuda.max_memory_allocated(dev)

    x2 = torch.randn(b, t, v, device=dev, requires_grad=True)
    torch.cuda.reset_peak_memory_stats(dev)
    log_probs_from_logits(x2, labels, chunk_size=chunk).sum().backward()
    chk_peak = torch.cuda.max_memory_allocated(dev)

    # single-pass materializes full fp32 [b,t,v]; chunked bounds fp32 to one chunk.
    assert chk_peak < ref_peak * 0.6, f"chunked peak {chk_peak} not << single-pass {ref_peak}"
    print(f"[peak-mem] chunked {chk_peak/1e6:.0f}MB < single-pass {ref_peak/1e6:.0f}MB: OK")


def test_non_contiguous_input():
    """Real caller shape ``logits[:, :-1, :]`` is a non-contiguous view; the chunked
    path must handle it (guards the internal seq-slice + .to()). grad flows back to
    the contiguous leaf, matching the single-pass reference on the same view."""
    torch.manual_seed(12)
    labels = torch.randint(0, V, (B, T))

    # leaf is [B, T+1, V] contiguous; the [:, :-1, :] view fed to the op is [B, T, V]
    # and non-contiguous, mirroring the HF `logits[:, :-1, :]` caller pattern.
    param = torch.randn(B, T + 1, V, requires_grad=True)
    param2 = param.detach().clone().requires_grad_(True)
    x = param[:, :-1, :]
    x2 = param2[:, :-1, :]
    assert not x.is_contiguous(), "test setup: input should be non-contiguous"
    out = log_probs_from_logits(x, labels, chunk_size=CHUNK)
    ref = _ref_log_probs(x2, labels)
    assert torch.allclose(out, ref, atol=1e-6), "non-contiguous log_probs forward mismatch"
    out.sum().backward()
    ref.sum().backward()
    assert torch.allclose(param.grad, param2.grad, atol=1e-6), "non-contiguous log_probs grad mismatch"

    e_param = torch.randn(B, T + 1, V, requires_grad=True)
    e_param2 = e_param.detach().clone().requires_grad_(True)
    e = e_param[:, :-1, :]
    e2 = e_param2[:, :-1, :]
    out_e = entropy_from_logits(e, chunk_size=CHUNK)
    ref_e = _ref_entropy(e2)
    assert torch.allclose(out_e, ref_e, atol=1e-6), "non-contiguous entropy forward mismatch"
    out_e.sum().backward()
    ref_e.sum().backward()
    assert torch.allclose(e_param.grad, e_param2.grad, atol=1e-5), "non-contiguous entropy grad mismatch"
    print("[non-contiguous] sliced-view input (logits[:, :-1, :]) matches reference (both ops): OK")


if __name__ == "__main__":
    test_log_probs_forward_and_grad_match()
    test_entropy_forward_and_grad_match()
    test_bf16_input_grad_dtype_and_close()
    test_entropy_bf16_grad_dtype()
    test_gradcheck_float64()
    test_no_grad_path_unchanged()
    test_input_not_modified_by_backward()
    test_shared_logits_logprobs_plus_entropy()
    test_chunk_disabled_equals_reference()
    test_boundary_chunk_sizes()
    test_chunk_size_one()
    test_non_3d_input_falls_back()
    test_non_contiguous_input()
    test_double_backward_raises()
    test_grad_output_not_modified()
    test_o_of_chunk_peak_memory_cuda()
    print("\nAll checks passed.")
