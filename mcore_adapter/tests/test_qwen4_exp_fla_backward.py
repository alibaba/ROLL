"""Real FLA backward regressions; opt in with RUN_QWEN38_FLA_TESTS=1 on CUDA.

Use an isolated FLA checkout on PYTHONPATH to compare a dependency patch. No
kernel or backend is replaced in this test. The FP64 recurrence is independent
of FLA, including its initial-state gradient and packed-sequence boundaries.
"""
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def reference(q, k, v, g, beta, initial, offsets):
    factor = v.shape[2] // q.shape[2]
    q, k = (x.repeat_interleave(factor, dim=2) for x in (q, k))
    outputs, states = [], []
    for sequence, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = initial[sequence]
        for token in range(start, end):
            key, value = k[0, token], v[0, token]
            state = state * g[0, token].exp()[:, None, None]
            innovation = value - torch.einsum("hk,hkv->hv", key, state)
            state = state + torch.einsum("hk,hv->hkv", key, innovation * beta[0, token, :, None])
            outputs.append(torch.einsum("hk,hkv->hv", q[0, token], state) * q.shape[-1] ** -.5)
        states.append(state)
    return torch.stack(outputs)[None], torch.stack(states)


def fixture(lengths, key_dim, value_dim, groups, device):
    torch.manual_seed(420719 + key_dim + value_dim + sum(lengths))
    heads, value_heads, size = 2, 2 * groups, sum(lengths)
    q = torch.nn.functional.normalize(torch.randn(1, size, heads, key_dim, device=device), dim=-1)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, size, value_heads, value_dim, device=device) * .3
    g = -.02 - torch.rand(1, size, value_heads, device=device) * .08
    beta = .1 + torch.rand_like(g) * .8
    initial = torch.randn(len(lengths), value_heads, key_dim, value_dim, device=device) * .05
    values = [x.bfloat16() if i < 3 else x.float() for i, x in enumerate((q, k, v, g, beta, initial))]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return values, offsets


@pytest.fixture(scope="module")
def cuda():
    if os.environ.get("RUN_QWEN38_FLA_TESTS") != "1":
        pytest.skip("requires the pinned FLA CUDA environment")
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.12)
    return "cuda"


def test_reference_gradcheck_and_packed_isolation():
    values, offsets = fixture([2, 3], 2, 3, 2, "cpu")
    values = tuple(x.double().requires_grad_() for x in values)
    assert torch.autograd.gradcheck(lambda *xs: reference(*xs, offsets), values,
                                    fast_mode=True, atol=1e-5, rtol=1e-3)
    whole, final = reference(*values, offsets)
    parts = [reference(*(x[:, a:b] for x in values[:5]), values[5][i:i + 1], [0, b - a])
             for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:]))]
    assert torch.equal(whole, torch.cat([p[0] for p in parts], dim=1))
    assert torch.equal(final, torch.cat([p[1] for p in parts]))
    cold, _ = reference(*values[:5], torch.zeros_like(values[5]), offsets)
    assert not torch.equal(whole, cold)


def test_local_value_gradient_at_chunk_tail(cuda):
    from fla.ops.common.chunk_o import chunk_bwd_dv_local

    torch.manual_seed(420719)
    length, heads, value_heads, key_dim, value_dim = 129, 2, 4, 64, 32
    q = torch.randn(1, length, heads, key_dim, device=cuda, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    do = torch.randn(1, length, value_heads, value_dim, device=cuda, dtype=torch.bfloat16)
    g = -torch.rand(1, length, value_heads, device=cuda)
    cu = torch.tensor([0, length], device=cuda, dtype=torch.int32)
    # Exercise the installed Triton implementation rather than an optional
    # backend that might bypass the regressed kernel altogether.
    actual = inspect.unwrap(chunk_bwd_dv_local)(q=q, k=k, do=do, g=g,
        scale=key_dim ** -.5, cu_seqlens=cu)
    torch.cuda.synchronize()
    expected = torch.zeros_like(do, dtype=torch.float64)
    q_ref, k_ref = (x.double().repeat_interleave(value_heads // heads, dim=2) for x in (q, k))
    for start in range(0, length, 64):
        end = min(start + 64, length)
        attention = torch.einsum("bihk,bjhk->bhij", k_ref[:, start:end], q_ref[:, start:end])
        gate = g[:, start:end].double().transpose(1, 2)
        attention *= torch.exp2(gate[:, :, None, :] - gate[:, :, :, None]) * key_dim ** -.5
        attention = attention.triu().to(do.dtype).double()
        expected[:, start:end] = torch.einsum("bhij,bjhv->bihv", attention, do[:, start:end].double())
    assert torch.isfinite(actual).all()
    # Tail-only checks prevent the incorrect rows being hidden by an average.
    for positions in ([63], [127], [128], list(range(length))):
        left, right = actual[:, positions].double(), expected[:, positions]
        assert (left - right).norm() / right.norm() < .02


@pytest.mark.parametrize("lengths,key_dim,value_dim,groups,state_v_first", [
    ([31], 32, 64, 1, False),
    ([129], 64, 32, 2, False),
    ([17, 131], 32, 64, 2, False),
    ([65, 19], 64, 32, 1, True),
    ([17, 131], 128, 128, 3, False),
    ([17, 131], 128, 128, 3, True),
])
def test_packed_backward_and_nonzero_initial_state(cuda, lengths, key_dim, value_dim, groups, state_v_first):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    values, offsets = fixture(lengths, key_dim, value_dim, groups, cuda)
    values = [x.detach().requires_grad_() for x in values]
    initial = values[5].transpose(-1, -2).contiguous() if state_v_first else values[5]
    out, final = chunk_gated_delta_rule(q=values[0], k=values[1], v=values[2],
        g=values[3], beta=values[4], initial_state=initial, output_final_state=True,
        cu_seqlens=torch.tensor(offsets, device=cuda, dtype=torch.int32),
        use_qk_l2norm_in_kernel=False, state_v_first=state_v_first)
    if state_v_first:
        final = final.transpose(-1, -2)
    do, dh = torch.randn_like(out), torch.randn_like(final) * .1
    gradients = torch.autograd.grad((out, final), values, (do, dh))
    expected_values = [x.detach().double().requires_grad_() for x in values]
    expected_out, expected_final = reference(*expected_values, offsets)
    expected_gradients = torch.autograd.grad((expected_out, expected_final), expected_values, (do.double(), dh.double()))
    for actual, expected in zip((out, final, *gradients), (expected_out, expected_final, *expected_gradients)):
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
        torch.testing.assert_close(actual.double(), expected, atol=.02, rtol=.02)
    assert torch.cuda.max_memory_allocated() < 1024**3


def test_patch_rejects_unknown_dependency_source():
    script = Path(__file__).parents[2] / "scripts/qwen38/patch_fla_hopper_dv.py"
    assert script.exists(), "The pinned FLA dependency patch must be reproducible"
    spec = importlib.util.spec_from_file_location("fla_dv_patch", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="SHA256"):
        module.patch_source(b"# unknown dependency\n")


def test_patch_cli_is_idempotent_and_preserves_unrecognized_source(tmp_path):
    source_path = os.environ.get("QWEN38_FLA_SOURCE")
    if not source_path:
        pytest.skip("requires the pinned source path, without importing FLA or CUDA")
    source = Path(source_path).read_bytes()
    old = b"BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)"
    new = b"BV = min(max(triton.next_power_of_2(V), 64 if IS_NVIDIA_HOPPER else 16), CONST_TILING)"
    source = source.replace(new, old)
    target = tmp_path / "fla/ops/common/chunk_o.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    script = Path(__file__).parents[2] / "scripts/qwen38/patch_fla_hopper_dv.py"
    command = [sys.executable, str(script), str(tmp_path)]
    subprocess.run(command, capture_output=True, text=True, check=True)
    patched = target.read_bytes()
    assert patched != source and patched.replace(new, old) == source
    assert patched.count(new) == 1
    stamp = target.stat().st_mtime_ns
    subprocess.run(command, capture_output=True, text=True, check=True)
    assert target.read_bytes() == patched and target.stat().st_mtime_ns == stamp
    unknown = source + b"\n# different dependency revision\n"
    target.write_bytes(unknown)
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0 and "SHA256" in refused.stderr
    assert target.read_bytes() == unknown
