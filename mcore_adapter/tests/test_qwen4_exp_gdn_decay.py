"""Exercise the real GDN forward and recurrent kernel with BF16 A_log storage.

RUN_QWEN4_GDN_DECAY_TESTS=1 enables this CUDA test. QWEN4_GDN_FORWARD_SOURCE
optionally selects an isolated, patched source file for validation before applying
the dependency patch; its genuine GDN.forward is bound to the real layer.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
import torch.distributed as dist

from test_qwen4_exp_model import tiny_config, make_model


@pytest.fixture(scope="module")
def environment(tmp_path_factory):
    if os.environ.get("RUN_QWEN4_GDN_DECAY_TESTS") != "1":
        pytest.skip("requires real Megatron/TE/FLA CUDA environment")
    from megatron.core import parallel_state, tensor_parallel
    torch.cuda.set_device(0)
    rendezvous = tmp_path_factory.mktemp("gdn-decay-pg") / "init"
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(4201)
    torch.manual_seed(4201)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_real_gdn_forward_decay_and_parameter_gradient_use_fp32_exp(environment):
    """BF16 exp rounding, detached decay, or changing parameter dtype must fail."""
    import megatron.core.ssm.gated_delta_net as core_gdn
    model = make_model(tiny_config())
    gdn = model.decoder.layers[0].self_attention
    source_path = Path(core_gdn.__file__)
    if os.environ.get("QWEN4_GDN_FORWARD_SOURCE"):
        source_path = Path(os.environ["QWEN4_GDN_FORWARD_SOURCE"])
        module_spec = importlib.util.spec_from_file_location("qwen4_decay_preview_gdn", source_path)
        preview = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = preview
        module_spec.loader.exec_module(preview)
        gdn.forward = types.MethodType(preview.GatedDeltaNet.forward, gdn)
    assert gdn.A_log.dtype == torch.bfloat16
    with torch.no_grad():
        gdn.A_log.copy_(torch.tensor([-1.625, -0.625, 0.328125, 0.734375, 1.453125, 2.375],
                                     device="cuda", dtype=torch.bfloat16))
        gdn.dt_bias.copy_(torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                                      device="cuda", dtype=torch.bfloat16))
    captured = {}

    def capture_projection(module, args, output):
        captured["alpha"] = output[0].detach().transpose(0, 1)[..., -gdn.num_v_heads_local_tp:]

    real_recurrent_kernel = gdn.gated_delta_rule

    def observe_recurrent_kernel(*args, **kwargs):
        captured["g"] = kwargs["g"]
        captured["g"].retain_grad()
        # Run the real FLA recurrent kernel; this observer does not synthesize
        # its output or bypass its backward implementation.
        return real_recurrent_kernel(*args, **kwargs)

    handle = gdn.in_proj.register_forward_hook(capture_projection)
    gdn.gated_delta_rule = observe_recurrent_kernel
    hidden = torch.randn(16, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        output, _ = gdn(hidden, attention_mask=None)
        output.float().square().mean().backward()
    finally:
        handle.remove()
        gdn.gated_delta_rule = real_recurrent_kernel
    observed_g = captured["g"]
    assert observed_g.dtype == torch.float32
    assert gdn.A_log.dtype == torch.bfloat16 and gdn.A_log.grad.dtype == torch.bfloat16
    assert observed_g.grad is not None and observed_g.grad.norm() > 0
    assert gdn.A_log.grad.norm() > 0
    reference_a = gdn.A_log.detach().float().requires_grad_()
    reference_g = -torch.exp(reference_a) * torch.nn.functional.softplus(
        captured["alpha"].float() + gdn.dt_bias.detach().float())
    reference_a_grad = torch.autograd.grad(reference_g, reference_a, observed_g.grad)[0].bfloat16()
    decay_relative = (observed_g.detach() - reference_g.detach()).norm() / reference_g.detach().norm()
    gradient_relative = (gdn.A_log.grad - reference_a_grad).float().norm() / reference_a_grad.float().norm()
    peak = torch.cuda.max_memory_allocated()
    print(json.dumps(dict(test="gdn_fp32_decay", source=str(source_path),
                          source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                          parameter_dtype=str(gdn.A_log.dtype), decay_relative_l2=float(decay_relative),
                          a_log_gradient_relative_l2=float(gradient_relative), gpu_peak_bytes=peak)), flush=True)
    assert peak < 4 * 1024**3
    torch.testing.assert_close(observed_g, reference_g, atol=0, rtol=0)
    torch.testing.assert_close(gdn.A_log.grad, reference_a_grad, atol=0, rtol=0)


def test_dependency_patch_cli_is_locked_replayable_and_rejects_unknown_source(environment, tmp_path):
    """An unrecognized dependency must remain untouched; a known patch is idempotent."""
    import megatron.core.ssm.gated_delta_net as core_gdn
    script = Path(__file__).parents[2] / "scripts/qwen38/patch_megatron_gdn_decay.py"
    assert script.exists(), "SHA-locked GDN decay patch has not been implemented"
    target = tmp_path / "megatron/core/ssm/gated_delta_net.py"
    target.parent.mkdir(parents=True)
    source = Path(core_gdn.__file__).read_bytes()
    # The test can run both before and after deployment of the known patch.
    old = b"A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)"
    new = b"A_log_local_cp.float().exp() * F.softplus(alpha.float() + dt_bias_local_cp)"
    original = source.replace(new, old)
    expected = original.replace(old, new)
    target.write_bytes(original)
    first = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert target.read_bytes() == expected
    initial_stat = target.stat().st_mtime_ns
    second = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert target.stat().st_mtime_ns == initial_stat
    assert target.read_bytes() == expected
    unknown = original + b"\n# a different dependency checkout\n"
    target.write_bytes(unknown)
    refused = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True)
    assert refused.returncode != 0
    assert "SHA256" in refused.stderr
    assert target.read_bytes() == unknown
