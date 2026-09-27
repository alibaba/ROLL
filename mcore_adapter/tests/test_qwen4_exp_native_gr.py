"""Native GR rounding boundaries and independently computed derivatives."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).parents[1] / 'src/mcore_adapter/models/qwen4_exp'
PACKAGE = '_qwen38_native_gr_test'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f'{PACKAGE}.hyperconnection', ROOT / 'hyperconnection.py')
hc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hc
spec.loader.exec_module(hc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native GR GEMM rounding requires CUDA')
@pytest.mark.parametrize('rows', [149, 2048, 8192])
@pytest.mark.parametrize('adapter_target,adapter_mode', [
    (None, None),
    ('input_mix_weight_down', 'active'), ('block_inject_weight', 'active'),
    ('input_mix_weight_down', 'disabled'), ('block_inject_weight', 'disabled'),
    ('input_mix_weight_down', 'merged'), ('block_inject_weight', 'merged'),
])
def test_gr_projections_match_native_merged_gemm_with_active_adapters(rows, adapter_target, adapter_mode):
    # Separate four-column GEMMs round differently from the native padded GEMM.
    # Wrapping either projection must not bypass its adapter to fuse the bases.
    torch.manual_seed(692)
    module = hc.HyperConnection(2560, 4, 320, dtype=torch.bfloat16, device='cuda')
    x = torch.randn(rows, 10240, device='cuda', dtype=torch.bfloat16)
    down, injection = module.input_mix_weight_down, module.block_inject_weight
    merged_weight = torch.cat((down.weight, injection.weight, injection.weight.new_zeros(12, 10240)))
    expected = torch.nn.functional.linear(x, merged_weight)
    expected_down, expected_injection = expected[:, :320], expected[:, 320:324]
    if adapter_target:
        peft = pytest.importorskip('peft.tuners.lora.layer')
        layer = peft.Linear(getattr(module, adapter_target), 'default', r=2, lora_alpha=2)
        with torch.no_grad():
            layer.lora_A['default'].weight.normal_(std=0.01)
            layer.lora_B['default'].weight.normal_(std=0.01)
        setattr(module, adapter_target, layer)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x, layer.lora_A['default'].weight),
            layer.lora_B['default'].weight,
        )
        if adapter_mode == 'disabled':
            layer.enable_adapters(False)
        elif adapter_mode == 'merged':
            layer.merge()
            expected = torch.nn.functional.linear(x, torch.cat((
                down.weight, injection.weight, injection.weight.new_zeros(12, 10240),
            )))
            expected_down, expected_injection = expected[:, :320], expected[:, 320:324]
        else:
            if adapter_target == 'input_mix_weight_down':
                expected_down = expected_down + delta
            else:
                expected_injection = expected_injection + delta
    with torch.no_grad():
        block, observed_injection = module._mix_from_normed(x)
        expected_block = hc.hc_gate_mix(
            x, module.input_mix_weight_up(hc.hc_silu(expected_down, 4)), 4,
        )
    torch.testing.assert_close(observed_injection, expected_injection, rtol=0, atol=0)
    torch.testing.assert_close(block, expected_block, rtol=0, atol=0)


@pytest.mark.parametrize('adapter', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_gr_injection_retains_input_weight_and_adapter_gradients(adapter, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA projection backward')
    torch.manual_seed(53)
    module = hc.HyperConnection(7, 4, 5, dtype=torch.float64, device=device)
    if adapter:
        peft = pytest.importorskip('peft.tuners.lora.layer')
        module.block_inject_weight = peft.Linear(
            module.block_inject_weight, 'default', r=2, lora_alpha=2,
        )
        with torch.no_grad():
            module.block_inject_weight.lora_B['default'].weight.normal_()
    layer = module.block_inject_weight
    x = torch.randn(3, 28, dtype=torch.float64, device=device, requires_grad=True)
    upstream = torch.randn(3, 4, dtype=torch.float64, device=device)
    _, output = module._mix_from_normed(x)
    parameters = list(layer.parameters())
    observed = torch.autograd.grad(output, [x] + parameters, upstream)
    independent = [v.detach().clone().requires_grad_() for v in [x] + parameters]
    reference = torch.nn.functional.linear(independent[0], independent[1])
    if adapter:
        reference = reference + torch.nn.functional.linear(
            torch.nn.functional.linear(independent[0], independent[2]), independent[3],
        )
    expected = torch.autograd.grad(reference, independent, upstream)
    for got, wanted in zip(observed, expected):
        torch.testing.assert_close(got, wanted, rtol=1e-12, atol=1e-12)
    assert all(torch.count_nonzero(grad) for grad in observed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA fused projection path')
@pytest.mark.parametrize('hook_kind', ['pre_forward', 'forward', 'backward'])
def test_gr_projection_preserves_registered_module_hooks(hook_kind):
    # Megatron may wait for asynchronous parameter all-gather in a leaf hook.
    module = hc.HyperConnection(7, 4, 5, dtype=torch.float64, device='cuda')
    x = torch.randn(3, 28, dtype=torch.float64, device='cuda', requires_grad=True)
    injection = module.block_inject_weight
    if hook_kind == 'pre_forward':
        def refresh_weights(layer, inputs):
            with torch.no_grad():
                layer.weight.zero_()
        handle = injection.register_forward_pre_hook(refresh_weights)
    elif hook_kind == 'forward':
        handle = injection.register_forward_hook(lambda layer, inputs, output: torch.ones_like(output))
    else:
        handle = injection.register_full_backward_hook(
            lambda layer, grad_inputs, grad_outputs: (torch.zeros_like(grad_inputs[0]),),
        )
    try:
        _, output = module._mix_from_normed(x)
        if hook_kind == 'backward':
            output.sum().backward()
            torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)
            assert torch.count_nonzero(injection.weight.grad)
        else:
            expected = torch.zeros_like(output) if hook_kind == 'pre_forward' else torch.ones_like(output)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
    finally:
        handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA projection backward')
@pytest.mark.parametrize('adapters', [False, True])
def test_gr_complete_mix_gradients_match_independent_equations(adapters):
    torch.manual_seed(61)
    module = hc.HyperConnection(7, 4, 5, device='cuda')
    if adapters:
        peft = pytest.importorskip('peft.tuners.lora.layer')
        for name in ('input_mix_weight_down', 'block_inject_weight'):
            layer = peft.Linear(getattr(module, name), 'default', r=2, lora_alpha=2)
            with torch.no_grad():
                layer.lora_B['default'].weight.normal_(std=0.1)
            setattr(module, name, layer)
    x = torch.randn(3, 28, device='cuda', requires_grad=True)
    parameters = {name: value for name, value in module.named_parameters() if name != 'hc_norm'}
    independent = {name: value.detach().double().requires_grad_() for name, value in parameters.items()}
    independent_x = x.detach().double().requires_grad_()

    def reference_projection(name):
        if adapters and name != 'input_mix_weight_up':
            return torch.nn.functional.linear(independent_x, independent[name + '.base_layer.weight']) + \
                torch.nn.functional.linear(
                    torch.nn.functional.linear(independent_x, independent[name + '.lora_A.default.weight']),
                    independent[name + '.lora_B.default.weight'],
                )
        return torch.nn.functional.linear(independent_x, independent[name + '.weight'])

    down = reference_projection('input_mix_weight_down') / 4
    gate = torch.nn.functional.linear(down * down.sigmoid(), independent['input_mix_weight_up.weight'])
    expected_block = (independent_x.reshape(3, 4, 7) * gate.sigmoid().reshape(3, 4, 7)).mean(1)
    expected_injection = reference_projection('block_inject_weight')
    actual = module._mix_from_normed(x)
    upstream = tuple(torch.randn_like(value) for value in actual)
    observed = torch.autograd.grad(actual, [x, *parameters.values()], upstream)
    expected = torch.autograd.grad(
        (expected_block, expected_injection), [independent_x, *independent.values()],
        tuple(value.double() for value in upstream),
    )
    for got, wanted in zip(observed, expected):
        torch.testing.assert_close(got.double(), wanted, rtol=5e-6, atol=2e-6)
        assert torch.count_nonzero(got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native GR kernels require CUDA')
@pytest.mark.parametrize('rows', [2, 149, 2048])
@pytest.mark.parametrize('operator', ['norm', 'silu', 'mix', 'combine'])
def test_gr_matches_native_bf16_rounding(operator, rows):
    native = pytest.importorskip('vllm.models.qwen3_8_flash_next.nvidia.ops.hc')
    gen = torch.Generator(device='cuda').manual_seed(729)
    groups, width = 4, 2560
    def sample(*shape):
        return torch.randn(shape, device='cuda', generator=gen).bfloat16()
    x = sample(rows, groups * width)
    if operator == 'norm':
        w = sample(groups * width)
        actual = hc.grouped_gemma_rmsnorm(x, w, 1e-6, groups)
        expected = native.grouped_gemma_rmsnorm(x, w, 1e-6, groups)
    elif operator == 'silu':
        x = sample(rows, 320)
        actual, expected = hc.hc_silu(x, groups), native.hc_silu(x, groups)
    elif operator == 'mix':
        g = sample(*x.shape)
        actual, expected = hc.hc_gate_mix(x, g, groups), native.hc_gate_mix(x, g, groups)
    else:
        block, inj = sample(rows, width), sample(rows, groups)
        actual = hc.hc_combine(x, block, inj, groups)
        expected = native.hc_combine(x, block, inj, groups)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native GR kernels require CUDA')
@pytest.mark.parametrize('rows', [2, 149, 2048])
@pytest.mark.parametrize('shared_weight', [False, True])
def test_combined_residual_and_norm_match_native_fused_boundary(rows, shared_weight):
    native = pytest.importorskip('vllm.models.qwen3_8_flash_next.nvidia.ops.hc')
    gen = torch.Generator(device='cuda').manual_seed(945)
    groups, width = 4, 2560
    def sample(*shape):
        return torch.randn(shape, device='cuda', generator=gen).bfloat16()
    x, block, injection = sample(rows, groups * width), sample(rows, width), sample(rows, groups)
    weight = sample(width if shared_weight else groups * width)
    actual = hc.hc_combine(x, block, injection, groups)
    normed = hc.grouped_gemma_rmsnorm(actual, weight, 1e-6, groups, after_combine=True)
    expected, expected_normed = native.hc_combine_norm(x, block, injection, weight, 1e-6, groups)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(normed, expected_normed, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA derivatives')
@pytest.mark.parametrize('operator', ['norm', 'combined_norm', 'silu', 'mix', 'combine'])
def test_gr_cuda_gradients_match_independent_double_equations(operator):
    gen = torch.Generator().manual_seed(431)
    groups, width = 4, 17
    def sample(*shape):
        return torch.randn(shape, generator=gen).cuda().requires_grad_()
    x = sample(3, 2, groups * width).transpose(0, 1)
    if operator in ('norm', 'combined_norm'):
        values = (x, sample(groups * width))
        def reference(a, w):
            grouped = a.reshape(-1, groups, width)
            y = grouped * (grouped.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            return (y * (1 + w.reshape(1, groups, width))).reshape_as(a)
        actual = hc.grouped_gemma_rmsnorm(*values, 1e-6, groups, after_combine=operator == 'combined_norm')
    elif operator == 'silu':
        values = (x,)
        def reference(a):
            return (a / groups) * (a / groups).sigmoid()
        actual = hc.hc_silu(x, groups)
    elif operator == 'mix':
        values = (x, sample(*x.shape))
        def reference(a, g):
            return (a * g.sigmoid()).unflatten(-1, (groups, width)).mean(-2)
        actual = hc.hc_gate_mix(*values, groups)
    else:
        values = (x, sample(2, 3, width), sample(2, 3, groups))
        def reference(a, b, inj):
            write = 2 * (inj / groups).sigmoid()
            return (a.unflatten(-1, (groups, width)) + b.unsqueeze(-2) * write.unsqueeze(-1)).flatten(-2)
        actual = hc.hc_combine(*values, groups)
    independent = tuple(v.detach().double().requires_grad_() for v in values)
    expected = reference(*independent)
    grad = torch.randn(actual.shape, generator=gen).cuda()
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
    observed = torch.autograd.grad(actual, values, grad)
    wanted = torch.autograd.grad(expected, independent, grad.double())
    for got, want in zip(observed, wanted):
        torch.testing.assert_close(got.double(), want, atol=3e-6, rtol=3e-6)
