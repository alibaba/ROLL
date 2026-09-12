"""GR arithmetic and actual Megatron TransformerLayer integration regressions.

The independent equations below are transcribed from pinned HF
Qwen4ExpTextGatedResidual and Qwen4ExpTextRMSNorm, not adapter helpers.
Pinned modeling_qwen4_exp.py SHA256:
2a44aeadb215acbb5c75939fcc97e9f14bccff5a51c232826427594993f6a760
Run GPU cases with RUN_MEGATRON_GR_TESTS=1 in the Megatron/TE environment.
"""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

_ROOT = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp"
_PACKAGE = "_qwen4_gr_test"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_ROOT)]
sys.modules[_PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}.{name}", _ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hc = load("hyperconnection")


class ReferenceGR(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.count = source.hc_count
        self.hidden = source.hidden_size
        self.eps = source.eps
        self.norm_weight = nn.Parameter(source.hc_norm.detach().clone())
        self.down = copy.deepcopy(source.input_mix_weight_down)
        self.up = copy.deepcopy(source.input_mix_weight_up)
        self.inject = copy.deepcopy(source.block_inject_weight)

    def forward(self, x):
        grouped = x.float().unflatten(-1, (self.count, self.hidden))
        normalized = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + self.eps)).flatten(-2)
        normalized = (normalized * (1 + self.norm_weight.float())).type_as(x)
        gates = torch.sigmoid(self.up(F.silu(self.down(normalized) / self.count)))
        mixed = (gates.unflatten(-1, (self.count, self.hidden)) * normalized.unflatten(-1, (self.count, self.hidden))).mean(-2)
        write = 2 * torch.sigmoid(self.inject(normalized) / self.count)
        return mixed, write

    def combine(self, x, output, write):
        return (x.unflatten(-1, (self.count, self.hidden)) + output.unsqueeze(-2) * write.unsqueeze(-1)).flatten(-2)


def assert_hc_grads(actual, reference):
    pairs = [(actual.hc_norm, reference.norm_weight),
             (actual.input_mix_weight_down.weight, reference.down.weight),
             (actual.input_mix_weight_up.weight, reference.up.weight),
             (actual.block_inject_weight.weight, reference.inject.weight)]
    for got, want in pairs:
        assert got.grad is not None and want.grad is not None
        torch.testing.assert_close(got.grad, want.grad, atol=3e-6, rtol=3e-5)


def test_gr_reference_forward_and_all_gradients():
    torch.manual_seed(714)
    actual = hc.HyperConnection(8, 4, 5)
    with torch.no_grad():
        actual.hc_norm.uniform_(-0.5, 0.5)
    reference = ReferenceGR(actual)
    x = (torch.randn(3, 2, 32) * torch.tensor([0.1, 1., 3., 10.]).repeat_interleave(8)).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    block = torch.randn(3, 2, 8, requires_grad=True)
    br = block.detach().clone().requires_grad_()
    stream, mixed, injection = actual.mix(x)
    mr, wr = reference(xr)
    y = actual.combine(stream, block, injection)
    yr = reference.combine(xr, br, wr)
    torch.testing.assert_close(mixed, mr, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(y, yr, atol=1e-6, rtol=1e-6)
    cotangent = torch.randn_like(y)
    (y.mul(cotangent).sum() + mixed.square().sum()).backward()
    (yr.mul(cotangent).sum() + mr.square().sum()).backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(block.grad, br.grad, atol=3e-6, rtol=3e-5)
    assert_hc_grads(actual, reference)


def test_hyperconnection_and_final_mixer_mark_sequence_parallel_parameters():
    modules = [hc.HyperConnection(8, 4, 5, sequence_parallel=True),
               hc.HyperConnectionMixer(8, 4, 5, sequence_parallel=True)]
    for module in modules:
        assert all(p.sequence_parallel for p in module.parameters())
        assert all(not p.tensor_model_parallel for p in module.parameters())


@pytest.fixture(scope="module")
def megatron():
    if os.environ.get("RUN_MEGATRON_GR_TESTS") != "1":
        pytest.skip("set RUN_MEGATRON_GR_TESTS=1 for actual Megatron CUDA integration")
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        if "RANK" in os.environ:
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("nccl", init_method=f"file://{tempfile.mktemp()}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    model_parallel_cuda_manual_seed(1234)
    module = load("hyperconnection_layer")
    yield module
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def make_layer(megatron, sequence_parallel=False, backend="te"):
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import build_module
    import torch.distributed as dist
    config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=2,
                               ffn_hidden_size=64, use_cpu_initialization=False,
                               params_dtype=torch.float32, hidden_dropout=0., attention_dropout=0.,
                               add_bias_linear=True, normalization="RMSNorm",
                               gradient_accumulation_fusion=False, sequence_parallel=sequence_parallel,
                               tensor_model_parallel_size=dist.get_world_size())
    config.hc_count, config.hc_lowrank = 4, 8
    base = get_gpt_layer_with_transformer_engine_spec() if backend == "te" else get_gpt_layer_local_spec(normalization="RMSNorm")
    spec = megatron.make_hc_layer_spec(base)
    return build_module(spec, config=config, layer_number=1).cuda()


@pytest.mark.parametrize("backend", ["local", "te"])
def test_real_megatron_zero_branches_are_identity(megatron, backend):
    layer = make_layer(megatron, backend=backend)
    with torch.no_grad():
        for module in (layer.self_attention.linear_proj, layer.mlp.linear_fc2):
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()
    x = torch.randn(4, 2, 128, device="cuda", requires_grad=True)
    y, context = layer(x, attention_mask=None)
    torch.testing.assert_close(y, x, atol=0, rtol=0)
    assert context is None
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)


def test_real_megatron_constructs_without_subsumed_norms(megatron):
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    layer = make_layer(megatron)
    assert not isinstance(layer.self_attention.linear_qkv, TELayerNormColumnParallelLinear)
    assert not isinstance(layer.mlp.linear_fc1, TELayerNormColumnParallelLinear)
    assert not any("layer_norm_weight" in name for name, _ in layer.named_parameters())


@pytest.mark.parametrize("helper", ["_forward_attention", "_forward_mlp"])
def test_split_base_layer_paths_cannot_apply_ordinary_residuals(megatron, helper):
    layer = make_layer(megatron)
    x = torch.randn(4, 2, 32, device="cuda")
    with pytest.raises(NotImplementedError, match="unified GR forward"):
        getattr(layer, helper)(x)


def test_hybrid_specs_remove_only_input_norms(megatron):
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TEColumnParallelLinear
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import get_transformer_layer_with_experimental_attention_variant_spec
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.transformer_config import TransformerConfig
    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=2,
                               experimental_attention_variant="gated_delta_net", linear_attention_freq=2,
                               linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
                               linear_num_key_heads=2, linear_num_value_heads=2)
    specs = get_transformer_layer_with_experimental_attention_variant_spec(config)
    for base in specs:
        converted = megatron.make_hc_layer_spec(base)
        assert converted.submodules.input_layernorm is IdentityOp
        assert converted.submodules.pre_mlp_layernorm is IdentityOp
        old_attention = base.submodules.self_attention.submodules
        new_attention = converted.submodules.self_attention.submodules
        field = "in_proj" if hasattr(new_attention, "in_proj") else "linear_qkv"
        assert getattr(old_attention, field) is TELayerNormColumnParallelLinear
        assert getattr(new_attention, field) is TEColumnParallelLinear
        if hasattr(new_attention, "out_norm"):
            assert new_attention.out_norm is old_attention.out_norm
        assert converted.submodules.mlp.submodules.linear_fc1 is TEColumnParallelLinear


@pytest.mark.parametrize("backend", ["local", "te"])
def test_real_megatron_layer_reference_forward_and_gradients(megatron, backend):
    layer = make_layer(megatron, backend=backend)
    reference_layer = make_layer(megatron, backend=backend)
    reference_layer.load_state_dict(layer.state_dict())
    attention, mlp = reference_layer.self_attention, reference_layer.mlp
    ar, mr = ReferenceGR(layer.attn_hyper_connection), ReferenceGR(layer.mlp_hyper_connection)
    x = torch.randn(4, 2, 128, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    y, _ = layer(x, attention_mask=None)
    ai, aw = ar(xr)
    aout, abias = attention(ai, attention_mask=None)
    aout = aout if abias is None else aout + abias
    yr = ar.combine(xr, aout, aw)
    mi, mw = mr(yr)
    mout, mbias = mlp(mi)
    mout = mout if mbias is None else mout + mbias
    yr = mr.combine(yr, mout, mw)
    torch.testing.assert_close(y, yr, atol=3e-6, rtol=3e-5)
    grad = torch.randn_like(y)
    y.backward(grad)
    yr.backward(grad)
    torch.testing.assert_close(x.grad, xr.grad, atol=3e-6, rtol=3e-5)
    assert_hc_grads(layer.attn_hyper_connection, ar)
    assert_hc_grads(layer.mlp_hyper_connection, mr)
    for actual, expected in ((layer.self_attention, attention), (layer.mlp, mlp)):
        for (name, parameter), (ref_name, reference) in zip(actual.named_parameters(), expected.named_parameters()):
            assert name == ref_name and parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, reference.grad, atol=3e-6, rtol=3e-5, msg=name)


def test_sequence_parallel_hc_parameters_are_reduced_by_megatron(megatron):
    import torch.distributed as dist
    from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    if dist.get_world_size() < 2:
        pytest.skip("requires torchrun --nproc-per-node=2")
    layer = make_layer(megatron, sequence_parallel=True)
    for module in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
        assert all(getattr(p, "sequence_parallel", False) for p in module.parameters())
        # Every rank sees disjoint tokens. Megatron must SUM replicated GR
        # gradients, giving the same result as the unpartitioned reference.
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=0)
        reference = ReferenceGR(module)
        torch.manual_seed(181)
        full_x = torch.randn(8, 2, 128, device="cuda")
        local_x = full_x.chunk(dist.get_world_size(), dim=0)[dist.get_rank()].clone().requires_grad_()
        residual, mixed, injection = module.mix(local_x)
        output = module.combine(residual, mixed.tanh(), injection)
        output.square().sum().backward()
        full_x.requires_grad_()
        rm, rw = reference(full_x)
        reference.combine(full_x, rm.tanh(), rw).square().sum().backward()
        torch.testing.assert_close(local_x.grad, full_x.grad.chunk(dist.get_world_size(), dim=0)[dist.get_rank()], atol=3e-6, rtol=3e-5)
        module.ddp_config = DistributedDataParallelConfig()
        for parameter in module.parameters():
            parameter.main_grad = parameter.grad
        _allreduce_non_tensor_model_parallel_grads([module], layer.config, layer.tp_group)
        assert_hc_grads(module, reference)
