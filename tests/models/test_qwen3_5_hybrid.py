"""Qwen hybrid-attention regressions; run with torchrun and pytest on 8 GPUs.

torchrun --standalone --nproc_per_node=8 -m pytest -q tests/models/test_qwen3_5_hybrid.py
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def parallel_groups():
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "0")) != 8:
        pytest.skip("requires torchrun with eight CUDA processes")
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.process_groups_config import ProcessGroupCollection

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=3), device_id=torch.cuda.current_device())
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def make_config(num_query_groups: int = 4):
    from mcore_adapter.models.qwen3_5.config_qwen3_5 import Qwen3_5Config

    return Qwen3_5Config(
        num_layers=4,
        hidden_size=384,
        ffn_hidden_size=768,
        num_attention_heads=24,
        num_query_groups=num_query_groups,
        kv_channels=16,
        tensor_model_parallel_size=dist.get_world_size(),
        normalization="RMSNorm",
        attention_output_gate=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
        swiglu=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        vision_config={},
        rope_scaling={"mrope_section": [1, 1, 0], "rope_theta": 10000, "partial_rotary_factor": 0.25},
    )


def build_attention(config, parallel_groups, layer_index: int):
    from mcore_adapter.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from megatron.core.transformer.spec_utils import build_module

    # Exercise the model's real spec selection without allocating its vision encoder.
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    torch.nn.Module.__init__(model)
    model.config = config
    model.vp_stage = None
    block = model._get_transformer_layer_spec()
    return build_module(
        block.layer_specs[layer_index].submodules.self_attention,
        config=config,
        layer_number=layer_index + 1,
        pg_collection=parallel_groups,
    ).cuda()


@pytest.mark.parametrize("num_query_groups", [4, 8])
def test_output_gate_matches_local_query_heads(parallel_groups, num_query_groups):
    """Replicated KV heads must not leave twice as many gate heads as query heads."""
    config = make_config(num_query_groups)
    attention = build_attention(config, parallel_groups, 3)
    hidden = torch.randn(8, 1, config.hidden_size, device="cuda", requires_grad=True)
    query, key, value, gate = attention.get_query_key_value_tensors(hidden, output_gate=True)
    expected_heads = config.num_attention_heads // dist.get_world_size()
    assert query.shape == gate.shape == (8, 1, expected_heads, 16)
    loss = (query * gate.sigmoid()).square().mean() + key.square().mean() + value.square().mean()
    loss.backward()
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad.abs().sum() > 0


def test_output_gate_uses_rank_local_slice(parallel_groups):
    """Gate values, not just dimensions, must belong to this rank's query heads."""
    from megatron.core.tensor_parallel.mappings import all_gather_last_dim_from_tensor_parallel_region

    config = make_config()
    attention = build_attention(config, parallel_groups, 3)
    hidden = torch.randn(8, 1, config.hidden_size, device="cuda")
    dist.broadcast(hidden, src=0)
    with torch.no_grad():
        projected, _ = attention.linear_qkv(hidden)
        full_projection = all_gather_last_dim_from_tensor_parallel_region(projected)
        # Each KV group contains 6 query heads, 6 gate heads, one K and one V.
        grouped = full_projection.reshape(8, 1, 4, 14, 16)
        full_gates = grouped[:, :, :, 6:12, :].reshape(8, 1, 24, 16)
        rank = dist.get_rank()
        expected = full_gates[:, :, rank * 3 : (rank + 1) * 3, :]
        *_, gate = attention.get_query_key_value_tensors(hidden, output_gate=True)
    torch.testing.assert_close(gate, expected)


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_torch_gdn_backend_bypasses_conv_and_delta_kernels(parallel_groups, monkeypatch, sequence_parallel):
    """The fallback must bypass causal-conv1d and the FLA delta-rule kernel."""
    import megatron.core.ssm.gated_delta_net as gdn

    config = make_config()
    config.gdn_backend = "torch"
    config.sequence_parallel = sequence_parallel
    config.params_dtype = torch.bfloat16
    attention = build_attention(config, parallel_groups, 0)

    def unavailable(*args, **kwargs):
        raise RuntimeError("fast GDN kernel unavailable in this environment")

    monkeypatch.setattr(gdn, "causal_conv1d_fn", unavailable)
    monkeypatch.setattr(gdn, "chunk_gated_delta_rule", unavailable)
    hidden = torch.randn(8, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attention.train()
    training_output, _ = attention(hidden, attention_mask=None)
    assert torch.isfinite(training_output).all()
    training_output.square().mean().backward()
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad.abs().sum() > 0
    attention.eval()
    with torch.no_grad():
        evaluation_output, _ = attention(hidden.detach(), attention_mask=None)
    torch.testing.assert_close(training_output, evaluation_output)
    assert config.deterministic_mode is False


def test_backend_config_is_saved_and_validated(parallel_groups, tmp_path):
    from mcore_adapter.models.qwen3_5.config_qwen3_5 import Qwen3_5Config

    config = make_config()
    config.gdn_backend = "torch"
    path = tmp_path / "config.json"
    config.to_json_file(str(path))
    restored = Qwen3_5Config.from_json_file(str(path))
    assert restored.gdn_backend == "torch"
    config.gdn_backend = "unknown"
    with pytest.raises(ValueError, match="gdn_backend"):
        config.__post_init__()


def test_default_backend_keeps_megatron_kernel_selection(parallel_groups):
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    config = make_config()
    attention = build_attention(config, parallel_groups, 0)
    assert type(attention) is GatedDeltaNet
    assert attention.config.deterministic_mode is False


def test_already_sliced_gate_is_preserved(parallel_groups, monkeypatch):
    """Newer Megatron's correct local gate must not be sliced a second time."""
    from megatron.core.transformer.attention import SelfAttention

    attention = build_attention(make_config(), parallel_groups, 3)
    query = torch.randn(8, 1, 3, 16, device="cuda")
    key = torch.randn(8, 1, 1, 16, device="cuda")
    value = torch.randn_like(key)
    expected_gate = torch.randn_like(query)

    def upstream_with_fix(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
        return query, key, value, expected_gate

    monkeypatch.setattr(SelfAttention, "get_query_key_value_tensors", upstream_with_fix)
    *_, gate = attention.get_query_key_value_tensors(query, output_gate=True)
    torch.testing.assert_close(gate, expected_gate)
