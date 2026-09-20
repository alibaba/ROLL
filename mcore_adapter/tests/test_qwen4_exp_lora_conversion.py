"""CPU regressions for exact Qwen4 MCA-to-HF LoRA conversion.

RUN_QWEN4_STREAMING_TESTS=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q <this file>
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN4_STREAMING_TESTS") != "1",
    reason="requires the installed Megatron converter",
)

RANK = 2
HIDDEN = 8


def _converter(tp_size=1, ep_rank=0):
    from mcore_adapter.models.converter.model_converter import ModelConverter

    config = SimpleNamespace(
        hf_model_type="qwen4_exp",
        num_moe_experts=512,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        num_layers=1,
        hidden_size=HIDDEN,
        moe_ffn_hidden_size=16,
        swiglu=True,
        moe_grouped_gemm=True,
        transformer_impl="transformer_engine",
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        tie_embeddings_and_output_weights=False,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
    )
    result = ModelConverter(
        config,
        tensor_model_parallel_rank=0,
        pipeline_model_parallel_rank=0,
        expert_model_parallel_rank=ep_rank,
        expert_tensor_parallel_rank=0,
        to_hf=True,
        efficient_mode=True,
    )
    result.template = copy.deepcopy(result.template)
    result.template._expert_buffer.clear()
    result.template.release()
    return result


def _matrix(rows, cols, offset):
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return (values + offset) / 17


def _duplicated(weight, tp_size):
    return [weight.clone() for _ in range(tp_size)]


def _row_shards(weight, tp_size):
    return [part.clone() for part in torch.chunk(weight, tp_size, dim=1)]


def _swiglu_shards(gate, up, tp_size):
    gate_shards = torch.chunk(gate, tp_size, dim=0)
    up_shards = torch.chunk(up, tp_size, dim=0)
    return [torch.cat([gate_shards[i], up_shards[i]], dim=0) for i in range(tp_size)]


def _convert(item, name, shards):
    return item.convert_to_hf(
        {name: shards}, vp_stage=0, lora_rank=RANK, expert_format="per_expert"
    )


def _assert_owned(weight):
    assert weight.untyped_storage().nbytes() == weight.numel() * weight.element_size()


def _assert_pair(actual, hf_name, expected_a, expected_b):
    a_name = f"{hf_name}.lora_A.weight"
    b_name = f"{hf_name}.lora_B.weight"
    assert set(actual) == {a_name, b_name}
    torch.testing.assert_close(actual[a_name], expected_a, rtol=0, atol=0)
    torch.testing.assert_close(actual[b_name], expected_b, rtol=0, atol=0)
    expected_delta = expected_b @ expected_a
    actual_delta = actual[b_name] @ actual[a_name]
    assert float(actual_delta.norm()) > 0
    torch.testing.assert_close(actual_delta, expected_delta, rtol=0, atol=0)
    _assert_owned(actual[a_name])
    _assert_owned(actual[b_name])


@pytest.mark.parametrize("tp_size", [1, 2])
def test_replicated_and_row_parallel_targets_preserve_exact_delta(tp_size):
    # A missing Qwen4 LoRA dist rule for any entry in this table must fail here.
    cases = [
        (
            "decoder.layers.0.self_attention.out_proj",
            "model.language_model.layers.0.linear_attn.out_proj",
            "row",
        ),
        (
            "decoder.layers.0.self_attention.linear_proj",
            "model.language_model.layers.0.self_attn.o_proj",
            "row",
        ),
        (
            "decoder.layers.0.attn_hyper_connection.input_mix_weight_down",
            "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down",
            "duplicated",
        ),
        (
            "decoder.layers.0.attn_hyper_connection.input_mix_weight_up",
            "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up",
            "duplicated",
        ),
        (
            "decoder.layers.0.attn_hyper_connection.block_inject_weight",
            "model.language_model.layers.0.attn_hyper_connection.block_inject_weight",
            "duplicated",
        ),
        (
            "decoder.layers.0.mlp_hyper_connection.input_mix_weight_down",
            "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down",
            "duplicated",
        ),
        (
            "decoder.layers.0.mlp_hyper_connection.input_mix_weight_up",
            "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_up",
            "duplicated",
        ),
        (
            "decoder.layers.0.mlp_hyper_connection.block_inject_weight",
            "model.language_model.layers.0.mlp_hyper_connection.block_inject_weight",
            "duplicated",
        ),
        (
            "decoder.hyper_connection_mixer.hc.input_mix_weight_down",
            "model.language_model.hyper_connection_mixer.input_mix_weight_down",
            "duplicated",
        ),
        (
            "decoder.hyper_connection_mixer.hc.input_mix_weight_up",
            "model.language_model.hyper_connection_mixer.input_mix_weight_up",
            "duplicated",
        ),
        (
            "decoder.layers.0.ple.key_proj",
            "model.language_model.layers.0.ple.key_proj",
            "duplicated",
        ),
        (
            "decoder.layers.0.ple.value_proj",
            "model.language_model.layers.0.ple.value_proj",
            "duplicated",
        ),
        (
            "decoder.layers.0.self_attention.indexer.index_qk_proj",
            "model.language_model.layers.0.self_attn.indexer.index_qk_proj",
            "duplicated",
        ),
    ]

    for index, (mca_name, hf_name, distribution) in enumerate(cases):
        item = _converter(tp_size)
        a = _matrix(RANK, HIDDEN, 1000 + index * 100)
        b = _matrix(HIDDEN, RANK, 1050 + index * 100)
        a_shards = _row_shards(a, tp_size) if distribution == "row" else _duplicated(a, tp_size)
        actual = {}
        actual.update(_convert(item, f"{mca_name}.lora_A.weight", a_shards))
        actual.update(_convert(item, f"{mca_name}.lora_B.weight", _duplicated(b, tp_size)))
        _assert_pair(actual, hf_name, a, b)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_gdn_in_projection_splits_all_four_hf_targets_with_exact_delta(tp_size):
    item = _converter(tp_size)
    a = _matrix(RANK, HIDDEN, 2000)
    q = _matrix(4, RANK, 2100)
    k = _matrix(4, RANK, 2200)
    v = _matrix(8, RANK, 2300)
    gate = _matrix(8, RANK, 2400)
    beta = _matrix(4, RANK, 2500)
    alpha = _matrix(4, RANK, 2600)
    components = [q, k, v, gate, beta, alpha]
    local_components = [torch.chunk(component, tp_size, dim=0) for component in components]
    b_shards = [
        torch.cat([parts[tp_rank] for parts in local_components], dim=0)
        for tp_rank in range(tp_size)
    ]
    actual = {}
    base = "decoder.layers.0.self_attention.in_proj"
    actual.update(_convert(item, f"{base}.lora_A.weight", _duplicated(a, tp_size)))
    actual.update(_convert(item, f"{base}.lora_B.weight", b_shards))

    expected = {
        "model.language_model.layers.0.linear_attn.in_proj_qkv": torch.cat([q, k, v]),
        "model.language_model.layers.0.linear_attn.in_proj_z": gate,
        "model.language_model.layers.0.linear_attn.in_proj_b": beta,
        "model.language_model.layers.0.linear_attn.in_proj_a": alpha,
    }
    assert len(actual) == 8
    for hf_name, b in expected.items():
        _assert_pair(
            {
                f"{hf_name}.lora_A.weight": actual[f"{hf_name}.lora_A.weight"],
                f"{hf_name}.lora_B.weight": actual[f"{hf_name}.lora_B.weight"],
            },
            hf_name,
            a,
            b,
        )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_qsa_gated_qkv_preserves_q_gate_k_v_layout_and_delta(tp_size):
    item = _converter(tp_size)
    a = _matrix(RANK, HIDDEN, 3000)
    q = _matrix(16, RANK, 3100)
    k = _matrix(4, RANK, 3200)
    v = _matrix(4, RANK, 3300)

    # Explicit Qwen4 layout: for each KV group, Q heads, gate heads, K, then V.
    q_by_group = q.reshape(2, 2, 4, RANK)
    query, gate = torch.chunk(q_by_group, 2, dim=2)
    fused_by_group = torch.cat(
        [query, gate, k.reshape(2, 1, 2, RANK), v.reshape(2, 1, 2, RANK)], dim=1
    )
    fused_shards = [part.reshape(-1, RANK).clone() for part in torch.chunk(fused_by_group, tp_size, dim=0)]

    actual = {}
    base = "decoder.layers.0.self_attention.linear_qkv"
    actual.update(_convert(item, f"{base}.lora_A.weight", _duplicated(a, tp_size)))
    actual.update(_convert(item, f"{base}.lora_B.weight", fused_shards))
    expected = {
        "model.language_model.layers.0.self_attn.q_proj": q,
        "model.language_model.layers.0.self_attn.k_proj": k,
        "model.language_model.layers.0.self_attn.v_proj": v,
    }
    assert len(actual) == 6
    for hf_name, b in expected.items():
        _assert_pair(
            {
                f"{hf_name}.lora_A.weight": actual[f"{hf_name}.lora_A.weight"],
                f"{hf_name}.lora_B.weight": actual[f"{hf_name}.lora_B.weight"],
            },
            hf_name,
            a,
            b,
        )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_shared_expert_fc1_fc2_preserve_gate_up_down_deltas(tp_size):
    item = _converter(tp_size)
    fc1_a = _matrix(RANK, HIDDEN, 4000)
    gate_b = _matrix(16, RANK, 4100)
    up_b = _matrix(16, RANK, 4200)
    fc2_a = _matrix(RANK, 16, 4300)
    fc2_b = _matrix(HIDDEN, RANK, 4400)
    actual = {}
    prefix = "decoder.layers.0.mlp.shared_experts"
    actual.update(_convert(item, f"{prefix}.linear_fc1.lora_A.weight", _duplicated(fc1_a, tp_size)))
    actual.update(_convert(item, f"{prefix}.linear_fc1.lora_B.weight", _swiglu_shards(gate_b, up_b, tp_size)))
    actual.update(_convert(item, f"{prefix}.linear_fc2.lora_A.weight", _row_shards(fc2_a, tp_size)))
    actual.update(_convert(item, f"{prefix}.linear_fc2.lora_B.weight", _duplicated(fc2_b, tp_size)))

    hf_prefix = "model.language_model.layers.0.mlp.shared_expert"
    for projection, a, b in (
        ("gate_proj", fc1_a, gate_b),
        ("up_proj", fc1_a, up_b),
        ("down_proj", fc2_a, fc2_b),
    ):
        hf_name = f"{hf_prefix}.{projection}"
        _assert_pair(
            {
                f"{hf_name}.lora_A.weight": actual[f"{hf_name}.lora_A.weight"],
                f"{hf_name}.lora_B.weight": actual[f"{hf_name}.lora_B.weight"],
            },
            hf_name,
            a,
            b,
        )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("ep_rank,global_expert", [(0, 0), (7, 448)])
def test_routed_expert_export_uses_global_2d_vllm_names_and_exact_delta(
    tp_size, ep_rank, global_expert
):
    item = _converter(tp_size=tp_size, ep_rank=ep_rank)
    fc1_a = _matrix(RANK, HIDDEN, 5000 + global_expert)
    gate_b = _matrix(16, RANK, 5100 + global_expert)
    up_b = _matrix(16, RANK, 5200 + global_expert)
    fc2_a = _matrix(RANK, 16, 5300 + global_expert)
    fc2_b = _matrix(HIDDEN, RANK, 5400 + global_expert)
    actual = {}
    prefix = "decoder.layers.0.mlp.experts"
    actual.update(_convert(item, f"{prefix}.linear_fc1.lora_A.weight0", [fc1_a]))
    actual.update(_convert(item, f"{prefix}.linear_fc1.lora_B.weight0", [torch.cat([gate_b, up_b])]))
    actual.update(_convert(item, f"{prefix}.linear_fc2.lora_A.weight0", [fc2_a]))
    actual.update(_convert(item, f"{prefix}.linear_fc2.lora_B.weight0", [fc2_b]))

    hf_prefix = f"model.language_model.layers.0.mlp.experts.{global_expert}"
    assert all("gate_up_proj" not in name for name in actual)
    for projection, a, b in (
        ("gate_proj", fc1_a, gate_b),
        ("up_proj", fc1_a, up_b),
        ("down_proj", fc2_a, fc2_b),
    ):
        hf_name = f"{hf_prefix}.{projection}"
        _assert_pair(
            {
                f"{hf_name}.lora_A.weight": actual[f"{hf_name}.lora_A.weight"],
                f"{hf_name}.lora_B.weight": actual[f"{hf_name}.lora_B.weight"],
            },
            hf_name,
            a,
            b,
        )
    assert not item.template._expert_buffer
    assert not item.dist_converter.weights_waiting_for_convert
