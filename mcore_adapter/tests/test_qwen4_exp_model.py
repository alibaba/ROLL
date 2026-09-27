"""Complete tiny text model transport and recomputation regression on H800."""
import copy
import os

import pytest
import torch


@pytest.fixture(scope="module")
def environment():
    if os.environ.get("RUN_QWEN4_MODEL_TESTS") != "1":
        pytest.skip("requires real Megatron/TE CUDA environment")
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29748", RANK="0", WORLD_SIZE="1")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(717)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def tiny_config():
    from mcore_adapter.models.qwen4_exp.config_qwen4_exp import Qwen4ExpConfig
    from megatron.core.transformer.enums import AttnBackend
    return Qwen4ExpConfig(
        num_layers=4, hidden_size=128, num_attention_heads=4, num_query_groups=2,
        kv_channels=32, ffn_hidden_size=256, padded_vocab_size=256, max_sequence_length=64,
        normalization="RMSNorm", layernorm_epsilon=1e-6, transformer_impl="transformer_engine", bf16=True,
        params_dtype=torch.bfloat16, add_bias_linear=False, gated_linear_unit=True,
        hidden_dropout=0.0, attention_dropout=0.0, experimental_attention_variant="gated_delta_net",
        linear_attention_type="gated_delta_net", linear_attention_freq=4, linear_conv_kernel_dim=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=6, num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64,
        gdn_output_gate_type="sigmoid",
        moe_grouped_gemm=True, moe_router_load_balancing_type="none",
        moe_token_dispatcher_type="alltoall", moe_shared_expert_intermediate_size=64,
        moe_shared_expert_gate=True, layernorm_zero_centered_gamma=True, qk_layernorm=True,
        attention_output_gate=True, hc_count=4, hc_lowrank=16,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16,
        indexer_compress_ratio=4, ple_layer_ids=[2], ple_embed_dim=128,
        qsa_indexer_kl_coef=0.01,
        ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=16, eos_token_id=0, rotary_percent=0.25,
        position_embedding_type="rope", gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
    )


def make_model(config):
    from mcore_adapter.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
    from mcore_adapter.models.qwen4_exp.ngram_embedding import TensorNGramStore
    model = Qwen4ExpModel(config)
    model.decoder.layers[1].ple.ple_embedding.store = TensorNGramStore(torch.randn(16, 32).bfloat16())
    return model


def test_full_model_uses_original_ids_and_all_required_modules(environment):
    torch.manual_seed(718)
    config = tiny_config()
    model = make_model(config)
    assert [layer.attn_hyper_connection.norm_after_combine for layer in model.decoder.layers] == [
        False, False, True, True,
    ]
    assert all(layer.mlp_hyper_connection.norm_after_combine for layer in model.decoder.layers)
    assert model.decoder.hyper_connection_mixer.hc.norm_after_combine
    ids = torch.randint(1, 16, (2, 32), device="cuda")
    positions = torch.arange(32, device="cuda").expand(2, -1)
    labels = ids.roll(-1, -1)
    losses = model(ids, positions, None, labels=labels)
    assert losses.shape == (2, 32)
    losses.mean().backward()
    required = ["embedding.word_embeddings.weight", "decoder.layers.1.ple.key_proj.weight",
                "decoder.layers.3.self_attention.indexer.index_qk_proj.weight",
                "decoder.hyper_connection_mixer.hc.input_mix_weight_down.weight"]
    params = dict(model.named_parameters())
    for name in required:
        assert params[name].grad is not None and torch.isfinite(params[name].grad).all(), name
        assert params[name].grad.float().norm() > 0, name
    assert not any("final_layernorm" in name for name in params)
    assert not any("in_proj.layer_norm" in name or "linear_qkv.layer_norm" in name for name in params)


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_gdn_output_gate_and_gradients_match_checkpoint_semantics(environment, activation):
    config = tiny_config()
    config.gdn_output_gate_type = activation
    gdn = make_model(config).decoder.layers[0].self_attention
    assert gdn.activation == "silu", "the output gate must not change convolution activation"
    x = torch.randn(4, 6, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    actual = gdn._apply_gated_norm(x, gate)
    normed = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config.layernorm_epsilon)
    normed = normed * (gdn.out_norm.weight.float() + 1)
    activate = torch.sigmoid if activation == "sigmoid" else torch.nn.functional.silu
    expected = (normed * activate(gate.float())).bfloat16().reshape_as(actual)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    upstream = torch.randn_like(actual)
    inputs = (x, gate, gdn.out_norm.weight)
    gradients = torch.autograd.grad(actual, inputs, upstream, retain_graph=True)
    reference = torch.autograd.grad(expected, inputs, upstream)
    for got, want in zip(gradients, reference):
        relative = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-9)
        assert relative < 0.02


def test_full_recompute_keeps_each_microbatch_ids(environment):
    torch.manual_seed(719)
    config = tiny_config()
    baseline = make_model(config)
    recompute_config = copy.deepcopy(config)
    recompute_config.recompute_granularity = "full"
    recompute_config.recompute_method = "uniform"
    recompute_config.recompute_num_layers = 3
    recompute = make_model(recompute_config)
    recompute.load_state_dict(baseline.state_dict())
    recompute.decoder.layers[1].ple.ple_embedding.store = baseline.decoder.layers[1].ple.ple_embedding.store
    positions = torch.arange(32, device="cuda").expand(2, -1)
    batches = [torch.randint(1, 16, (2, 32), device="cuda") for _ in range(2)]
    result = [baseline(ids, positions, None, labels=ids.roll(-1, -1)).mean() for ids in batches]
    replay = [recompute(ids, positions, None, labels=ids.roll(-1, -1)).mean() for ids in batches]
    torch.testing.assert_close(torch.stack(result), torch.stack(replay), atol=2e-2, rtol=2e-2)
    sum(result).backward()
    sum(replay).backward()
    for (name, p), (_, rp) in zip(baseline.named_parameters(), recompute.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, rp.grad, atol=2e-2, rtol=2e-2, msg=name)


def test_roll_padding_mask_reaches_qsa_and_ple(environment):
    """ROLL's public forward signature must exclude padding in both modules."""
    model = make_model(tiny_config())
    ids = torch.randint(1, 16, (2, 32), device="cuda")
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, :4] = False
    valid[1, 24:] = False
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    captured = {}

    def capture_ple(module, args, kwargs):
        captured["ids"] = args[1].clone()
        captured["ple_valid"] = kwargs["valid_mask"].clone()

    def capture_indexer(module, args):
        captured["qsa_valid"] = args[2].clone()

    ple = model.decoder.layers[1].ple
    indexer = model.decoder.layers[3].self_attention.indexer
    handles = [ple.register_forward_pre_hook(capture_ple, with_kwargs=True),
               indexer.register_forward_pre_hook(capture_indexer)]
    try:
        model(ids, positions, valid.long(), labels=ids.roll(-1, -1))
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(captured["ple_valid"], valid)
    torch.testing.assert_close(captured["qsa_valid"], valid)
    torch.testing.assert_close(captured["ids"], ids.masked_fill(~valid, 0))


def test_rejects_custom_attention_mask_before_layers(environment):
    model = make_model(tiny_config())
    ids = torch.ones((1, 32), dtype=torch.long, device="cuda")
    position = torch.arange(32, device="cuda").unsqueeze(0)
    with pytest.raises(ValueError, match="token attention_mask"):
        model(ids, position, torch.zeros(1, 1, 32, 32, device="cuda", dtype=torch.bool))


def test_padding_token_values_do_not_change_valid_logits(environment):
    model = make_model(tiny_config()).eval()
    ids = torch.randint(1, 16, (1, 32), device="cuda")
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[:, :4] = False
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    changed = ids.clone()
    changed[:, :4] = (changed[:, :4] % 15) + 1
    with torch.no_grad():
        original = model(ids, positions, valid)
        perturbed = model(changed, positions, valid)
    torch.testing.assert_close(original[:, 4:], perturbed[:, 4:], atol=0, rtol=0)


@pytest.mark.parametrize("case", ["nonbinary", "conflict", "custom_with_padding"])
def test_invalid_mask_combinations_fail_before_forward(environment, case):
    from mcore_adapter.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
    ids = torch.ones((1, 8), dtype=torch.long, device="cuda")
    attention = torch.ones_like(ids)
    padding = torch.zeros_like(ids, dtype=torch.bool)
    if case == "nonbinary":
        attention[0, 0] = -1
    elif case == "conflict":
        padding[0, 0] = True
    else:
        attention = torch.zeros((1, 1, 8, 8), dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="mask"):
        Qwen4ExpModel._resolve_valid_mask(ids, attention, padding)
