"""Real Qwen4 model integration for bounded SFT/RL vocabulary losses.

RUN_QWEN4_CHUNKED_MODEL_TESTS=1 torchrun --master_addr=127.0.0.1
    --master_port=29765 --nproc-per-node=2 -m pytest -q -s this_file.py
"""
import copy
import gc
import json
import os

import pytest
import torch
import torch.distributed as dist

from test_qwen4_exp_model import tiny_config, make_model


@pytest.fixture(scope="module")
def distributed():
    if os.environ.get("RUN_QWEN4_CHUNKED_MODEL_TESTS") != "1":
        pytest.skip("requires real Qwen4 Megatron/TE CUDA ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


@pytest.fixture
def parallel(distributed, request):
    from megatron.core import parallel_state, tensor_parallel
    tp = getattr(request, "param", 1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    tensor_parallel.model_parallel_cuda_manual_seed(351)
    torch.manual_seed(351)
    yield tp
    parallel_state.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def model_config(tp, tied=False):
    config = tiny_config()
    config.padded_vocab_size = 1536
    config.tensor_model_parallel_size = tp
    config.expert_tensor_parallel_size = tp
    if tp == 4:
        # Eight-rank capacity candidate: replicated KV groups (< TP), GDN
        # heads divisible by TP, EP8 experts with no expert tensor sharding.
        config.linear_num_key_heads = 4
        config.linear_num_value_heads = 12
        config.num_moe_experts = 16
        config.expert_model_parallel_size = 8
        config.expert_tensor_parallel_size = 1
    config.sequence_parallel = tp > 1
    config.tie_embeddings_and_output_weights = tied
    return config


@pytest.mark.parametrize("parallel", [1, 2], indirect=True)
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_roll_patched_model_bounds_vocabulary_projection(parallel, tied, training, monkeypatch):
    """ROLL's MTP patch must preserve bounded token losses in train and eval."""
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
    from roll.third_party.megatron.mtp_patcher import patch_mtp_functions
    from torch.utils._python_dispatch import TorchDispatchMode

    # Restore the actual upstream methods after exercising ROLL's real patch.
    monkeypatch.setattr(GPTModel, "_postprocess", GPTModel._postprocess)
    monkeypatch.setattr(MultiTokenPredictionLayer, "_get_embeddings",
                        MultiTokenPredictionLayer._get_embeddings)
    patch_mtp_functions()
    cfg = model_config(parallel, tied)
    model = make_model(cfg)
    model.train(training)
    ids = (torch.arange(320, device="cuda").reshape(10, 32) % 15) + 1
    valid = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(32, device="cuda").expand_as(ids)
    labels = ids.roll(-1, -1)
    cfg.vocab_loss_chunk_size = 0
    with torch.no_grad():
        expected = model(ids, positions, valid, labels=labels, loss_mask=valid)
    cfg.vocab_loss_chunk_size = 256
    projected_rows = []

    class ObserveProjection(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            # Observe actual GEMMs, including an unbounded ColumnParallelLinear.
            if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
                if result.shape[-1] == cfg.padded_vocab_size // parallel:
                    projected_rows.append(result.numel() // result.shape[-1])
            return result

    original_postprocess = model._postprocess

    def observe_postprocess(*args, **kwargs):
        with ObserveProjection():
            return original_postprocess(*args, **kwargs)

    monkeypatch.setattr(model, "_postprocess", observe_postprocess)
    with torch.set_grad_enabled(training):
        actual = model(ids, positions, valid, labels=labels, loss_mask=valid)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert projected_rows and max(projected_rows) <= 256, projected_rows
    if training:
        actual.mean().backward()
        head = model.shared_embedding_or_output_weight() if tied else model.output_layer.weight
        assert head.grad is not None and torch.isfinite(head.grad).all() and head.grad.norm() > 0


@pytest.mark.parametrize("parallel", [1, 2], indirect=True)
@pytest.mark.parametrize("tied", [False, True])
def test_full_model_default_bounds_saved_logits_and_matches_ddp_gradients(parallel, tied):
    """A missing integration, wrong untied head, or lost DDP gradient must fail."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    cfg = model_config(parallel, tied)
    # Leave the actual config at its default: real 8K callers must be bounded
    # without requiring a validation-only override.
    cfg.padded_vocab_size = 1536
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg.vocab_loss_chunk_size = 0
    reference = make_model(baseline_cfg)
    actual = make_model(cfg)
    actual.load_state_dict(reference.state_dict())
    actual.decoder.layers[1].ple.ple_embedding.store = reference.decoder.layers[1].ple.ple_embedding.store
    # More than 256 tokens exercises the default chunk boundary in a tiny model.
    ids = (torch.arange(320, device="cuda").reshape(10, 32) % 15) + 1
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, :4] = False
    labels = ids.roll(-1, -1)
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    upstream = torch.linspace(-0.25, 1.0, ids.numel(), device="cuda").view_as(ids) / ids.numel()
    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False, use_distributed_optimizer=False, overlap_grad_reduce=True,
    )
    reference_ddp = DistributedDataParallel(baseline_cfg, ddp_cfg, reference)
    actual_ddp = DistributedDataParallel(cfg, ddp_cfg, actual)
    MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    reference_losses = reference_ddp(ids, positions, valid, labels=labels, loss_mask=valid)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual_losses = actual_ddp(ids, positions, valid, labels=labels, loss_mask=valid)
    full_logits_shape = (32, 10, cfg.padded_vocab_size // parallel)
    assert full_logits_shape not in saved_shapes, "training retained full-sequence vocabulary logits"
    torch.testing.assert_close(actual_losses, reference_losses, atol=1e-4, rtol=1e-4)
    (reference_losses * upstream * valid).sum().backward()
    (actual_losses * upstream * valid).sum().backward()
    finalize_model_grads([reference_ddp])
    finalize_model_grads([actual_ddp])
    expected_params, actual_params = dict(reference.named_parameters()), dict(actual.named_parameters())
    assert expected_params.keys() == actual_params.keys()
    errors = {}
    failures = []
    aggregate_error, aggregate_norm = 0.0, 0.0
    for name, expected in expected_params.items():
        observed = actual_params[name]
        assert hasattr(observed, "main_grad"), name
        delta = (observed.main_grad - expected.main_grad).float()
        reference_norm = expected.main_grad.float().norm()
        errors[name] = float(delta.norm() / reference_norm.clamp_min(1e-20))
        aggregate_error += float(delta.square().sum())
        aggregate_norm += float(expected.main_grad.float().square().sum())
        try:
            torch.testing.assert_close(observed.main_grad, expected.main_grad, atol=2e-3, rtol=2e-2, msg=name)
        except AssertionError as error:
            failures.append(str(error))
    for name in ("embedding.word_embeddings.weight",
                 "decoder.layers.0.self_attention.in_proj.weight",
                 "decoder.hyper_connection_mixer.hc.input_mix_weight_down.weight"):
        # An untied embedding shard with no local input IDs correctly has zero
        # gradient. Require a nonzero result wherever the reference has one.
        if expected_params[name].main_grad.float().norm() > 0:
            assert actual_params[name].main_grad.float().norm() > 0, name
    if not tied:
        assert actual_params["output_layer.weight"].main_grad.float().norm() > 0
    assert (aggregate_error / aggregate_norm) ** 0.5 < 0.01
    print(json.dumps(dict(test="chunked_model_ddp", tp=parallel, tied=tied,
                          gradient_relative_l2=(aggregate_error / aggregate_norm) ** 0.5,
                          max_gradient_relative_l2=max(errors.items(), key=lambda item: item[1]))), flush=True)
    all_failures = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_failures, failures)
    assert not any(all_failures), all_failures
    with torch.no_grad():
        expected_logits = reference(ids, positions, valid)
        actual_logits = actual(ids, positions, valid)
    torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)


@pytest.mark.parametrize("parallel", [1, 2], indirect=True)
def test_reductions_ignore_labels_and_explicit_head_weight(parallel):
    """The hook must use explicit projection weights and preserve token reduction semantics."""
    cfg = model_config(parallel)
    cfg.vocab_loss_chunk_size = 7
    model = make_model(cfg)
    hidden = torch.randn(16 // parallel, 2, 128, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True)
    labels = torch.arange(32, device="cuda").reshape(2, 16)
    labels[0, :3] = -100
    explicit_weight = (model.output_layer.weight.detach() * 0.25).requires_grad_()
    kwargs = dict(labels=labels, weight=model.embedding.word_embeddings.weight,
                  sequence_parallel_enabled=parallel > 1, column_parallel_linear=model.output_layer,
                  col_linear_kwargs={"weight": explicit_weight})
    none = model.compute_output_layer_and_language_model_loss(hidden, **kwargs)
    total = model.compute_output_layer_and_language_model_loss(hidden, **kwargs, reduction="sum")
    mean = model.compute_output_layer_and_language_model_loss(hidden, **kwargs, reduction="mean")
    torch.testing.assert_close(total, none.sum())
    torch.testing.assert_close(mean, none.sum() / (labels != -100).sum())
    torch.testing.assert_close(none[0, :3], torch.zeros(3, device="cuda"))
    logits, _ = model.output_layer(hidden, weight=explicit_weight)
    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
    expected = vocab_parallel_cross_entropy(logits.float(), labels.T.contiguous().clamp_min(0)).T
    torch.testing.assert_close(none[labels != -100], expected[labels != -100])
    mean.backward()
    assert explicit_weight.grad is not None and explicit_weight.grad.norm() > 0
    assert model.output_layer.weight.grad is None


@pytest.mark.parametrize("case", ["bias", "adapter", "deferred"])
def test_unsupported_heads_fail_explicitly(parallel, case):
    """A bounded path must not silently discard an output-head operation."""
    model = make_model(model_config(parallel))
    if case == "bias":
        model.output_layer.bias = torch.nn.Parameter(torch.ones(1536, device="cuda", dtype=torch.bfloat16))
    elif case == "adapter":
        # PEFT's actual Linear adapter supplies a real nonzero low-rank branch.
        from peft.tuners.lora.layer import Linear
        original = model.output_layer
        dense = torch.nn.Linear(128, 1536, bias=False, device="cuda", dtype=torch.bfloat16)
        dense.weight = original.weight
        model.output_layer = Linear(dense, "default", r=2, lora_alpha=2, init_lora_weights=False)
    else:
        model.output_layer.config.defer_embedding_wgrad_compute = True
    hidden = torch.randn(16, 2, 128, device="cuda", dtype=torch.bfloat16)
    labels = torch.arange(32, device="cuda").reshape(2, 16)
    with pytest.raises(NotImplementedError, match="vocab_loss_chunk_size=0"):
        model.compute_output_layer_and_language_model_loss(
            hidden, labels, column_parallel_linear=model.output_layer)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_chunk_configuration_is_rejected(distributed, value):
    from mcore_adapter.models.qwen4_exp.config_qwen4_exp import Qwen4ExpConfig
    with pytest.raises(ValueError, match="vocab_loss_chunk_size"):
        Qwen4ExpConfig(num_layers=4, hidden_size=128, num_attention_heads=4,
                       vocab_loss_chunk_size=value)
