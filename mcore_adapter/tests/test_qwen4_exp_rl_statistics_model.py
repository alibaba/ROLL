"""Tiny real-model integration for bounded Qwen4 RL token statistics.

RUN_QWEN4_RL_STATISTICS_MODEL_TESTS=1 torchrun --master_addr=127.0.0.1 \
    --master_port=29774 --nproc-per-node=2 -m pytest -q -s this_file.py
"""
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from test_qwen4_exp_model import make_model, tiny_config


@pytest.fixture(scope="module")
def distributed():
    if os.environ.get("RUN_QWEN4_RL_STATISTICS_MODEL_TESTS") != "1":
        pytest.skip("requires real Qwen4 Megatron/TE CUDA ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


@pytest.fixture(params=[(1, False), (2, True)], ids=["tp1", "tp2-sp"])
def parallel(distributed, request):
    from megatron.core import parallel_state, tensor_parallel

    tp, sequence_parallel = request.param
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    tensor_parallel.model_parallel_cuda_manual_seed(1771)
    torch.manual_seed(1771)
    yield tp, sequence_parallel
    parallel_state.destroy_model_parallel()
    torch.cuda.empty_cache()


def model_config(parallel, enabled):
    tp, sequence_parallel = parallel
    config = tiny_config()
    config.padded_vocab_size = 512
    config.tensor_model_parallel_size = tp
    config.expert_tensor_parallel_size = tp
    config.sequence_parallel = sequence_parallel
    config.bounded_rl_token_statistics = enabled
    return config


def strategy_for(model, enabled):
    from roll.distributed.strategy.megatron_strategy import MegatronInferStrategy

    strategy = object.__new__(MegatronInferStrategy)
    strategy.model = type("ModelHandle", (), {"config": model.config})()
    strategy.models_unwrapped = [model]
    strategy._bounded_rl_token_statistics_enabled = enabled
    strategy.megatron_train_args = SimpleNamespace(cross_entropy_loss_fusion=False)
    strategy.worker_config = SimpleNamespace(logits_in_fp32=True)
    return strategy


def test_strategy_rejects_processed_temperature_override():
    from roll.distributed.strategy.megatron_strategy import MegatronInferStrategy

    strategy = object.__new__(MegatronInferStrategy)
    strategy.worker_config = SimpleNamespace(
        model_args=SimpleNamespace(
            model_config_kwargs={
                "bounded_rl_token_statistics": True,
                "rl_token_statistics_temperature": 0.7,
            },
            lora_target=[],
        )
    )
    strategy.megatron_train_args = SimpleNamespace(context_parallel_size=1, mtp_num_layers=None)
    strategy.use_sequence_packing = False
    with pytest.raises(ValueError, match="raw logits.*temperature=1"):
        strategy._validate_bounded_rl_token_statistics_request()


def test_inner_forward_step_keeps_shared_microbatch_forward_args_immutable(monkeypatch):
    from tensordict import TensorDict

    from roll.distributed.scheduler.protocol import DataProto
    from roll.distributed.strategy.megatron_strategy import MegatronInferStrategy

    shared_forward_args = {
        "caller_option": "preserved",
        "extra_block_kwargs": {"caller_block_option": "preserved"},
    }
    batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "response_mask": torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]]),
                "final_response_mask": torch.tensor([[0, 0, 1, 0], [0, 1, 0, 1]]),
            },
            batch_size=[2],
        ),
        meta_info={"forward_args": shared_forward_args},
    )
    microbatches = batch.chunk(2)
    assert microbatches[0].meta_info is microbatches[1].meta_info

    strategy = object.__new__(MegatronInferStrategy)
    strategy.use_sequence_packing = False
    strategy._get_feature_on_this_cp_rank = lambda value, _name: value
    strategy._bounded_rl_token_statistics_enabled = True
    strategy.enable_router_replay = False
    strategy.worker_config = SimpleNamespace(apply_loss_scale=False)
    strategy.model = SimpleNamespace(config=SimpleNamespace(virtual_pipeline_model_parallel_size=None))
    monkeypatch.setattr(
        "roll.distributed.strategy.megatron_strategy.RouterReplayHelper.is_r2_record_action",
        staticmethod(lambda *_args, **_kwargs: False),
    )

    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        return torch.zeros(())

    for microbatch in microbatches:
        strategy.inner_forward_step(lambda *_args: None, iter([microbatch]), model)

    assert shared_forward_args == {
        "caller_option": "preserved",
        "extra_block_kwargs": {"caller_block_option": "preserved"},
    }
    assert len(calls) == 2
    for microbatch, call in zip(microbatches, calls):
        expected_labels = strategy._build_next_token_labels(
            microbatch.batch["input_ids"], microbatch.batch["response_mask"]
        )
        assert torch.equal(call["extra_block_kwargs"]["rl_token_labels"], expected_labels)
        assert call["extra_block_kwargs"]["caller_block_option"] == "preserved"
        assert torch.equal(call["loss_mask"], microbatch.batch["final_response_mask"].float())
    assert not torch.equal(
        calls[0]["extra_block_kwargs"]["rl_token_labels"],
        calls[1]["extra_block_kwargs"]["rl_token_labels"],
    )


def independent_statistics(local_logits, input_ids, response_mask, tp_group):
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

    logits = gather_from_tensor_model_parallel_region(local_logits, group=tp_group).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    labels = torch.cat((input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])), dim=1)
    chosen = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    mask = response_mask[:, 1:]
    return chosen[:, :-1] * mask, entropy[:, :-1] * mask


def assert_gradient_parity(reference, actual):
    reference_parameters = dict(reference.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    assert reference_parameters.keys() == actual_parameters.keys()
    required = {
        "output_layer.weight",
        "decoder.hyper_connection_mixer.hc.input_mix_weight_down.weight",
        "decoder.layers.1.ple.key_proj.weight",
        "decoder.layers.0.mlp.router.weight",
        "decoder.layers.3.self_attention.indexer.index_qk_proj.weight",
    }
    for name in required:
        assert name in reference_parameters
        expected = reference_parameters[name].main_grad
        observed = actual_parameters[name].main_grad
        assert expected is not None and expected.float().norm() > 0, name
        assert observed is not None and observed.float().norm() > 0, name

    for name, reference_parameter in reference_parameters.items():
        if not reference_parameter.requires_grad:
            continue
        actual_parameter = actual_parameters[name]
        expected = reference_parameter.main_grad
        observed = actual_parameter.main_grad
        assert (expected is None) == (observed is None), name
        if expected is None:
            continue
        expected_norm = expected.float().norm()
        difference_norm = (observed - expected).float().norm()
        if expected_norm == 0:
            assert difference_norm < 3e-4, (name, float(difference_norm))
        else:
            relative = difference_norm / expected_norm
            assert relative < 0.025, (name, float(relative))


def test_two_microbatch_rl_statistics_match_full_logits_and_gradients(parallel):
    """A wrong label/mask/channel or lost aux gradient must fail this check."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    baseline_config = model_config(parallel, False)
    actual_config = model_config(parallel, True)
    reference = make_model(baseline_config)
    actual = make_model(actual_config)
    actual.load_state_dict(reference.state_dict())
    actual.decoder.layers[1].ple.ple_embedding.store = reference.decoder.layers[1].ple.ple_embedding.store
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False, overlap_grad_reduce=False, use_distributed_optimizer=False,
    )
    reference_ddp = DistributedDataParallel(baseline_config, ddp_config, reference)
    actual_ddp = DistributedDataParallel(actual_config, ddp_config, actual)
    reference_strategy = strategy_for(reference, False)
    actual_strategy = strategy_for(actual, True)
    MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))

    for microbatch in range(2):
        ids = (torch.arange(64, device="cuda").reshape(2, 32) * (microbatch + 3) % 15) + 1
        valid = torch.ones_like(ids)
        valid[0, : 3 + microbatch] = 0
        valid[1, 27 - microbatch :] = 0
        response = valid.clone()
        response[:, : 11 + microbatch] = 0
        auxiliary_mask = response.clone()
        auxiliary_mask[:, 17 + microbatch] = 0
        positions = (valid.cumsum(-1) - 1).clamp_min(0)
        token_labels = actual_strategy._build_next_token_labels(ids, response)
        shifted_labels = ids[:, 1:].clone()
        shifted_labels[response[:, 1:] == 0] = 0
        expected_labels = torch.cat((shifted_labels, torch.zeros_like(ids[:, :1])), dim=1)
        assert torch.equal(token_labels, expected_labels)
        block_kwargs = {"rl_token_labels": token_labels}

        reference_logits = reference_ddp(
            ids, positions, valid, loss_mask=auxiliary_mask, extra_block_kwargs=block_kwargs,
        )
        actual_statistics = actual_ddp(
            ids, positions, valid, loss_mask=auxiliary_mask, extra_block_kwargs=block_kwargs,
        )
        assert actual_statistics.shape == (2, 32, 2)
        assert actual_statistics.dtype == torch.float32

        roll_log_probs = reference_strategy.op_compute_log_probs(reference_logits, ids, response)
        roll_entropy = reference_strategy.op_compute_entropy(reference_logits, response)
        expected_log_probs, expected_entropy = independent_statistics(
            reference_logits, ids, response, reference.output_layer.tp_group,
        )
        torch.testing.assert_close(roll_log_probs, expected_log_probs, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(roll_entropy, expected_entropy, atol=2e-3, rtol=2e-3)

        observed_log_probs = actual_strategy.op_compute_log_probs(actual_statistics, ids, response)
        observed_entropy = actual_strategy.op_compute_entropy(actual_statistics, response)
        torch.testing.assert_close(observed_log_probs, expected_log_probs, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(observed_entropy, expected_entropy, atol=2e-3, rtol=2e-3)
        policy_weight = torch.linspace(-0.4, 1.1, observed_log_probs.numel(), device="cuda").view_as(observed_log_probs)
        reference_loss = (expected_log_probs * policy_weight - 0.07 * expected_entropy).sum()
        actual_loss = (observed_log_probs * policy_weight - 0.07 * observed_entropy).sum()
        reference_loss.backward()
        actual_loss.backward()

    finalize_model_grads([reference_ddp])
    finalize_model_grads([actual_ddp])
    assert_gradient_parity(reference, actual)


def test_sft_and_legacy_two_logit_output_keep_original_paths(parallel):
    """The strategy must use the explicit flag, never a last-dimension heuristic."""
    config = model_config(parallel, True)
    model = make_model(config)
    ids = torch.randint(1, 16, (2, 32), device="cuda")
    valid = torch.ones_like(ids)
    positions = torch.arange(32, device="cuda").expand(2, -1)
    labels = ids.roll(-1, -1)
    losses = model(ids, positions, valid, labels=labels, loss_mask=valid)
    assert losses.shape == labels.shape

    config.bounded_rl_token_statistics = False
    two_logits = torch.randn(2, 32, 2, device="cuda", requires_grad=True)
    legacy_ids = ids.remainder(2 * parallel[0])
    strategy = strategy_for(model, False)
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

    global_logits = gather_from_tensor_model_parallel_region(
        two_logits, group=model.output_layer.tp_group
    ).float().detach().clone()
    # The legacy operator mutates its logits workspace, so preserve the raw
    # global values before calling it.
    observed = strategy.op_compute_log_probs(two_logits, legacy_ids, valid)
    labels = strategy._build_next_token_labels(legacy_ids, valid)
    expected = torch.log_softmax(global_logits, -1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)[:, :-1]
    torch.testing.assert_close(observed, expected)


def test_frozen_plain_head_keeps_transformer_statistics_gradients(parallel):
    config = model_config(parallel, True)
    model = make_model(config)
    model.output_layer.weight.requires_grad_(False)
    ids = torch.randint(1, 16, (2, 32), device="cuda")
    mask = torch.ones_like(ids)
    positions = torch.arange(32, device="cuda").expand(2, -1)
    statistics = model(ids, positions, mask, loss_mask=mask)
    (statistics[..., 0].mean() - 0.05 * statistics[..., 1].mean()).backward()
    assert model.output_layer.weight.grad is None
    gradient = model.decoder.hyper_connection_mixer.hc.input_mix_weight_down.weight.grad
    assert gradient is not None and gradient.float().norm() > 0


@pytest.mark.parametrize("case", ["packing", "mtp", "head_adapter"])
def test_unsupported_rl_statistics_modes_fail_before_projection(parallel, case):
    config = model_config(parallel, True)
    model = make_model(config)
    ids = torch.randint(1, 16, (2, 32), device="cuda")
    positions = torch.arange(32, device="cuda").expand(2, -1)
    mask = torch.ones_like(ids)
    kwargs = {"loss_mask": mask}
    if case == "packing":
        kwargs["packed_seq_params"] = object()
    elif case == "mtp":
        model.config.mtp_num_layers = 1
    else:
        from peft.tuners.lora.layer import Linear

        original = model.output_layer
        dense = torch.nn.Linear(128, 512, bias=False, device="cuda", dtype=torch.bfloat16)
        dense.weight = original.weight
        model.output_layer = Linear(dense, "default", r=2, lora_alpha=2, init_lora_weights=False)
    with pytest.raises(NotImplementedError, match={
        "packing": "packing", "mtp": "MTP", "head_adapter": "output-head",
    }[case]):
        model(ids, positions, mask, **kwargs)
