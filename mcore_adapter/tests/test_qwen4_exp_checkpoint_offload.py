"""Checkpoint CPU offload preserves real TP/SP/DDP training gradients.

Run two torchrun ranks with RUN_QWEN4_CHUNKED_MODEL_TESTS=1.
"""
import copy
from dataclasses import replace

import pytest
import torch

from test_qwen4_exp_chunked_model import distributed, parallel, model_config
from test_qwen4_exp_model import make_model


@pytest.mark.parametrize("granularity", [None, "selective"])
def test_checkpoint_cpu_offload_rejects_missing_full_recomputation(granularity):
    from test_qwen4_exp_model import tiny_config

    with pytest.raises(ValueError, match="checkpoint_cpu_offload requires full recomputation"):
        replace(tiny_config(), checkpoint_cpu_offload=True,
                recompute_granularity=granularity)


@pytest.mark.parametrize("parallel", [1, 2], indirect=True)
def test_cpu_checkpoint_releases_saved_gpu_streams_and_preserves_two_microbatches(parallel):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    config = model_config(parallel)
    config.recompute_granularity = "full"
    config.recompute_method = "uniform"
    config.recompute_num_layers = 1
    reference = make_model(config)
    offloaded_config = copy.deepcopy(config)
    offloaded_config.checkpoint_cpu_offload = True
    offloaded = make_model(offloaded_config)
    offloaded.load_state_dict(reference.state_dict())
    offloaded.decoder.layers[1].ple.ple_embedding.store = reference.decoder.layers[1].ple.ple_embedding.store
    ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=False, overlap_grad_reduce=False)
    baseline = DistributedDataParallel(config, ddp_config, reference)
    actual = DistributedDataParallel(offloaded_config, ddp_config, offloaded)
    batches = [(torch.arange(64, device="cuda").reshape(2, 32) + offset) % 15 + 1 for offset in (0, 5)]
    valid = torch.ones_like(batches[0], dtype=torch.bool)
    valid[0, :4] = False
    positions = (valid.long().cumsum(-1)-1).clamp_min(0)
    shape = (32 // parallel, 2, config.hidden_size * config.hc_count)

    def run(wrapped):
        retained_streams, losses = [], []
        mixer_saved_bytes = []
        in_mixer = False

        def enter_mixer(*_):
            nonlocal in_mixer
            in_mixer = True

        def leave_mixer(*_):
            nonlocal in_mixer
            in_mixer = False

        mixer = wrapped.module.decoder.hyper_connection_mixer
        handles = [mixer.register_forward_pre_hook(enter_mixer),
                   mixer.register_forward_hook(leave_mixer)]

        def pack(value):
            if in_mixer and value.device.type == "cuda":
                mixer_saved_bytes.append(value.numel() * value.element_size())
            if tuple(value.shape) == shape and value.device.type == "cuda":
                retained_streams.append(tuple(value.shape))
            return value.detach()

        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            for ids in batches:
                losses.append(wrapped(ids, positions, valid, labels=ids.roll(-1, -1), loss_mask=valid))
        for handle in handles:
            handle.remove()
        # Observe actual saved-tensor lifetimes while reentrant checkpoints
        # rebuild their autograd graphs. Attention activations must be freed
        # before the same layer's MoE graph is built in CPU-offload mode.
        in_attention = False
        live_attention = 0
        overlap = []

        class AttentionSave:
            def __init__(self, tensor):
                nonlocal live_attention
                self.tensor = tensor.detach()
                live_attention += 1

            def __del__(self):
                nonlocal live_attention
                live_attention -= 1

        def attention_enter(*_):
            nonlocal in_attention
            in_attention = True

        def attention_exit(*_):
            nonlocal in_attention
            in_attention = False

        def mlp_enter(*_):
            if torch.is_grad_enabled():
                overlap.append(live_attention)

        hooks = []
        for layer in wrapped.module.decoder.layers:
            hooks.extend([layer.self_attention.register_forward_pre_hook(attention_enter),
                          layer.self_attention.register_forward_hook(attention_exit),
                          layer.mlp.register_forward_pre_hook(mlp_enter)])

        def track(tensor):
            return AttentionSave(tensor) if in_attention else tensor.detach()

        def unpack(saved):
            return saved.tensor if isinstance(saved, AttentionSave) else saved

        with torch.autograd.graph.saved_tensors_hooks(track, unpack):
            sum((loss * valid).sum()/valid.sum() for loss in losses).backward()
        for hook in hooks:
            hook.remove()
        finalize_model_grads([wrapped])
        return torch.stack(losses).detach(), retained_streams, sum(mixer_saved_bytes), max(overlap)

    expected, baseline_streams, baseline_mixer, baseline_overlap = run(baseline)
    result, offloaded_streams, offloaded_mixer, offloaded_overlap = run(actual)
    assert baseline_overlap > 0
    assert offloaded_overlap == 0, "attention and MoE saved graphs must not overlap"
    assert baseline_mixer > 0
    assert offloaded_mixer == 0, "final GR mixer must not retain its GPU activation graph"
    # CPU hooks are scoped only to checkpoint.save_for_backward. The final GR
    # mixer may still save a stream; each checkpoint must release its own input.
    assert len(baseline_streams) - len(offloaded_streams) >= config.num_layers * len(batches)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    for (name, wanted), (_, got) in zip(reference.named_parameters(), offloaded.named_parameters()):
        torch.testing.assert_close(got.main_grad, wanted.main_grad, atol=0, rtol=0, msg=name)
    for part in ("ple.key_proj", "indexer.index_qk_proj", "attn_hyper_connection.input_mix_weight_down"):
        assert any(part in name and parameter.main_grad.float().norm() > 0
                   for name, parameter in offloaded.named_parameters())
