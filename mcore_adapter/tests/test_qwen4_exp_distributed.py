"""TP1 vs TP2/SP complete model with real Megatron DDP gradient finalization.

Run with torchrun --nproc-per-node=2 and RUN_QWEN4_DISTRIBUTED_TESTS=1.
The same tiny converted weights and CPU table are used on both layouts.
"""
import copy
import gc
import os

import pytest
import torch
import torch.distributed as dist

from test_qwen4_exp_model import tiny_config, make_model


@pytest.fixture(scope="module")
def distributed():
    if os.environ.get("RUN_QWEN4_DISTRIBUTED_TESTS") != "1":
        pytest.skip("requires two real Megatron/TE CUDA ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


def to_hf(config, state, tp_group=None):
    from mcore_adapter.models.converter.model_converter import ModelConverter
    converter = ModelConverter(config, to_hf=True)
    result = {}
    for name, value in sorted(state.items()):
        if name.endswith("._extra_state"):
            continue
        if tp_group is None:
            values = [value.detach().cpu()]
        else:
            values = [torch.empty_like(value) for _ in range(dist.get_world_size(tp_group))]
            dist.all_gather(values, value.contiguous(), group=tp_group)
            values = [v.cpu() for v in values]
        result.update(converter.convert_to_hf({name: values}, vp_stage=0))
    return result


def run_backward(model, config, ids, valid):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False,
                                              use_distributed_optimizer=False)
    wrapped = DistributedDataParallel(config, ddp_config, model)
    MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    positions = (valid.long().cumsum(-1)-1).clamp_min(0)
    result = wrapped(ids, positions, valid, labels=ids.roll(-1, -1), loss_mask=valid)
    loss = (result*valid).sum()/valid.sum()
    loss.backward()
    finalize_model_grads([wrapped])
    gradients = {name: p.main_grad.detach().clone() for name, p in model.named_parameters()}
    return result.detach(), gradients


def test_full_tp2_sp_forward_and_ddp_gradients_match_tp1(distributed):
    from megatron.core import parallel_state, tensor_parallel
    from mcore_adapter.models.converter.model_converter import ModelConverter
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(1729)
    torch.manual_seed(1729)
    config = tiny_config()
    config.hf_model_type = "qwen4_exp"
    config.swiglu = True
    baseline = make_model(config)
    table = baseline.decoder.layers[1].ple.ple_embedding.store
    # Both DP ranks use identical data and initialization in the TP1 reference.
    for p in baseline.parameters():
        dist.broadcast(p.data, src=0)
    hf_weights = to_hf(config, baseline.state_dict())
    ids = torch.arange(64, device="cuda").reshape(2,32)%15+1
    valid = torch.ones_like(ids,dtype=torch.bool)
    valid[0,:4] = False
    valid[1,28:] = False
    reference, reference_grads = run_backward(baseline, config, ids, valid)
    reference_hf_grads = to_hf(config, reference_grads)
    del baseline, reference_grads
    gc.collect()
    torch.cuda.empty_cache()
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    tensor_parallel.model_parallel_cuda_manual_seed(1729)
    tp_config = tiny_config()
    tp_config.hf_model_type = "qwen4_exp"
    tp_config.swiglu = True
    tp_config.tensor_model_parallel_size = 2
    tp_config.expert_tensor_parallel_size = 2
    tp_config.sequence_parallel = True
    model = make_model(tp_config)
    model.decoder.layers[1].ple.ple_embedding.store = table
    converter = ModelConverter(tp_config)
    converted = converter.get_mca_state_dict(iter(hf_weights.items()), vp_stage=0)
    loaded = model.load_state_dict(converted, strict=False)
    assert not [x for x in loaded.missing_keys if not x.endswith("._extra_state")], loaded
    assert not loaded.unexpected_keys, loaded
    actual, actual_grads = run_backward(model, tp_config, ids, valid)
    torch.testing.assert_close(actual[valid], reference[valid], atol=2e-2, rtol=2e-2)
    actual_hf_grads = to_hf(tp_config, actual_grads, parallel_state.get_tensor_model_parallel_group())
    assert actual_hf_grads.keys() == reference_hf_grads.keys()
    errors = {}
    total_error, total_norm = 0.0, 0.0
    for name, expected in reference_hf_grads.items():
        actual = actual_hf_grads[name]
        relative = (actual-expected).float().norm()/expected.float().norm().clamp_min(1e-12)
        errors[name] = float(relative)
        total_error += float((actual-expected).float().square().sum())
        total_norm += float(expected.float().square().sum())
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2, msg=name)
    print("TP2 gradient relative L2 maximum", max(errors.items(),key=lambda item:item[1]))
    print("TP2 aggregate gradient relative L2", (total_error/max(total_norm,1e-24))**0.5)
    assert (total_error/max(total_norm,1e-24))**0.5 < 0.02
    for name in ("model.language_model.layers.1.ple.key_proj.weight",
                 "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
                 "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"):
        expected = reference_hf_grads[name]
        actual = actual_hf_grads[name]
        assert expected.float().norm() > 0 and actual.float().norm() > 0, name
        # A missing/doubled collective must fail even for tiny absolute gradients.
        assert 0.9 < float(actual.float().norm()/expected.float().norm()) < 1.1, name
        assert torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.995, name
    parallel_state.destroy_model_parallel()
