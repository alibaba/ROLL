"""Completed dispatcher caches must release CPU-restored checkpoint leaves.

Run with two torchrun ranks and RUN_QWEN4_CHUNKED_MODEL_TESTS=1.
The reference uses the exact pinned pre-patch method; actual uses installed code.
"""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import types
import weakref

import pytest
import torch

from test_qwen4_exp_chunked_model import distributed, parallel, model_config
from test_qwen4_exp_model import make_model


def patch_module():
    path = Path(__file__).resolve().parents[2] / 'scripts/qwen38/patch_megatron_moe_cache.py'
    spec = importlib.util.spec_from_file_location('moe_cache_patch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned_source():
    import megatron.core.transformer.moe.token_dispatcher as dispatcher
    patch = patch_module()
    source = Path(dispatcher.__file__).read_bytes()
    checksum = hashlib.sha256(source).hexdigest()
    assert checksum in (patch.BEFORE_SHA256, patch.AFTER_SHA256)
    before = source.replace(patch.NEW, patch.OLD) if checksum == patch.AFTER_SHA256 else source
    return dispatcher, patch, before


def test_patch_is_idempotent_and_rejects_unknown_source():
    _, patch, source = pinned_source()
    after = patch.patch_source(source)
    assert hashlib.sha256(after).hexdigest() == patch.AFTER_SHA256
    assert patch.patch_source(after) == after
    with pytest.raises(RuntimeError, match='SHA256'):
        patch.patch_source(source + b'\n# unexpected source drift\n')
    compile(after, '<patched-dispatcher>', 'exec')


def original_combine():
    dispatcher, _, before = pinned_source()
    tree = ast.parse(before)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'MoEAlltoAllTokenDispatcher')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'combine_postprocess')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<pinned-original-combine>', 'exec'),
         vars(dispatcher), namespace)
    return namespace['combine_postprocess']


@pytest.mark.parametrize('parallel', [1, 2], indirect=True)
def test_completed_moe_cache_releases_inputs_and_exact_two_microbatch_ddp_gradients(parallel, monkeypatch):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.tensor_parallel import random as checkpoint_random
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    config = model_config(parallel)
    config.recompute_granularity = 'full'
    config.recompute_method = 'uniform'
    config.recompute_num_layers = 1
    config.checkpoint_cpu_offload = True
    reference, actual = make_model(config), make_model(copy.deepcopy(config))
    actual.load_state_dict(reference.state_dict())
    actual.decoder.layers[1].ple.ple_embedding.store = reference.decoder.layers[1].ple.ple_embedding.store
    old_combine = original_combine()
    for layer in reference.decoder.layers:
        dispatcher = layer.mlp.token_dispatcher
        dispatcher.combine_postprocess = types.MethodType(old_combine, dispatcher)
    ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=False, overlap_grad_reduce=False)
    expected_ddp = DistributedDataParallel(config, ddp_config, reference)
    actual_ddp = DistributedDataParallel(config, ddp_config, actual)
    batches = [(torch.arange(64, device='cuda').reshape(2, 32) + offset) % 15 + 1
               for offset in (0, 5)]
    valid = torch.ones_like(batches[0], dtype=torch.bool)
    valid[0, :4] = False
    positions = (valid.long().cumsum(-1)-1).clamp_min(0)
    refs = []
    active_moe = False
    original_backward = checkpoint_random.CheckpointFunction.backward
    original_detach = checkpoint_random.detach_variable

    def detach(inputs):
        result = original_detach(inputs)
        if active_moe:
            # Both wrappers alias the actual CPU-unpacked CUDA storage. Store
            # only weakrefs; a Python observer must not create this leak.
            refs.append({'restored': weakref.ref(inputs[0]), 'leaf': weakref.ref(result[0])})
        return result

    def backward(ctx, *args):
        nonlocal active_moe
        function = getattr(ctx.run_function, 'func', ctx.run_function)
        active_moe = getattr(function, '__name__', '') == '_mlp_block'
        try:
            result = original_backward(ctx, *args)
            if active_moe:
                refs[-1]['gradient'] = weakref.ref(result[2])
            return result
        finally:
            active_moe = False

    monkeypatch.setattr(checkpoint_random, 'detach_variable', detach)
    monkeypatch.setattr(checkpoint_random.CheckpointFunction, 'backward', staticmethod(backward))
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    tracker = checkpoint_random.get_cuda_rng_tracker()
    tracker_states = tracker.get_states()

    def run(wrapped):
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        tracker.set_states(tracker_states)
        losses, lifetimes = [], []
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device='cuda'))
        for ids in batches:
            refs.clear()
            loss = wrapped(ids, positions, valid, labels=ids.roll(-1, -1), loss_mask=valid)
            losses.append(loss.detach().clone())
            ((loss * valid).sum()/valid.sum()).backward()
            del loss
            torch.cuda.synchronize()
            assert len(refs) == config.num_layers
            lifetimes.append({key: sum(item[key]() is not None for item in refs)
                              for key in ('restored', 'leaf', 'gradient')})
        finalize_model_grads([wrapped])
        return torch.stack(losses), lifetimes

    expected, baseline_lifetimes = run(expected_ddp)
    observed, actual_lifetimes = run(actual_ddp)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    wanted, got = dict(reference.named_parameters()), dict(actual.named_parameters())
    assert wanted.keys() == got.keys()
    for name in wanted:
        torch.testing.assert_close(got[name].main_grad, wanted[name].main_grad,
                                   atol=0, rtol=0, msg=name)
    for part in ('ple.key_proj', 'indexer.index_qk_proj', 'attn_hyper_connection.input_mix_weight_down',
                 'mlp.router.weight'):
        assert any(part in name and bool(parameter.main_grad.float().norm() > 0)
                   for name, parameter in actual.named_parameters()), part
    # The restored wrapper itself may die while its detached leaf retains the
    # same storage. Require both leaf and gradient witnesses in the RED control.
    assert all(item['leaf'] > 0 and item['gradient'] > 0 for item in baseline_lifetimes)
    assert all(not any(item.values()) for item in actual_lifetimes), actual_lifetimes
    print({'tp': parallel, 'baseline_lifetimes': baseline_lifetimes,
           'actual_lifetimes': actual_lifetimes, 'all_main_grad_exact': True}, flush=True)
