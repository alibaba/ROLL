"""Pristine CPU Adam checkpoints must preserve lazy optimizer semantics."""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def initializer():
    path = Path(__file__).parents[3] / 'roll/third_party/megatron/checkpoint_optimizer_state.py'
    assert path.exists(), 'CPU Adam checkpoint initializer is missing'
    spec = importlib.util.spec_from_file_location('checkpoint_optimizer_state', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.initialize_cpu_adamw_state


@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('amsgrad', [False, True])
def test_pristine_checkpoint_preserves_exact_first_updates(fused, amsgrad):
    values = [torch.linspace(-.7, .9, n) for n in (3, 11)]
    params = [torch.nn.Parameter(p.clone()) for p in values]
    baseline = [torch.nn.Parameter(p.clone()) for p in values]
    kwargs = dict(lr=.013, betas=(.7, .91), weight_decay=.1, fused=fused, amsgrad=amsgrad)
    optimizer, control = torch.optim.AdamW(params, **kwargs), torch.optim.AdamW(baseline, **kwargs)
    prior_gradient = torch.ones_like(params[0])
    params[0].grad = prior_gradient
    rng = torch.get_rng_state().clone()
    initializer()(optimizer)
    assert torch.equal(rng, torch.get_rng_state())
    assert params[0].grad is prior_gradient and params[1].grad is None
    for p, expected in zip(params, values):
        torch.testing.assert_close(p, expected, rtol=0, atol=0)
        assert optimizer.state[p]['step'].item() == 0
        assert torch.count_nonzero(optimizer.state[p]['exp_avg']) == 0
        assert torch.count_nonzero(optimizer.state[p]['exp_avg_sq']) == 0
    saved = copy.deepcopy(optimizer.state_dict())
    restored_params = [torch.nn.Parameter(p.clone()) for p in values]
    restored = torch.optim.AdamW(restored_params, **kwargs)
    restored.load_state_dict(saved)
    for step in range(3):
        for group in (params, baseline, restored_params):
            for index, p in enumerate(group):
                p.grad = torch.full_like(p, .03 * (step + 1) * (index + 1))
        for opt in (optimizer, control, restored):
            opt.step()
        before = copy.deepcopy(optimizer.state_dict())
        initializer()(optimizer)
        torch.testing.assert_close(optimizer.state_dict(), before, rtol=0, atol=0)
        for trained in (params, restored_params):
            for p, q in zip(trained, baseline):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), control.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(restored.state_dict(), control.state_dict(), rtol=0, atol=0)


def test_initialization_failure_restores_existing_gradient(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(3))
    gradient = torch.arange(3.)
    parameter.grad = gradient
    optimizer = torch.optim.AdamW([parameter])
    def fail(*args):
        optimizer.state[parameter]['step'] = torch.tensor(0.)
        optimizer.state[parameter]['exp_avg'] = torch.zeros_like(parameter)
        raise RuntimeError('state allocation failed')
    monkeypatch.setattr(optimizer, '_init_group', fail)
    with pytest.raises(RuntimeError, match='state allocation failed'):
        initializer()(optimizer)
    assert parameter.grad is gradient
    assert not optimizer.state, 'a failed allocation must not leave partial Adam state'


def test_other_optimizer_is_not_silently_reinterpreted():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(3))], lr=.1)
    with pytest.raises(TypeError, match='AdamW'):
        initializer()(optimizer)


@pytest.mark.skipif(os.environ.get('RUN_INITIAL_HYBRID_CPU_TEST') != '1', reason='requires installed Megatron')
def test_native_hybrid_sync_and_distributed_state_extraction_without_cuda():
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    original = torch.nn.Parameter(torch.linspace(-.7, .9, 11, dtype=torch.bfloat16))
    master = original.detach().float().clone()
    before, rng = original.detach().clone(), torch.get_rng_state().clone()
    # Exercise the real native state methods without creating CUDA streams or
    # allocating a GPU model alongside the full-weight validation job.
    hybrid = HybridDeviceOptimizer.__new__(HybridDeviceOptimizer)
    torch.optim.Optimizer.__init__(hybrid, [original], {})
    hybrid.param_update_in_fp32 = True
    hybrid.cpu_optimizers = [torch.optim.AdamW([master], lr=.013, fused=True)]
    hybrid.gpu_optimizer = None
    hybrid.inner_param_to_orig_param = {master: original}
    wrapped = SimpleNamespace(optimizer=hybrid, model_param_group_index_map={original: (0, 0)},
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True))
    extract = DistributedOptimizer._get_main_param_and_optimizer_states
    with pytest.raises(KeyError, match='master_param'):
        extract(wrapped, original)
    prepare = initializer().__globals__['prepare_cpu_adam_for_checkpoint']
    prepare(SimpleNamespace(chained_optimizers=[wrapped]))
    state = extract(wrapped, original)
    assert set(state) == {'param', 'exp_avg', 'exp_avg_sq'}
    assert state['param'] is master
    assert hybrid.state[original]['master_param'] is master
    assert hybrid.state[original]['step'].item() == 0
    assert all(t.device.type == 'cpu' for t in state.values())
    torch.testing.assert_close(original, before, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
