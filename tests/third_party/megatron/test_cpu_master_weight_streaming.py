"""Full-model synchronization must never read parked placeholder parameters."""
from types import SimpleNamespace

import pytest
import torch

from roll.third_party.megatron import cpu_master_params


def fixture():
    model = torch.nn.Module()
    model.config = SimpleNamespace(pipeline_model_parallel_size=1)
    model.register_parameter('weight', torch.nn.Parameter(torch.full((2, 3), -77., dtype=torch.bfloat16)))
    model.register_buffer('hash_offsets', torch.tensor([3, 7], dtype=torch.long))
    master = torch.tensor([1.001, -2.019, 0.125, 3.14159, 42.03125, -0.75])
    shard = model.weight.detach().view(-1)
    interval = SimpleNamespace(start=0, end=6)
    dtype = (torch.bfloat16, torch.bfloat16)
    leaf = SimpleNamespace(
        config=SimpleNamespace(offload_model_from_cpu_master=True),
        buffers=[SimpleNamespace(params=[model.weight])],
        data_parallel_group=None, shard_float16_groups=[[shard]], shard_fp32_groups=[[]],
        opt_group_ranges=[{'params': [model.weight]}],
        model_param_gbuf_map={model.weight: (0, dtype, 0)},
        gbuf_ranges=[{dtype: [{'param_map': {model.weight: {'param': interval}}}]}],
        optimizer=SimpleNamespace(param_groups=[{'params': [shard]}],
            param_to_fp32_param={shard: master}, gpu_params_map_cpu_copy={shard: master}),
    )
    return model, leaf, master


def provider(model, optimizer):
    cls = getattr(cpu_master_params, 'CpuMasterWeightProvider', None)
    assert cls is not None, 'CPU master weight streaming is required instead of reloading the actor'
    return cls([model], optimizer, device='cpu')


def test_streams_current_master_and_real_buffers_without_reading_model_placeholders():
    model, optimizer, master = fixture()
    source = provider(model, optimizer)
    expected = master.clone().reshape(2, 3).bfloat16()
    actual = source('weight', model.weight.detach())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(model.weight, torch.full_like(model.weight, -77))
    master.add_(0.25)
    updated = source('weight', model.weight.detach())
    torch.testing.assert_close(updated, master.reshape(2, 3).bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(source('hash_offsets', model.hash_offsets), model.hash_offsets)
    assert source('hash_offsets', model.hash_offsets).data_ptr() != model.hash_offsets.data_ptr()


@pytest.mark.parametrize('fault', ['missing_master', 'short_range', 'missing_parameter', 'wrong_mode', 'pipeline'])
def test_rejects_invalid_master_ownership_before_export(fault):
    model, optimizer, master = fixture()
    if fault == 'missing_master':
        optimizer.optimizer.param_to_fp32_param.clear()
    elif fault == 'short_range':
        next(iter(optimizer.gbuf_ranges[0].values()))[0]['param_map'][model.weight]['param'].end = 5
    elif fault == 'missing_parameter':
        optimizer.buffers[0].params.clear()
    elif fault == 'wrong_mode':
        optimizer.config.offload_model_from_cpu_master = False
    else:
        model.config.pipeline_model_parallel_size = 2
    with pytest.raises(ValueError):
        provider(model, optimizer)


def test_export_rejects_unknown_key_and_wrong_shape():
    model, optimizer, _ = fixture()
    source = provider(model, optimizer)
    with pytest.raises(ValueError):
        source('unknown', model.weight)
    with pytest.raises(ValueError):
        source('weight', model.weight.flatten())


def test_invalid_local_master_agrees_failure_before_peers_enter_gathers(monkeypatch):
    model, optimizer, _ = fixture()
    optimizer.optimizer.param_to_fp32_param.clear()
    statuses = []
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda status, **kwargs: statuses.append(int(status)))
    with pytest.raises(ValueError):
        provider(model, optimizer)
    assert statuses == [1], 'a failing rank leaves peers blocked in DP gathers'


def test_streaming_strategy_does_not_reload_complete_model():
    import ast
    from pathlib import Path

    path = Path(__file__).parents[3] / 'roll/distributed/strategy/megatron_strategy.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'MegatronTrainStrategy')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'get_model_update_load_kwargs')
    parent = type('Parent', (), {'get_model_update_load_kwargs': lambda self: {'include': ['model_params']}})
    node = ast.ClassDef(name='MegatronTrainStrategy', bases=[ast.Name(id='Parent', ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[], type_params=[])
    ns = {'Parent': parent, 'is_peft_available': lambda: False}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), ns)
    strategy = ns['MegatronTrainStrategy']()
    strategy.worker_config = SimpleNamespace(strategy_args=SimpleNamespace(
        strategy_config={'offload_model_from_cpu_master': True}))
    assert strategy.get_model_update_load_kwargs()['include'] == [], (
        'full actor parameters would overlap with the waking inference model')
    strategy.worker_config.strategy_args.strategy_config.clear()
    assert strategy.get_model_update_load_kwargs()['include'] == ['model_params']


@pytest.mark.skipif(__import__('os').environ.get('RUN_CPU_OPTIMIZER_TESTS') != '1',
                    reason='requires Megatron and CUDA')
@pytest.mark.parametrize('fault', ['missing_master', 'coverage'])
def test_native_streaming_rejects_one_rank_corruption_on_all_ranks(hybrid_parallel, fault):
    import torch.distributed as dist

    model, optimizer, master = fixture()
    if dist.get_rank() == 0:
        if fault == 'missing_master':
            optimizer.optimizer.param_to_fp32_param.clear()
        else:
            shard = optimizer.shard_float16_groups[0][0]
            optimizer.optimizer.param_to_fp32_param[shard] = master[:3]
            optimizer.optimizer.gpu_params_map_cpu_copy[shard] = optimizer.optimizer.param_to_fp32_param[shard]
            shard.data = shard.data[:3]
            next(iter(optimizer.gbuf_ranges[0].values()))[0]['param_map'][model.weight]['param'].end = 3
    with pytest.raises(ValueError, match='CPU master'):
        cpu_master_params.CpuMasterWeightProvider([model], optimizer)
    # This barrier can complete only if every rank returned the validation error.
    dist.barrier()


@pytest.mark.skipif(__import__('os').environ.get('RUN_CPU_OPTIMIZER_TESTS') != '1',
                    reason='requires Megatron and CUDA')
def test_native_streaming_export_preserves_weights_and_next_adam_updates(monkeypatch, hybrid_parallel):
    import types
    from pathlib import Path
    import roll.third_party.megatron.offload_states_patch as offload
    from roll.third_party.megatron.model_update import gather_all_hf_weights
    from tests.third_party.megatron.test_hybrid_phase_alias import (
        test_distributed_hybrid_phase_matches_uninterrupted_trajectory,
    )

    bind = offload.bind_megatron_offload_states_func
    observations = []
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / 'mcore_adapter/tests'))
    import test_qwen4_exp_model as fixture_module
    original_config = fixture_module.tiny_config

    def exportable_config():
        config = original_config()
        config.hf_model_type = 'qwen4_exp'
        config.swiglu = True
        return config

    monkeypatch.setattr(fixture_module, 'tiny_config', exportable_config)

    def bind_and_observe(optimizer):
        bind(optimizer)
        leaves = getattr(optimizer, 'chained_optimizers', [])
        if not leaves or not all(leaf.config.offload_model_from_cpu_master for leaf in leaves):
            return
        original = optimizer.offload_states
        # Dense/expert optimizers refer to the same DDP-wrapped model.
        model = leaves[0].model_chunks[0].module

        def parked_export(_self, *args, **kwargs):
            expected = {name: value.detach().cpu().clone()
                        for name, value in model.named_parameters()}
            expected_hf = {}
            for bucket in gather_all_hf_weights([model], 1 << 20, None):
                expected_hf.update({name: value.detach().cpu().clone() for name, value in bucket})
            result = original(*args, **kwargs)
            source = cpu_master_params.CpuMasterWeightProvider([model], optimizer)
            for name, value in model.named_parameters():
                assert value.device.type == 'cpu'
                torch.testing.assert_close(source(name, value).cpu(), expected[name], atol=0, rtol=0)
            observed_names = set()
            for bucket in gather_all_hf_weights([model], 1 << 20, None, weight_provider=source):
                for name, value in bucket:
                    torch.testing.assert_close(value.cpu(), expected_hf[name], atol=0, rtol=0, msg=name)
                    observed_names.add(name)
            assert observed_names == expected_hf.keys()
            assert all(p.device.type == 'cpu' for p in model.parameters())
            observations.append(len(expected_hf))
            return result

        optimizer.offload_states = types.MethodType(parked_export, optimizer)

    monkeypatch.setattr(offload, 'bind_megatron_offload_states_func', bind_and_observe)
    # The existing real-optimizer scenario checks three uninterrupted Adam
    # updates, dense DP4 / expert DP2, and authoritative master identity.
    test_distributed_hybrid_phase_matches_uninterrupted_trajectory(
        hybrid_parallel, monkeypatch, bounded=True, from_master=True)
    assert len(observations) == 3 and all(observations)


from tests.third_party.megatron.test_hybrid_phase_alias import hybrid_parallel
