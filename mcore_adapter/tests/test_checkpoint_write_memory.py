"""Exercise real DCP tensor lifetime, round trips, and partial-write failures."""
import importlib.util
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as filesystem
import torch.multiprocessing as mp


def implementation():
    path = Path(__file__).parents[1] / 'src/mcore_adapter/checkpoint_write.py'
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location('checkpoint_write', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def writer(path):
    module = implementation()
    if module is None:
        # RED: even the native serial writer retains previously written tensors
        # in its safetensors bookkeeping when serialization uses torch.save.
        return filesystem.FileSystemWriter(path, thread_count=1, per_thread_copy_ahead=0)
    return module.StreamingFileSystemWriter(path)


def states(device):
    model = torch.arange(16 * 128, dtype=torch.float32, device=device).to(torch.bfloat16)
    adam = torch.arange(24 * 128, dtype=torch.float32, device=device) / 13
    result = {}
    for index in range(12):
        # Views emulate packed model/Adam buffers. Serialization must compact
        # each slice without cloning and retaining the whole checkpoint.
        result[f'model_{index}'] = model[index * 128:(index + 1) * 128].reshape(8, 16).T
        result[f'adam_{index}'] = adam[index * 128:(index + 1) * 128].reshape(8, 16)
    result['empty'] = adam[:0]
    result['scalar'] = torch.tensor(19, dtype=torch.int64, device=device)
    result['step'] = {'num_steps': 20, 'nested': [True, 'state', None]}
    return result


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a bounded CUDA writer check'))])
def test_written_tensors_are_released_before_next_item_and_native_reload_is_exact(tmp_path, monkeypatch, device):
    source = states(device)
    retained = []
    peaks = []
    original = filesystem._write_item

    def observe(*args, **kwargs):
        tensor = args[2]
        if isinstance(tensor, torch.Tensor):
            retained.append(weakref.ref(tensor))
            peaks.append(sum(ref() is not None for ref in retained))
            assert tensor.device.type == 'cpu'
            assert tensor.untyped_storage().nbytes() == tensor.nbytes
        return original(*args, **kwargs)

    monkeypatch.setattr(filesystem, '_write_item', observe)
    dcp.save(source, storage_writer=writer(tmp_path / 'checkpoint'), no_dist=True)
    assert max(peaks) == 1, 'serialized checkpoint tensors remain live across write items'
    assert not any(ref() is not None for ref in retained), 'writer retained serialized payloads'
    target = {name: torch.empty_like(value, device='cpu') if isinstance(value, torch.Tensor) else value
              for name, value in source.items()}
    dcp.load(target, checkpoint_id=tmp_path / 'checkpoint', no_dist=True)
    for name, value in source.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(target[name], value.cpu())
            assert target[name].dtype == value.dtype and target[name].shape == value.shape
        else:
            assert target[name] == value


def test_native_partial_write_failure_never_commits_metadata(tmp_path, monkeypatch):
    calls = 0
    original = filesystem._write_item

    def fail_second_item(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('injected bounded checkpoint IO failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(filesystem, '_write_item', fail_second_item)
    with pytest.raises(BaseException, match='injected bounded checkpoint IO failure'):
        dcp.save(states('cpu'), storage_writer=writer(tmp_path), no_dist=True)
    assert not (tmp_path / '.metadata').exists()


def test_native_mcore_streaming_model_and_optimizer_roundtrip(tmp_path):
    pytest.importorskip('megatron.core')
    import torch.distributed as dist
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper

    module = implementation()
    assert module is not None, 'native MCore synchronous save needs a streaming strategy'
    dist.init_process_group('gloo', init_method=f'file://{tmp_path}/rendezvous', rank=0, world_size=1)
    try:
        for role, dtype in [('model', torch.bfloat16), ('optimizer', torch.float32)]:
            save_strategy = FullyParallelSaveStrategyWrapper(
                module.streaming_save_strategy(), dist.group.WORLD, do_cache_distribution=True)
            original = torch.arange(240, dtype=dtype).reshape(12, 20)
            state = {str(i): ShardedTensor.from_rank_offsets(f'{role}.{i}', original[i]) for i in range(12)}
            state['object'] = ShardedObject(f'{role}.extra', {'step': 20}, (1,), (0,))
            directory = tmp_path / role
            directory.mkdir()
            dist_checkpointing.save(state, directory, sharded_strategy=save_strategy, async_sharded_save=False)
            target = {str(i): ShardedTensor.from_rank_offsets(f'{role}.{i}', torch.empty_like(original[i]))
                      for i in range(12)}
            target['object'] = ShardedObject(f'{role}.extra', None, (1,), (0,))
            restored = dist_checkpointing.load(target, directory)
            assert all(torch.equal(restored[str(i)], original[i]) for i in range(12))
            assert restored['object'] == {'step': 20}
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def _two_rank_worker(rank, root, fail_rank):
    import torch.distributed as dist
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper

    root = Path(root)
    dist.init_process_group('gloo', init_method=f'file://{root}/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        original_write = filesystem._write_item
        if rank == fail_rank:
            def fail(*args, **kwargs):
                raise OSError('injected nonzero-rank streaming failure')
            filesystem._write_item = fail
        for role, dtype in [('model', torch.bfloat16), ('optimizer', torch.float32)]:
            # Production gives model and optimizer separate distribution caches.
            strategy = FullyParallelSaveStrategyWrapper(
                implementation().streaming_save_strategy(), dist.group.WORLD, do_cache_distribution=True)
            directory = root / role
            directory.mkdir(exist_ok=True)
            full = torch.arange(48, dtype=dtype).reshape(8, 6)
            local = full[rank * 4:(rank + 1) * 4]
            replicated = torch.arange(7, dtype=torch.float32)
            state = {
                'sharded': ShardedTensor.from_rank_offsets(f'{role}.sharded', local, (0, rank, 2)),
                'replicated': ShardedTensor.from_rank_offsets(f'{role}.replicated', replicated, replica_id=rank),
                'object': ShardedObject(f'{role}.extra', {'rank': rank, 'step': 20}, (2,), (rank,)),
            }
            rejected = False
            try:
                dist_checkpointing.save(state, directory, sharded_strategy=strategy, async_sharded_save=False)
            except BaseException as error:
                if fail_rank is None:
                    raise
                assert 'injected nonzero-rank streaming failure' in str(error)
                rejected = True
            if fail_rank is not None:
                failures = [None, None]
                dist.all_gather_object(failures, rejected)
                assert all(failures), 'every rank must reject a failed shard write'
                assert not (directory / '.metadata').exists()
                break
            target = {
                'sharded': ShardedTensor.from_rank_offsets(f'{role}.sharded', torch.empty_like(local), (0, rank, 2)),
                'replicated': ShardedTensor.from_rank_offsets(f'{role}.replicated', torch.empty_like(replicated), replica_id=rank),
                'object': ShardedObject(f'{role}.extra', None, (2,), (rank,)),
            }
            loaded = dist_checkpointing.load(target, directory)
            assert torch.equal(loaded['sharded'], local)
            assert torch.equal(loaded['replicated'], replicated)
            assert loaded['object'] == {'rank': rank, 'step': 20}
        filesystem._write_item = original_write
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('fail_rank', [None, 1])
def test_native_mcore_two_rank_shards_replicas_and_collective_failure(tmp_path, fail_rank):
    pytest.importorskip('megatron.core')
    mp.spawn(_two_rank_worker, args=(str(tmp_path), fail_rank), nprocs=2, join=True)


def test_native_hybrid_adam_shard_construction_borrows_payloads_without_cloning(monkeypatch):
    pytest.importorskip('megatron.core')
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    intervals = [(0, 11), (16, 29), (32, 55)]
    originals = [torch.nn.Parameter(torch.arange(end - start, dtype=torch.bfloat16))
                 for start, end in intervals]
    masters = [parameter.detach().float() for parameter in originals]
    hybrid = HybridDeviceOptimizer.__new__(HybridDeviceOptimizer)
    torch.optim.Optimizer.__init__(hybrid, originals, {})
    hybrid.param_update_in_fp32 = True
    adam = torch.optim.AdamW(masters, lr=.01, fused=True)
    for parameter in masters:
        parameter.grad = torch.ones_like(parameter)
    adam.step()
    adam.zero_grad(set_to_none=True)
    hybrid.cpu_optimizers = [adam]
    hybrid.gpu_optimizer = None
    hybrid.inner_param_to_orig_param = dict(zip(masters, originals))
    hybrid._sync_sub_optimizers_state_to_hdo()

    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.optimizer = hybrid
    optimizer.model_param_group_index_map = {parameter: (0, i) for i, parameter in enumerate(originals)}
    optimizer.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True)
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.grad_scaler = None
    optimizer.data_parallel_group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    optimizer.data_parallel_group_idx = 0
    optimizer.distributed_optimizer_instance_id = 0
    dtype = (torch.bfloat16, torch.float32)
    optimizer.gbuf_ranges = [{dtype: [{'param_map': {
        parameter: {'gbuf_local': SimpleNamespace(start=start, end=end)}
        for parameter, (start, end) in zip(originals, intervals)}}]}]
    optimizer.per_bucket_numel = [[64]]
    optimizer.per_bucket_numel_unpadded = [[64]]
    optimizer.buffers = [SimpleNamespace(buckets=[SimpleNamespace(numel_unpadded=64, grad_data=torch.zeros(64))])]

    def forbidden_clone(*args, **kwargs):
        raise AssertionError('native optimizer sharding cloned a resident CPU payload')

    monkeypatch.setattr(torch.Tensor, 'clone', forbidden_clone)
    saved = optimizer.sharded_state_dict({}, metadata={'distrib_optim_sharding_type': 'dp_reshardable'})
    entries = saved['param_state'][0][dtype][0]
    payloads = [entry for entry in entries if not entry['padding'].unwrap()]
    assert len(payloads) == len(originals)
    for entry, original, master in zip(payloads, originals, masters):
        assert entry['param'].data is master
        assert entry['exp_avg'].data is adam.state[master]['exp_avg']
        assert entry['exp_avg_sq'].data is adam.state[master]['exp_avg_sq']
    for entry in entries:
        if entry['padding'].unwrap():
            assert all(torch.count_nonzero(entry[key].data) == 0 for key in ('param', 'exp_avg', 'exp_avg_sq'))
    assert not torch.cuda.is_initialized()
