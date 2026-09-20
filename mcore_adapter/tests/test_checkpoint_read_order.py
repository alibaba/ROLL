"""Read real DCP payloads in physical order without changing restored values."""
import dataclasses
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, save
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


def implementation():
    path = Path(__file__).parents[1] / 'src/mcore_adapter/checkpoint_read_order.py'
    assert path.is_file(), 'MCore checkpoint reader must avoid backwards payload seeks'
    spec = importlib.util.spec_from_file_location('checkpoint_read_order', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ObservedReader(FileSystemReader):
    def __init__(self, path):
        super().__init__(path)
        self.payload_offsets = []

    def _slice_file(self, stream, item):
        self.payload_offsets.append((item.relative_path, item.offset))
        return super()._slice_file(stream, item)


def checkpoint(tmp_path, threads=1):
    source = {'weight': torch.arange(15, dtype=torch.bfloat16).reshape(3, 5),
              'exp_avg': torch.arange(24, dtype=torch.float32).reshape(6, 4) / 7,
              'exp_avg_sq': torch.ones(11, dtype=torch.float32),
              'step': {'num_steps': 19, 'labels': ['en', 'zh', 'code']}}
    save(source, storage_writer=FileSystemWriter(tmp_path, thread_count=threads), no_dist=True)
    target = {k: torch.empty_like(v) for k, v in source.items() if isinstance(v, torch.Tensor)}
    target['step'] = {'num_steps': -1, 'labels': ['placeholder']}
    reader = ObservedReader(tmp_path)
    metadata = reader.read_metadata()
    reader.set_up_storage_reader(metadata, is_coordinator=True)
    planner = DefaultLoadPlanner()
    planner.set_up_planner(target, metadata, is_coordinator=True)
    plan = planner.create_local_plan()
    # Actual state-dict traversal can differ from physical writer order. Force
    # the worst order so the test detects a missing read-order optimization.
    plan = dataclasses.replace(plan, items=sorted(plan.items,
        key=lambda req: (reader.storage_data[req.storage_index].relative_path,
                         reader.storage_data[req.storage_index].offset), reverse=True))
    plan = reader.prepare_local_plan(plan)
    planner.finish_plan(plan)
    return source, target, reader, planner, plan


@pytest.mark.parametrize('threads', [1, 2])
def test_reads_forward_and_preserves_tensors_objects_and_original_plan(tmp_path, threads):
    source, target, reader, planner, plan = checkpoint(tmp_path, threads)
    items = list(plan.items)
    module = implementation()
    module.order_checkpoint_reader(reader)
    module.order_checkpoint_reader(reader)  # Repeated strategy initialization.
    reader.read_data(plan, planner).wait()
    assert reader.payload_offsets == sorted(reader.payload_offsets)
    assert len(reader.payload_offsets) == len(items)
    assert plan.items == items
    for key, value in source.items():
        if isinstance(value, torch.Tensor):
            assert target[key].dtype == value.dtype
            assert torch.equal(target[key], value)
        else:
            assert target[key] == value


def test_missing_payload_propagates_native_read_error(tmp_path):
    _, _, reader, planner, plan = checkpoint(tmp_path)
    implementation().order_checkpoint_reader(reader)
    next(tmp_path.glob('*.distcp')).unlink()
    with pytest.raises(FileNotFoundError):
        reader.read_data(plan, planner).wait()


def test_reordered_subtensor_requests_keep_destination_offsets(tmp_path):
    source, target, reader, planner, plan = checkpoint(tmp_path)
    request = next(item for item in plan.items if item.dest_index.fqn == 'weight')
    # Two disjoint destination rows both read slices of the same stored tensor.
    parts = [dataclasses.replace(request, storage_offsets=torch.Size([start, 0]),
             dest_offsets=torch.Size([start, 0]), lengths=torch.Size([size, 5]))
             for start, size in [(1, 2), (0, 1)]]
    plan = dataclasses.replace(plan, items=parts + [item for item in plan.items if item is not request])
    implementation().order_checkpoint_reader(reader)
    reader.read_data(plan, planner).wait()
    for key in ('weight', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(target[key], source[key])
    assert target['step'] == source['step']


def test_actual_mcore_cached_reader_loads_dcp_tensors(tmp_path, monkeypatch):
    pytest.importorskip('megatron.core')
    import torch.distributed as dist
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.dist_checkpointing.strategies import torch as strategy

    original = strategy._get_filesystem_reader
    # Let monkeypatch restore MCore's factory after exercising the actual patch.
    monkeypatch.setattr(strategy, '_get_filesystem_reader', original)
    implementation().patch_mcore_checkpoint_read_order()
    implementation().patch_mcore_checkpoint_read_order()
    reader = strategy._get_filesystem_reader(tmp_path, cache_metadata=True)
    assert isinstance(reader, strategy.CachedMetadataFileSystemReader)
    assert reader.read_data._mcore_storage_ordered is True
    dist.init_process_group('gloo', init_method=f'file://{tmp_path}/rendezvous', rank=0, world_size=1)
    try:
        model = torch.arange(30, dtype=torch.bfloat16).reshape(6, 5)
        adam = torch.arange(30, dtype=torch.float32).reshape(6, 5) / 11
        directory = tmp_path / 'native'
        directory.mkdir()
        # The reader consumes standard DCP files. MCore's asynchronous writer
        # unconditionally synchronizes CUDA, so build this CPU fixture with the
        # real PyTorch writer and exercise the unchanged MCore load strategy.
        save({'model': model, 'adam': adam}, checkpoint_id=directory, no_dist=True)
        requested = {'adam': ShardedTensor.from_rank_offsets('adam', torch.empty_like(adam)),
                     'model': ShardedTensor.from_rank_offsets('model', torch.empty_like(model))}
        restored = strategy.TorchDistLoadShardedStrategy().load(requested, directory)
        assert torch.equal(restored['model'], model)
        assert torch.equal(restored['adam'], adam)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()
