"""Exercise SFT's real DataLoader loop and saved driver RNG across cold resume."""
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.sft.sft_pipeline import SFTPipeline
from roll.utils import worker_state
from roll.utils.worker_state import WorkerState


class _RandomDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        # Actual single-process DataLoader transforms exercise all three RNGs.
        values = [index, int(random.random() * 100000), int(np.random.random() * 100000),
                  int(torch.rand(()).item() * 100000)]
        return {'input_ids': torch.tensor(values), 'attention_mask': torch.ones(4, dtype=torch.long)}


def _pipeline(directory, resume=None, pipeline_max_steps=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    pipeline = object.__new__(SFTPipeline)
    pipeline.pipeline_config = SimpleNamespace(
        sft_train=SimpleNamespace(training_args=SimpleNamespace(num_train_epochs=2)),
        eval_steps=2,
    )
    pipeline.resume_from_checkpoint = str(resume) if resume else False
    pipeline._pipeline_max_steps = pipeline_max_steps
    pipeline.state = (WorkerState.load_from_json(str(resume / 'pipeline'), 'pipeline')
                      if resume else WorkerState())
    pipeline.dataloader = DataLoader(_RandomDataset(), batch_size=1, shuffle=False, num_workers=0)
    pipeline.global_train_batch_size = 1
    pipeline.val_dataset = [1]
    pipeline.batches = {}
    pipeline.tracker = SimpleNamespace(log=lambda **kwargs: None)

    def train_step(batch, blocking):
        step = batch.meta_info['global_step']
        pipeline.batches[step] = batch.batch['input_ids'].tolist()
        return DataProto(meta_info={'metrics': {'sft_train/loss@sum': 1.0}})

    pipeline.sft_train = SimpleNamespace(dp_size=1, train_step=train_step)

    def val():
        # Validation's native DataLoader iterator consumes an independent CPU
        # base seed even for deterministic inputs and num_workers=0.
        list(DataLoader([torch.zeros(1)], batch_size=1, num_workers=0))
        return {'sft_train/val_loss': [0.5]}

    pipeline.val = val

    def checkpoint(global_step):
        target = directory / f'checkpoint-{global_step}' / 'pipeline'
        pipeline.state.save_to_json(str(target), 'pipeline')
        pipeline.state.save_rng_state(str(target), 'pipeline')

    pipeline.do_checkpoint = checkpoint
    return pipeline


@pytest.mark.parametrize('saved_step', [1, 3, 4])
def test_sft_resume_preserves_next_batches_and_driver_rng(tmp_path, monkeypatch, saved_step):
    # Worker/Ray execution is irrelevant to driver RNG. Keep actual DataProto,
    # SFT run, DataLoader, RNG serializers and batch-balancing code.
    monkeypatch.setattr(DataProto, 'materialize_concat', staticmethod(lambda data_refs: data_refs))
    monkeypatch.setattr(worker_state, 'current_platform', SimpleNamespace(
        device_type='cuda', random=SimpleNamespace(get_rng_state_all=lambda: [], set_rng_state_all=lambda value: None)))
    continuous = _pipeline(tmp_path / 'continuous')
    continuous.run()
    resume = tmp_path / 'continuous' / f'checkpoint-{saved_step}'
    recovered = _pipeline(tmp_path / 'recovered', resume)
    recovered.run()
    expected = {step: batch for step, batch in continuous.batches.items() if step > saved_step}
    assert recovered.batches == expected
    left = torch.load(tmp_path / 'continuous/checkpoint-7/pipeline/rng_state_pipeline.pth', weights_only=False)
    right = torch.load(tmp_path / 'recovered/checkpoint-7/pipeline/rng_state_pipeline.pth', weights_only=False)
    assert left['python'] == right['python']
    assert left['numpy'][0] == right['numpy'][0]
    np.testing.assert_array_equal(left['numpy'][1], right['numpy'][1])
    assert left['numpy'][2:] == right['numpy'][2:]
    torch.testing.assert_close(left['cpu'], right['cpu'], rtol=0, atol=0)


def test_sft_resume_rejects_missing_driver_rng_before_new_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(DataProto, 'materialize_concat', staticmethod(lambda data_refs: data_refs))
    monkeypatch.setattr(worker_state, 'current_platform', SimpleNamespace(
        device_type='cuda', random=SimpleNamespace(get_rng_state_all=lambda: [], set_rng_state_all=lambda value: None)))
    continuous = _pipeline(tmp_path / 'continuous')
    continuous.run()
    checkpoint = tmp_path / 'continuous/checkpoint-3'
    (checkpoint / 'pipeline/rng_state_pipeline.pth').unlink()
    recovered = _pipeline(tmp_path / 'recovered', checkpoint)
    with pytest.raises(FileNotFoundError, match='pipeline RNG'):
        recovered.run()
    assert recovered.batches == {}


def test_sft_run_honors_explicit_pipeline_step_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(DataProto, 'materialize_concat', staticmethod(lambda data_refs: data_refs))
    monkeypatch.setattr(worker_state, 'current_platform', SimpleNamespace(
        device_type='cuda', random=SimpleNamespace(get_rng_state_all=lambda: [], set_rng_state_all=lambda value: None)))
    pipeline = _pipeline(tmp_path / 'capped', pipeline_max_steps=2)

    pipeline.run()

    assert list(pipeline.batches) == [0, 1]
    assert pipeline.state.step == 1


@pytest.mark.parametrize('step_limit,expected_epochs,expected_worker_steps', [
    (None, 2, 8), (2, 1, 4),
])
def test_sft_initialization_preserves_epochs_unless_cap_is_explicit(
        monkeypatch, step_limit, expected_epochs, expected_worker_steps):
    from roll.configs.base_config import BaseConfig
    from roll.pipeline.sft.sft_config import SFTConfig
    from roll.pipeline.sft import sft_pipeline as module

    # Isolate external I/O and Ray startup, retaining the real SFT config,
    # pipeline initialization, DP batch sizing and DataLoader.
    monkeypatch.setattr(BaseConfig, '__post_init__', lambda self: None)
    worker = SimpleNamespace(
        model_args=SimpleNamespace(model_name_or_path=None), worker_cls=None,
        training_args=SimpleNamespace(num_train_epochs=2,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            dataloader_num_workers=0, max_steps=-1),
        strategy_args=SimpleNamespace(strategy_name='megatron_train', strategy_config={}),
        world_size=2,
        data_args=SimpleNamespace(file_name='fixture.json', template='native',
                                  preprocessing_num_workers=1),
    )
    kwargs = {} if step_limit is None else {'max_steps': step_limit}
    config = SFTConfig(sft_train=worker, **kwargs)
    monkeypatch.setattr(module.BasePipeline, '__init__',
                        lambda self, config: setattr(self, 'resource_manager', None))
    monkeypatch.setattr(module, 'default_tokenizer_provider', lambda args: SimpleNamespace())
    monkeypatch.setattr(module.datasets, 'load_dataset', lambda *a, **kw: {'train': _RandomDataset()})
    monkeypatch.setattr(module, 'get_encode_function', lambda *a, **kw: None)
    monkeypatch.setattr(module, 'preprocess_dataset', lambda dataset, *a, **kw: dataset)
    monkeypatch.setattr(module, 'DataCollatorForSFT', lambda **kw: None)
    monkeypatch.setattr(module, 'Cluster', lambda **kw: SimpleNamespace(
        dp_size=2, initialize=lambda **kw: None))
    monkeypatch.setattr(module.ray, 'get', lambda value: value)
    monkeypatch.setattr(SFTPipeline, 'set_checkpoint_clusters', lambda *a: None)

    pipeline = SFTPipeline(config)

    assert len(pipeline.dataloader) == 2
    assert worker.training_args.num_train_epochs == expected_epochs
    # Workers divide this budget by DP=2; the driver keeps the pre-DP value.
    assert worker.training_args.max_steps == expected_worker_steps
    assert pipeline._pipeline_max_steps == step_limit
