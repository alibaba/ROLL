"""Cold RL resume restores driver RNG before the first external actor phase."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
from roll.utils import worker_state
from roll.utils.worker_state import WorkerState


class AtActorBoundary(Exception):
    pass


@pytest.fixture(autouse=True)
def cpu_driver_platform(monkeypatch):
    # This test owns no CUDA devices; preserve real Python/NumPy/Torch RNG IO.
    monkeypatch.setattr(worker_state, 'current_platform', SimpleNamespace(
        device_type='cuda', random=SimpleNamespace(
            get_rng_state_all=lambda: [], set_rng_state_all=lambda states: None)))


def pipeline(resume):
    obj=object.__new__(RLVRPipeline)
    obj.pipeline_config=SimpleNamespace(async_pipeline=False,max_steps=1,adv_estimator='grpo',reward_system_config=None)
    obj.resume_from_checkpoint=str(resume) if resume else False
    obj.state=WorkerState()
    def stop(**kwargs):
        raise AtActorBoundary()
    obj.actor_train=SimpleNamespace(offload_states=stop)
    return obj


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def draw():
    return (random.random(),np.random.random(),torch.rand(()).item())


def test_rlvr_restores_saved_driver_rng_before_first_actor_phase(tmp_path):
    checkpoint=tmp_path/'checkpoint-10'
    seed(42)
    for _ in range(11):draw()
    WorkerState.save_rng_state(str(checkpoint/'pipeline'),'pipeline')
    expected=draw()
    seed(997)
    with pytest.raises(AtActorBoundary):pipeline(checkpoint).run()
    assert draw()==expected


def test_rlvr_missing_driver_rng_rejected_before_actor_phase(tmp_path):
    checkpoint=tmp_path/'checkpoint-10';checkpoint.mkdir()
    with pytest.raises(FileNotFoundError,match='pipeline RNG'):
        pipeline(checkpoint).run()


def test_rlvr_fresh_run_keeps_current_driver_rng():
    seed(997);expected=draw()
    seed(997)
    with pytest.raises(AtActorBoundary):pipeline(None).run()
    assert draw()==expected
