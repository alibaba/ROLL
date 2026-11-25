import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from roll.pipeline import base_pipeline
from roll.pipeline.rlvr import rlvr_pipeline


def driver_without_training(tracker, *, backend="rl_insight"):
    driver = rlvr_pipeline.RLVRPipeline.__new__(rlvr_pipeline.RLVRPipeline)
    driver.tracker = tracker
    driver.pipeline_config = SimpleNamespace(track_with=backend, async_pipeline=False, max_steps=0)
    driver.resume_futures = []
    driver.generate_schedulers = {"train": Mock()}
    driver.val_dataset = None
    return driver


def test_run_joins_resume_even_when_no_checkpoint(monkeypatch):
    tracker = Mock()
    driver = driver_without_training(tracker)
    shutdown_called = threading.Event()
    allow_replay_to_finish = threading.Event()
    order = []

    def shutdown(refs):
        order.append("scheduler shutdown")
        shutdown_called.set()

    def replay():
        assert allow_replay_to_finish.wait(timeout=5)
        order.append("replay")
        tracker.log({"loss": 1}, step=2)

    tracker.finish.side_effect = lambda: order.append("tracker finish")
    monkeypatch.setattr(rlvr_pipeline.ray, "get", shutdown)
    with ThreadPoolExecutor(max_workers=2) as executor:
        driver.resume_futures = [executor.submit(replay)]
        run = executor.submit(driver.run)
        try:
            assert shutdown_called.wait(timeout=5)
        finally:
            allow_replay_to_finish.set()
        run.result(timeout=5)

    assert order == ["scheduler shutdown", "replay", "tracker finish"]
    tracker.finish.assert_called_once()


@pytest.mark.parametrize("phase", ["training", "scheduler"])
def test_run_preserves_primary_failure(monkeypatch, phase):
    tracker = Mock()
    tracker.finish.side_effect = RuntimeError("monitor cleanup")
    report = Mock()
    monkeypatch.setattr(rlvr_pipeline.logger, "exception", report)
    driver = driver_without_training(tracker)
    error = KeyboardInterrupt() if phase == "training" else RuntimeError("scheduler cleanup")
    if phase == "training":
        monkeypatch.setattr(rlvr_pipeline, "MetricsManager", Mock(side_effect=error))
    else:
        monkeypatch.setattr(rlvr_pipeline.ray, "get", Mock(side_effect=error))
    replay = Future()
    replay.set_exception(RuntimeError("broken history"))
    driver.resume_futures = [replay]

    with pytest.raises(type(error)) as raised:
        driver.run()
    assert raised.value is error
    tracker.finish.assert_called_once()
    assert report.call_count == 2


@pytest.mark.parametrize("phase", ["base", "rlvr"])
def test_constructor_failure_closes_created_tracker(monkeypatch, phase):
    tracker = Mock()
    error = RuntimeError("initialization failed")
    monkeypatch.setattr(base_pipeline, "ResourceManager", Mock())
    monkeypatch.setattr(base_pipeline, "create_tracker", Mock(return_value=tracker))
    monkeypatch.setattr(base_pipeline, "init_transfer_backend", Mock(side_effect=error if phase == "base" else None))
    monkeypatch.setattr(rlvr_pipeline, "default_tokenizer_provider", Mock(side_effect=error))
    config = SimpleNamespace(
        seed=1,
        num_nodes=1,
        num_gpus_per_node=0,
        checkpoint_config={},
        to_dict=dict,
        track_with="rl_insight",
        tracker_kwargs={},
        transfer_backend=None,
        resume_from_checkpoint=False,
        auto_resume=False,
        enable_reference=False,
        actor_train=SimpleNamespace(model_args=None),
    )

    with pytest.raises(RuntimeError) as raised:
        rlvr_pipeline.RLVRPipeline(config)
    assert raised.value is error
    tracker.finish.assert_called_once()


def test_rlvr_leaves_other_tracker_lifecycles_unchanged(monkeypatch):
    tracker = Mock()
    driver = driver_without_training(tracker, backend="tensorboard")
    monkeypatch.setattr(rlvr_pipeline.ray, "get", Mock())
    driver.run()
    tracker.finish.assert_not_called()
