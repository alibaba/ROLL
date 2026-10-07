"""Runtime exits release only explicitly owned local Ray backends."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def pipeline(monkeypatch):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    events = []
    runtime = module.TinkerBackendRuntimePipeline.__new__(module.TinkerBackendRuntimePipeline)
    runtime.args = SimpleNamespace(backend_config={"ray_address": "local"},
                                  preload_generator=False, exit_after_ready=False, runtime_id="rt_cleanup_test")
    runtime.generator = None
    runtime.roll_backend = None
    runtime.should_stop = False
    backend = SimpleNamespace(close=lambda: events.append("backend_close"))
    runtime._init_roll_backend = lambda: setattr(runtime, "roll_backend", backend)
    runtime.report_ready_with_retries = lambda: events.append("ready")
    runtime.report_heartbeat = lambda **kwargs: events.append(("heartbeat", kwargs["status"]))
    runtime.post_action_result = lambda *args, **kwargs: events.append("action_result")
    runtime.action_executor = SimpleNamespace(shutdown=lambda **kwargs: events.append("executor_shutdown"))
    monkeypatch.setattr(module, "log_event", lambda *args, **kwargs: None)
    return runtime, events


def test_local_runtime_ready_failure_releases_backend(pipeline):
    runtime, events = pipeline

    def ready_failure():
        raise RuntimeError("ready heartbeat unavailable")

    runtime.report_ready_with_retries = ready_failure
    with pytest.raises(RuntimeError, match="ready heartbeat unavailable"):
        runtime.run()
    assert events == ["backend_close", "executor_shutdown"]


def test_local_runtime_poll_failure_releases_backend(pipeline):
    runtime, events = pipeline

    async def poll_failure():
        raise RuntimeError("poll failed")

    runtime._poll_actions = poll_failure
    with pytest.raises(RuntimeError, match="poll failed"):
        runtime.run()
    assert events == ["ready", "backend_close", "executor_shutdown"]


def test_local_runtime_exit_after_ready_releases_backend(pipeline):
    runtime, events = pipeline
    runtime.args.exit_after_ready = True
    assert runtime.run() == 0
    assert events == ["ready", "backend_close", "executor_shutdown"]


def test_normal_close_and_finally_close_backend_exactly_once(pipeline):
    runtime, events = pipeline

    async def poll_close():
        runtime.handle_close_runtime({"action_id": 1})

    runtime._poll_actions = poll_close
    assert runtime.run() == 0
    assert events.count("backend_close") == 1
    assert runtime.should_stop
    assert events[-1] == "executor_shutdown"


def test_default_cluster_error_keeps_existing_cleanup_semantics(pipeline):
    runtime, events = pipeline
    runtime.args.backend_config = {"config_name": "default"}

    async def poll_failure():
        raise RuntimeError("default poll failed")

    runtime._poll_actions = poll_failure
    with pytest.raises(RuntimeError, match="default poll failed"):
        runtime.run()
    assert events == ["ready", "executor_shutdown"]


def test_cleanup_failure_still_shuts_executor_and_fails(pipeline):
    runtime, events = pipeline
    runtime.args.exit_after_ready = True

    def close_failure():
        events.append("backend_close")
        raise RuntimeError("backend cleanup failed")

    runtime._init_roll_backend = lambda: setattr(runtime, "roll_backend", SimpleNamespace(close=close_failure))
    with pytest.raises(RuntimeError, match="backend cleanup failed"):
        runtime.run()
    assert events == ["ready", "backend_close", "executor_shutdown"]
