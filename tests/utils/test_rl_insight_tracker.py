import os
import subprocess
import sys
from textwrap import dedent
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import ray
import torch

from roll.utils.tracking import create_tracker


@pytest.fixture
def monitor(monkeypatch):
    sdk = ModuleType("rl_insight")
    sdk.init = Mock()
    sdk.metric_log = Mock()
    sdk.get_context = Mock(
        side_effect=[None, SimpleNamespace(otel_traces_enabled=True, experiment_name="execution", project="roll")]
    )
    sdk.finish = Mock()
    monkeypatch.setitem(sys.modules, "rl_insight", sdk)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", "http://127.0.0.1:18080")
    monkeypatch.delenv("RL_INSIGHT_SERVER_BACKEND", raising=False)
    return sdk


def test_tensorboard_does_not_import_rl_insight(tmp_path):
    script = dedent("""\
        import builtins
        import sys

        import_module = builtins.__import__

        def reject_sdk(name, *args, **kwargs):
            assert name.split(".")[0] != "rl_insight", "TensorBoard must not import RL-Insight"
            return import_module(name, *args, **kwargs)

        builtins.__import__ = reject_sdk
        from roll.utils.tracking import create_tracker

        writer = create_tracker("tensorboard", {}, log_dir=sys.argv[1])
        writer.log({"loss": 1.0}, step=0)
        writer.finish()
        assert "rl_insight" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "ROLL_LOG_DIR": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.glob("events.out.tfevents.*"))


def test_selected_tracker_requires_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "rl_insight", None)
    with pytest.raises(ImportError, match="install.*RL-Insight"):
        create_tracker("rl_insight", {})


def test_selected_tracker_requires_metric_log(monitor):
    del monitor.metric_log
    with pytest.raises(ImportError, match="metric_log"):
        create_tracker("rl_insight", {})
    monitor.init.assert_not_called()


@pytest.mark.parametrize("url", [None, "", "localhost:18080", "http://localhost:xyz", "http://u:p@localhost"])
def test_selected_tracker_requires_server_url(monkeypatch, monitor, url):
    if url is None:
        monkeypatch.delenv("RL_INSIGHT_SERVER_URL")
    else:
        monkeypatch.setenv("RL_INSIGHT_SERVER_URL", url)
    with pytest.raises(ValueError, match="RL_INSIGHT_SERVER_URL"):
        create_tracker("rl_insight", {})
    monitor.init.assert_not_called()


def test_does_not_start_ray(monkeypatch, monitor):
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    ray_init = Mock()
    monkeypatch.setattr(ray, "init", ray_init)
    with pytest.raises(RuntimeError, match="Ray.*initialized"):
        create_tracker("rl_insight", {})
    ray_init.assert_not_called()
    monitor.init.assert_not_called()


def test_rejects_non_ray_backend_override(monkeypatch, monitor):
    monkeypatch.setenv("RL_INSIGHT_SERVER_BACKEND", "push")
    with pytest.raises(ValueError, match="RL_INSIGHT_SERVER_BACKEND"):
        create_tracker("rl_insight", {})
    monitor.init.assert_not_called()


def test_does_not_attach_to_existing_run(monitor):
    monitor.get_context.side_effect = [SimpleNamespace(otel_traces_enabled=True)]
    with pytest.raises(RuntimeError, match="already.*RL-Insight|RL-Insight.*already"):
        create_tracker("rl_insight", {})
    monitor.init.assert_not_called()
    monitor.finish.assert_not_called()


def test_preserves_steps_and_filters_values(monitor):
    tracker = create_tracker(
        "rl_insight", {"secret": "do not export"}, project="roll", name="experiment", tags=["unused"]
    )
    monitor.init.assert_called_once_with(
        project="roll",
        experiment_name="experiment",
        config={"server": {"backend": "ray"}, "hardware": {"enable": False}},
    )
    metrics = {
        "actor/loss@mean": 1.25,
        "reward": np.float32(0.5),
        "token/count": torch.tensor(12),
        "array/scalar": np.array(3),
        "system/step": 99,
        "nested": {"loss@mean": 2.0},
        "text": "12",
        "bool": True,
        "nan": float("nan"),
        "inf": np.float32("inf"),
        "overflow": 10**400,
        "vector": torch.tensor([1, 2]),
        "array": np.array([1, 2]),
        "non_cpu": torch.empty((), device="meta"),
        "complex": torch.tensor(1j),
    }
    tracker.log(metrics, step=0)
    tracker.log({"actor/loss@sum": 2}, step=np.int64(7))
    for step in (None, 2.5, "7", True, -1, 2**63):
        tracker.log({"not_sent": 1.0}, step=step)

    assert [(call.args, call.kwargs["step"]) for call in monitor.metric_log.call_args_list] == [
        (("actor/loss", 1.25), 0),
        (("reward", 0.5), 0),
        (("token/count", 12.0), 0),
        (("array/scalar", 3.0), 0),
        (("system/step", 99.0), 0),
        (("actor/loss", 2.0), 7),
    ]
    assert "actor/loss@mean" in metrics
    assert metrics["nested"] == {"loss@mean": 2.0}

    tracker.finish()
    tracker.finish()
    monitor.finish.assert_called_once_with()
    tracker.log({"closed": 1.0}, step=8)
    assert monitor.metric_log.call_count == 6


@pytest.mark.parametrize("context", [None, SimpleNamespace(otel_traces_enabled=False)])
def test_unavailable_tempo_warns_and_does_not_send(monkeypatch, monitor, context):
    from roll.utils import tracking

    warning = Mock()
    monkeypatch.setattr(tracking.logger, "warning", warning)
    monitor.get_context.side_effect = [None, context]
    tracker = create_tracker("rl_insight", {})
    warning.assert_called_once()
    tracker.log({"loss": 1}, 0)
    monitor.metric_log.assert_not_called()
    tracker.finish()
    monitor.finish.assert_called_once_with()


def test_setup_exception_releases_session(monitor):
    monitor.get_context.side_effect = [None, RuntimeError("failed after init")]
    with pytest.raises(RuntimeError, match="failed after init"):
        create_tracker("rl_insight", {})
    monitor.finish.assert_called_once_with()
