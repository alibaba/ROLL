import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from roll.utils.tracking import BaseTracker, TrackioTracker


def test_base_tracker_log_traces_is_noop():
    BaseTracker().log_traces("rollout/test", [{"messages": []}], step=1)


def test_trackio_tracker_logs_trace_records(monkeypatch):
    run = MagicMock()
    trace = MagicMock(return_value="trace-payload")
    trackio = SimpleNamespace(init=MagicMock(return_value=run), Trace=trace)
    monkeypatch.setitem(sys.modules, "trackio", trackio)

    tracker = TrackioTracker(config={"model": "tiny"}, project="roll", name="trace-smoke")
    records = [
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
            ],
            "metadata": {"step": 3, "sample_index": 0},
        }
    ]

    tracker.log_traces("rollout/rlvr", records, step=3)

    trace.assert_called_once_with(messages=records[0]["messages"], metadata=records[0]["metadata"])
    run.log.assert_called_once_with({"rollout/rlvr": ["trace-payload"]}, step=3)
