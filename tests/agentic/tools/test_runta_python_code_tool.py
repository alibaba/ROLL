"""Tests for the Runta-backed python code tool.

The unit tests run against a fake ``runta`` module. ``test_live_execution`` talks to
the real service and is skipped unless ``RUNTA_TOKEN`` is set.
"""

import os
import sys
from types import SimpleNamespace

import pytest

from roll.pipeline.agentic.tools import runta_python_code_tool as tool_module
from roll.pipeline.agentic.tools.python_code_tool import PythonCodeTool
from roll.pipeline.agentic.tools.registration import make_tool
from roll.pipeline.agentic.tools.runta_python_code_tool import RuntaPythonCodeTool


class FakeApiError(Exception):
    pass


def result(exit_code=0, stdout="", stderr=""):
    return SimpleNamespace(exit_code=exit_code, stdout_text=stdout, stderr_text=stderr)


@pytest.fixture
def runta(monkeypatch):
    state = SimpleNamespace(creates=[], execs=[], outcomes=[], deletes=0, closes=0)

    class FakeRuntime:
        id = "runtime-1"

        def exec(self, argv, timeout=None, stdin=None):
            state.execs.append(SimpleNamespace(argv=argv, timeout=timeout, stdin=stdin))
            outcome = state.outcomes.pop(0) if state.outcomes else result()
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def delete(self):
            state.deletes += 1

    class Runta:
        def __init__(self, **kwargs):
            self.runtimes = SimpleNamespace(create=self._create)

        def _create(self, name, **kwargs):
            state.creates.append((name, kwargs))
            return FakeRuntime()

        def close(self):
            state.closes += 1

    monkeypatch.setitem(sys.modules, "runta", SimpleNamespace(Runta=Runta, ApiError=FakeApiError))
    return state


def test_tool_is_registered_and_keeps_python_code_tool_interface(runta):
    tool = make_tool("runta_python_code", timeout=7)

    assert isinstance(tool, RuntaPythonCodeTool)
    assert isinstance(tool, PythonCodeTool)
    assert tool.instruction_string() == PythonCodeTool().instruction_string()
    assert runta.creates == []  # no runtime until code is executed


def test_execute_action_runs_code_on_runtime_and_formats_observation(runta):
    tool = RuntaPythonCodeTool(timeout=7, memory_mib=2048, max_lifetime_seconds=900)
    runta.outcomes = [result(stdout="4\n")]

    is_valid, has_error, observation, parsed_action = tool.execute_action(
        "Let me check.\n<code>print(2 + 2)</code> done"
    )

    assert (is_valid, has_error) == (True, False)
    assert observation == "Code execution result: 4\n\n"
    assert parsed_action == "Let me check.\n<code>print(2 + 2)</code>"
    name, kwargs = runta.creates[0]
    assert name.startswith("roll-python-")
    assert kwargs["memory_mib"] == 2048
    assert kwargs["auto_shutdown_after_secs"] == 900
    call = runta.execs[0]
    assert call.argv == ["python3", "-"]
    assert call.stdin == "print(2 + 2)"  # code travels on stdin, never through a shell
    assert call.timeout == 7


def test_action_without_code_does_not_touch_runtime(runta):
    tool = RuntaPythonCodeTool()

    is_valid, has_error, observation, _ = tool.execute_action("no code here")

    assert (is_valid, has_error, observation) == (False, True, "")
    assert runta.creates == []


def test_failed_execution_reports_stderr_and_error_flag(runta):
    tool = RuntaPythonCodeTool(keep_error_last_line=True)
    runta.outcomes = [
        result(exit_code=1, stdout="before\n", stderr="Traceback (most recent call last):\nZeroDivisionError: x")
    ]

    _, has_error, observation, _ = tool.execute_action("<code>print('before'); 1/0</code>")

    assert has_error is True
    assert observation == "Code execution result: before\n\nZeroDivisionError: x\n"


def test_runtime_is_reused_across_executions(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")
    tool.execute_action("<code>print(2)</code>")
    assert len(runta.creates) == 1


def test_image_runs_code_inside_container(runta):
    tool = RuntaPythonCodeTool(image="python:3.12-slim")
    tool.execute_action("<code>print(1)</code>")

    start, run = runta.execs
    assert start.argv == [
        "docker",
        "run",
        "-d",
        "--name",
        "roll-python",
        "--network",
        "host",
        "python:3.12-slim",
        "sleep",
        "infinity",
    ]
    assert run.argv == ["docker", "exec", "-i", "roll-python", "python3", "-"]


def test_failed_image_start_deletes_runtime(runta):
    tool = RuntaPythonCodeTool(image="missing:tag")
    runta.outcomes = [result(exit_code=125, stderr="manifest unknown")]

    with pytest.raises(RuntimeError, match="manifest unknown"):
        tool.execute_action("<code>print(1)</code>")

    assert (runta.deletes, runta.closes) == (1, 1)
    assert tool.runtime is None


def test_expired_runtime_is_replaced_once(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")
    runta.outcomes = [FakeApiError("runtime is shut down"), result(stdout="2\n")]

    _, has_error, observation, _ = tool.execute_action("<code>print(2)</code>")

    assert has_error is False
    assert observation == "Code execution result: 2\n\n"
    assert len(runta.creates) == 2
    assert runta.deletes == 1


def test_repeated_api_errors_become_an_error_observation(runta):
    tool = RuntaPythonCodeTool()
    runta.outcomes = [FakeApiError("unavailable"), FakeApiError("unavailable")]

    _, has_error, observation, _ = tool.execute_action("<code>print(1)</code>")

    assert has_error is True
    assert observation == "Code execution result: Sandbox error: unavailable\n"
    assert tool.runtime is None


def test_close_deletes_runtime_and_is_idempotent(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")
    assert tool in tool_module._LIVE_TOOLS

    tool.close()
    tool.close()

    assert (runta.deletes, runta.closes) == (1, 1)
    assert tool not in tool_module._LIVE_TOOLS


def test_exit_hook_closes_live_tools(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")

    tool_module._close_live_tools()

    assert runta.deletes == 1
    assert tool.runtime is None


@pytest.mark.skipif(not os.environ.get("RUNTA_TOKEN"), reason="RUNTA_TOKEN is required")
def test_live_execution():
    tool = RuntaPythonCodeTool(timeout=20, max_lifetime_seconds=900)
    try:
        _, has_error, observation, _ = tool.execute_action("<code>print(sum(range(10)))</code>")
        assert (has_error, observation) == (False, "Code execution result: 45\n\n")
        _, has_error, observation, _ = tool.execute_action("<code>1/0</code>")
        assert has_error is True and "ZeroDivisionError" in observation
    finally:
        tool.close()
