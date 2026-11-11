"""Tests for the Runta-backed python code tool.

The unit tests run against a fake ``runta`` module that mirrors the parts of the SDK
the tool relies on. ``test_live_execution`` talks to the real service and is skipped
unless ``RUNTA_TOKEN`` is set.
"""

import enum
import os
import sys
from types import SimpleNamespace

import pytest
from gem.utils.constants import BASE_IMPORTS

from roll.pipeline.agentic.tools import runta_python_code_tool as tool_module
from roll.pipeline.agentic.tools.python_code_tool import PythonCodeTool
from roll.pipeline.agentic.tools.registration import make_tool
from roll.pipeline.agentic.tools.runta_python_code_tool import RuntaPythonCodeTool


class FakeApiError(Exception):
    def __init__(self, message="", *, status_code=None, code=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class FakeRuntimeStatus(str, enum.Enum):
    RUNNING = "running"
    SHUTDOWN = "shutdown"
    CREATING = "creating"


def result(exit_code=0, stdout="", stderr=""):
    return SimpleNamespace(exit_code=exit_code, stdout_text=stdout, stderr_text=stderr)


@pytest.fixture
def runta(monkeypatch):
    """
    Install a fake ``runta`` module whose runtimes answer scripted outcomes.

    Like the real SDK, ``create`` makes the runtime before it waits for it, so a failed
    wait loses the handle of a runtime that exists, and a runtime that was deleted or is
    not running rejects ``exec``. ``wait_errors`` fail ``wait_until_ready``.
    """
    state = SimpleNamespace(
        creates=[], runtimes=[], waits=[], execs=[], outcomes=[], deletes=0, closes=0, wait_errors=[]
    )

    class FakeRuntime:
        def __init__(self):
            self.id = f"runtime-{len(state.runtimes) + 1}"
            self.status_value = FakeRuntimeStatus.CREATING
            self.deleted = False

        def wait_until_ready(self, *, timeout=240.0, poll_interval=0.5):
            state.waits.append(timeout)
            if state.wait_errors:
                raise state.wait_errors.pop(0)
            self.status_value = FakeRuntimeStatus.RUNNING

        def status(self):
            if self.deleted:
                raise FakeApiError("runtime not found", status_code=404)
            return self.status_value

        def exec(self, argv, timeout=None, stdin=None):
            if self.deleted or self.status_value != FakeRuntimeStatus.RUNNING:
                raise FakeApiError("runtime is not running")
            state.execs.append(SimpleNamespace(argv=argv, timeout=timeout, stdin=stdin))
            outcome = state.outcomes.pop(0) if state.outcomes else result()
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def delete(self):
            state.deletes += 1
            self.deleted = True

    class Runta:
        def __init__(self, **kwargs):
            self.runtimes = SimpleNamespace(create=self._create)

        def _create(self, name, **kwargs):
            state.creates.append((name, kwargs))
            runtime = FakeRuntime()
            state.runtimes.append(runtime)
            if kwargs.get("wait", True):
                runtime.wait_until_ready(timeout=kwargs.get("timeout", 240.0))
            return runtime

        def close(self):
            state.closes += 1

    fake_module = SimpleNamespace(Runta=Runta, ApiError=FakeApiError, RuntimeStatus=FakeRuntimeStatus)
    monkeypatch.setitem(sys.modules, "runta", fake_module)
    yield state
    tool_module._LIVE_TOOLS.clear()


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
    assert kwargs["wait"] is False
    assert runta.waits == [600]
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
        result(exit_code=1, stdout="before\n", stderr="Traceback (most recent call last):\nZeroDivisionError: x\n")
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


def test_runtime_that_never_becomes_ready_is_deleted(runta):
    tool = RuntaPythonCodeTool()
    runta.wait_errors = [TimeoutError("runtime did not become ready")] * 2

    _, has_error, observation, _ = tool.execute_action("<code>print(1)</code>")

    assert has_error is True
    assert observation == "Code execution result: Sandbox error: runtime did not become ready\n"
    assert len(runta.runtimes) == 2
    assert all(runtime.deleted for runtime in runta.runtimes)
    assert runta.closes == 2
    assert tool.runtime is None


def test_runtime_that_fails_to_boot_is_deleted(runta):
    tool = RuntaPythonCodeTool()
    runta.wait_errors = [RuntimeError("runtime runtime-1 entered error state")]

    with pytest.raises(RuntimeError, match="entered error state"):
        tool.execute_action("<code>print(1)</code>")

    assert runta.runtimes[0].deleted
    assert runta.closes == 1
    assert tool.runtime is None


def test_expired_runtime_is_replaced_once(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")
    runta.runtimes[0].status_value = FakeRuntimeStatus.SHUTDOWN  # e.g. max_lifetime_seconds was reached
    runta.outcomes = [result(stdout="2\n")]

    _, has_error, observation, _ = tool.execute_action("<code>print(2)</code>")

    assert has_error is False
    assert observation == "Code execution result: 2\n\n"
    assert len(runta.runtimes) == 2
    assert runta.runtimes[0].deleted
    assert runta.deletes == 1


def test_repeated_api_errors_become_an_error_observation(runta):
    tool = RuntaPythonCodeTool()
    runta.outcomes = [FakeApiError("unavailable"), FakeApiError("unavailable")]

    _, has_error, observation, _ = tool.execute_action("<code>print(1)</code>")

    assert has_error is True
    assert observation == "Code execution result: Sandbox error: unavailable\n"
    assert tool.runtime is None


def test_forbidden_code_is_blocked_like_the_local_tool(runta):
    action = "<code>import subprocess\nsubprocess.run(['ls'])</code>"

    outcome = RuntaPythonCodeTool().execute_action(action)

    assert outcome == PythonCodeTool().execute_action(action)  # gem blocks it before running anything
    assert outcome[1] is True
    assert outcome[2] == (
        "Code execution result: Execution blocked: Code contains potentially dangerous operations or imports.\n"
    )
    assert runta.creates == []


def test_undefined_name_is_retried_with_gem_base_imports(runta):
    tool = RuntaPythonCodeTool()
    runta.outcomes = [
        result(exit_code=1, stderr="Traceback (most recent call last):\nNameError: name 'sqrt' is not defined\n"),
        result(stdout="3.0\n"),
    ]

    _, has_error, observation, _ = tool.execute_action("<code>print(sqrt(9))</code>")

    assert has_error is False
    assert observation == "Code execution result: 3.0\n\n"
    assert [call.stdin for call in runta.execs] == ["print(sqrt(9))", BASE_IMPORTS + "\nprint(sqrt(9))"]
    assert len(runta.creates) == 1


def test_base_imports_retry_that_lacks_modules_keeps_the_original_error(runta):
    tool = RuntaPythonCodeTool()
    runta.outcomes = [
        result(exit_code=1, stderr="Traceback (most recent call last):\nNameError: name 'foo' is not defined\n"),
        result(exit_code=1, stderr="ModuleNotFoundError: No module named 'numpy'\n"),
    ]

    _, has_error, observation, _ = tool.execute_action("<code>print(foo)</code>")

    assert has_error is True
    assert "name 'foo' is not defined" in observation
    assert "No module named" not in observation


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
