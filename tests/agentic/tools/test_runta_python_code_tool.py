"""Tests for the Runta-backed python code tool.

The unit tests run against a fake ``runta`` module that mirrors the parts of the SDK
the tool relies on. ``test_live_execution`` talks to the real service and is skipped
unless ``RUNTA_TOKEN`` is set.
"""

import enum
import logging
import os
import sys
from types import SimpleNamespace

import pytest
from gem.utils.constants import BASE_IMPORTS

from roll.pipeline.agentic.tools import runta_python_code_tool as tool_module
from roll.pipeline.agentic.tools.python_code_tool import PythonCodeTool
from roll.pipeline.agentic.tools.registration import make_tool
from roll.pipeline.agentic.tools.runta_python_code_tool import RuntaPythonCodeTool
from roll.utils.runta_lease import RuntaLease


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
    not running rejects ``exec``. ``wait_errors`` fail ``wait_until_ready`` and
    ``policy_errors`` fail ``update_timeout_policies``, which otherwise records the
    renewal in the runtime's ``policy_updates``.

    Leases read the fake clock ``now`` and keep their real beat interval, so no heartbeat
    thread renews anything during a test: tests call ``beat()`` themselves.
    """
    state = SimpleNamespace(
        creates=[],
        runtimes=[],
        clients=[],
        leases=[],
        waits=[],
        execs=[],
        outcomes=[],
        deletes=0,
        closes=0,
        wait_errors=[],
        policy_errors=[],
        now=1000.0,
    )

    class FakeRuntime:
        def __init__(self):
            self.id = f"runtime-{len(state.runtimes) + 1}"
            self.status_value = FakeRuntimeStatus.CREATING
            self.deleted = False
            self.policy_updates = []

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

        def update_timeout_policies(self, *, auto_shutdown_after_secs=None, auto_delete_after_secs=None):
            if self.deleted:
                raise FakeApiError("runtime not found", status_code=404)
            if state.policy_errors:
                raise state.policy_errors.pop(0)
            self.policy_updates.append((auto_shutdown_after_secs, auto_delete_after_secs))

        def delete(self):
            state.deletes += 1
            self.deleted = True

    class Runta:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.runtimes = SimpleNamespace(create=self._create, get=self._get)
            state.clients.append(self)

        def _create(self, name, **kwargs):
            state.creates.append((name, kwargs))
            runtime = FakeRuntime()
            state.runtimes.append(runtime)
            if kwargs.get("wait", True):
                runtime.wait_until_ready(timeout=kwargs.get("timeout", 240.0))
            return runtime

        def _get(self, id):
            for runtime in state.runtimes:
                if runtime.id == id and not runtime.deleted:
                    return runtime
            raise FakeApiError("runtime not found", status_code=404)

        def close(self):
            self.closed = True
            state.closes += 1

    def make_lease(**kwargs):
        lease = RuntaLease(clock=lambda: state.now, **kwargs)
        state.leases.append(lease)
        return lease

    fake_module = SimpleNamespace(Runta=Runta, ApiError=FakeApiError, RuntimeStatus=FakeRuntimeStatus)
    monkeypatch.setitem(sys.modules, "runta", fake_module)
    monkeypatch.setattr(tool_module, "RuntaLease", make_lease)
    yield state
    tool_module._LIVE_TOOLS.clear()
    for lease in state.leases:
        lease.stop()


def test_tool_is_registered_and_keeps_python_code_tool_interface(runta):
    tool = make_tool("runta_python_code", timeout=7)

    assert isinstance(tool, RuntaPythonCodeTool)
    assert isinstance(tool, PythonCodeTool)
    assert tool.instruction_string() == PythonCodeTool().instruction_string()
    assert runta.creates == []  # no runtime until code is executed


def test_execute_action_runs_code_on_runtime_and_formats_observation(runta):
    tool = RuntaPythonCodeTool(timeout=7, memory_mib=2048, max_lifetime_seconds=900, lease_seconds=120)
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
    assert kwargs["auto_shutdown_after_secs"] == 120  # the lease, not the lifetime
    assert kwargs["auto_delete_after_secs"] == 600
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
    # The program still arrives on stdin; the wrapper only records its process group.
    assert run.argv[:6] == ["docker", "exec", "-i", "roll-python", "sh", "-c"]
    assert run.argv[6] == tool_module._TRACK
    assert run.argv[7] == "roll-exec"
    assert run.argv[8].startswith("/tmp/roll-exec-") and run.argv[8].endswith(".pid")
    assert run.stdin == "print(1)"
    assert "python3 -" in tool_module._TRACK and "<" not in tool_module._TRACKED  # stdin is left alone


def test_program_that_times_out_in_the_container_is_killed(runta):
    tool = RuntaPythonCodeTool(image="python:3.12-slim", timeout=3)
    runta.outcomes = [result(), result(exit_code=124, stderr="Command timed out after 3 seconds.")]

    _, has_error, observation, _ = tool.execute_action("<code>while True: pass</code>")

    assert has_error is True
    assert observation == "Code execution result: Command timed out after 3 seconds.\n"
    _, run, kill = runta.execs
    pidfile = run.argv[-1]
    assert kill.argv == [
        "docker",
        "exec",
        "-u",
        "0",
        "roll-python",
        "sh",
        "-c",
        tool_module._KILL,
        "roll-kill",
        pidfile,
    ]
    assert kill.stdin is None
    assert len(runta.creates) == 1  # the runtime is kept


def test_each_execution_in_the_container_gets_its_own_pid_file(runta):
    tool = RuntaPythonCodeTool(image="python:3.12-slim")
    tool.execute_action("<code>print(1)</code>")
    tool.execute_action("<code>print(2)</code>")

    _, first, second = runta.execs  # no kill command: both programs ended in time
    assert first.argv[-1] != second.argv[-1]


def test_failed_program_in_the_container_is_not_killed(runta):
    tool = RuntaPythonCodeTool(image="python:3.12-slim")
    runta.outcomes = [result(), result(exit_code=1, stderr="ZeroDivisionError: division by zero\n")]

    tool.execute_action("<code>1/0</code>")

    assert len(runta.execs) == 2


def test_timeout_without_image_issues_no_kill_command(runta):
    tool = RuntaPythonCodeTool(timeout=3)  # Runta itself kills a program that runs on the runtime
    runta.outcomes = [result(exit_code=124, stderr="Command timed out after 3 seconds.")]

    _, has_error, observation, _ = tool.execute_action("<code>while True: pass</code>")

    assert has_error is True
    assert [call.argv for call in runta.execs] == [["python3", "-"]]


def test_kill_command_failure_does_not_change_the_observation(runta, caplog):
    tool = RuntaPythonCodeTool(image="python:3.12-slim", timeout=3)
    runta.outcomes = [
        result(),
        result(exit_code=124, stderr="Command timed out after 3 seconds."),
        FakeApiError("exec stream broke"),
    ]

    with caplog.at_level(logging.WARNING):
        _, has_error, observation, _ = tool.execute_action("<code>while True: pass</code>")

    assert observation == "Code execution result: Command timed out after 3 seconds.\n"
    assert len(runta.creates) == 1
    assert any("Could not stop the timed-out program" in r.getMessage() for r in caplog.records)


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


def test_runtime_is_created_with_the_default_lease(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")

    assert runta.creates[0][1]["auto_shutdown_after_secs"] == 600
    assert tool._lease.interval == 200
    assert tool._lease.runtime_id == "runtime-1"
    assert tool._lease._thread.daemon  # the heartbeat never keeps the process alive


def test_lease_longer_than_the_lifetime_is_capped_at_creation(runta):
    tool = RuntaPythonCodeTool(max_lifetime_seconds=300, lease_seconds=900)
    tool.execute_action("<code>print(1)</code>")

    assert runta.creates[0][1]["auto_shutdown_after_secs"] == 300


def test_lease_that_would_disable_the_shutdown_policy_is_rejected(runta):
    with pytest.raises(ValueError, match="lease_seconds"):
        RuntaPythonCodeTool(lease_seconds=0)


def test_heartbeat_renews_the_lease_on_its_own_client(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")

    runta.now += 200
    assert tool._lease.beat() is True

    assert runta.runtimes[0].policy_updates == [(600, None)]  # only the shutdown clock is re-armed
    tool_client, lease_client = runta.clients
    assert tool.client is tool_client
    assert lease_client.kwargs == {"retry_max_attempts": 1}  # a failed renewal waits for the next beat


def test_heartbeat_never_renews_beyond_max_lifetime(runta):
    tool = RuntaPythonCodeTool(max_lifetime_seconds=1000, lease_seconds=600)
    tool.execute_action("<code>print(1)</code>")
    lease, runtime = tool._lease, runta.runtimes[0]

    runta.now += 200  # 800 s of lifetime left
    assert lease.beat() is True
    runta.now += 400  # 400 s left: less than a lease
    assert lease.beat() is True
    runta.now += 399.5  # half a second left: nothing to ask for, and 0 would disable the policy
    assert lease.beat() is False

    assert runtime.policy_updates == [(600, None), (400, None)]


def test_failed_renewal_is_logged_and_retried_on_the_next_beat(runta, caplog):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")
    runta.policy_errors = [FakeApiError("runtime revision conflict: expected 3", status_code=409)]

    with caplog.at_level(logging.WARNING):
        assert tool._lease.beat() is True
    assert runta.runtimes[0].policy_updates == []
    assert [r.levelname for r in caplog.records if "Could not renew the lease" in r.getMessage()] == ["WARNING"]

    # Executions are unaffected, and the next beat renews.
    runta.outcomes = [result(stdout="2\n")]
    assert tool.execute_action("<code>print(2)</code>")[2] == "Code execution result: 2\n\n"
    assert tool._lease.beat() is True
    assert runta.runtimes[0].policy_updates == [(600, None)]
    assert len(runta.creates) == 1


def test_close_stops_the_heartbeat(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")
    lease = tool._lease
    lease.beat()

    tool.close()

    assert tool._lease is None
    assert lease.stopped
    assert lease.join(timeout=5)
    assert runta.clients[1].closed  # the heartbeat's own client is released too
    assert lease.beat() is False
    assert runta.runtimes[0].policy_updates == [(600, None)]


def test_replacing_an_expired_runtime_stops_its_heartbeat_and_starts_a_new_one(runta):
    tool = RuntaPythonCodeTool()
    tool.execute_action("<code>print(1)</code>")
    old_lease = tool._lease
    runta.runtimes[0].status_value = FakeRuntimeStatus.SHUTDOWN  # the lease lapsed

    tool.execute_action("<code>print(2)</code>")

    assert old_lease.stopped and old_lease.join(timeout=5)
    assert tool._lease is not old_lease
    assert tool._lease.runtime_id == "runtime-2"
    assert not tool._lease.stopped
    tool._lease.beat()
    assert runta.runtimes[0].policy_updates == []
    assert runta.runtimes[1].policy_updates == [(600, None)]


def test_runtime_that_fails_to_start_stops_its_heartbeat(runta):
    tool = RuntaPythonCodeTool(image="missing:tag")
    runta.outcomes = [result(exit_code=125, stderr="manifest unknown")]

    with pytest.raises(RuntimeError, match="manifest unknown"):
        tool.execute_action("<code>print(1)</code>")

    assert [lease.stopped for lease in runta.leases] == [True]
    assert runta.leases[0].join(timeout=5)
    assert tool._lease is None


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
