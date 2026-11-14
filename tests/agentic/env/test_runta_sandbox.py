"""Tests for the Runta-hosted Sokoban sandbox environment.

The unit tests run against a fake ``runta`` module that mirrors the parts of the SDK
the environment relies on. ``test_live_game_round_trip`` talks to the real service and
is skipped unless ``RUNTA_TOKEN`` is set.
"""

import enum
import gc
import json
import logging
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from roll.pipeline.agentic.env.sandbox import runta_sandbox_env as env_module
from roll.pipeline.agentic.env.sandbox.runta_sandbox_env import (
    SOKOBAN_SANDBOX_IMAGE,
    SokobanRuntaSandboxEnv,
)
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
    CRASHED = "crashed"


def result(exit_code=0, stdout="", stderr=""):
    return SimpleNamespace(exit_code=exit_code, stdout_text=stdout, stderr_text=stderr)


RESET_STATE = {"observation": "push the boxes", "info": {"suffix": "Here is the state:\n###\n#P#\n###\n"}}
STEP_STATE = {
    "observation": "At turn 1, you moved Up.",
    "reward": 1.0,
    "terminated": True,
    "truncated": False,
    "info": {"suffix": "Here is the state:\n###\n#S#\n###\n"},
}


@pytest.fixture
def runta(monkeypatch):
    """
    Install a fake ``runta`` module whose runtimes answer scripted commands.

    Like the real SDK, ``create`` makes the runtime before it waits for it, so a failed
    wait loses the handle of a runtime that exists, and a runtime that was deleted or is
    not running rejects ``exec``. ``create_errors`` fail the create request itself,
    ``wait_errors`` fail ``wait_until_ready`` and ``policy_errors`` fail
    ``update_timeout_policies``, which otherwise records the renewal in the runtime's
    ``policy_updates``.

    Leases read the fake clock ``now`` and keep their real beat interval, so no heartbeat
    thread renews anything during a test: tests call ``beat()`` themselves.
    """
    state = SimpleNamespace(
        creates=[],
        runtimes=[],
        clients=[],
        leases=[],
        waits=[],
        commands=[],
        deletes=0,
        closes=0,
        create_errors=[],
        wait_errors=[],
        policy_errors=[],
        responses={},
        now=1000.0,
        renewed=threading.Event(),
    )

    def respond(argv):
        command = argv[-1] if argv[:2] == ["docker", "exec"] else " ".join(argv[:2])
        for needle, response in state.responses.items():
            if needle in command:
                return response
        if "curl" in command:
            return result(stdout="200")
        if "client.py reset" in command:
            return result(stdout=json.dumps(RESET_STATE))
        if "client.py action" in command:
            return result(stdout=json.dumps(STEP_STATE))
        return result()

    class FakeRuntime:
        def __init__(self):
            self.id = f"runtime-{len(state.runtimes) + 1}"
            self.status_value = FakeRuntimeStatus.CREATING
            self.deleted = False
            self.commands = []
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

        def exec(self, argv, timeout=None):
            if self.deleted or self.status_value != FakeRuntimeStatus.RUNNING:
                raise FakeApiError("runtime is not running")
            state.commands.append(argv)
            self.commands.append(argv)
            response = respond(argv)
            if isinstance(response, Exception):
                raise response
            return response

        def update_timeout_policies(self, *, auto_shutdown_after_secs=None, auto_delete_after_secs=None):
            if self.deleted:
                raise FakeApiError("runtime not found", status_code=404)
            if state.policy_errors:
                raise state.policy_errors.pop(0)
            self.policy_updates.append((auto_shutdown_after_secs, auto_delete_after_secs))
            state.renewed.set()

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
            if state.create_errors:
                raise state.create_errors.pop(0)
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
    monkeypatch.setattr("roll.pipeline.agentic.env.sandbox.runta_sandbox_env.time.sleep", lambda _: None)
    monkeypatch.setattr(env_module, "RuntaLease", make_lease)
    yield state
    env_module._LIVE_ENVS.clear()
    for lease in state.leases:
        lease.stop()


def client_commands(runtime):
    """The game client commands a fake runtime received, without the ``cd /app`` prefix."""
    return [argv[-1].split(" && ", 1)[1] for argv in runtime.commands if "client.py" in argv[-1]]


def test_env_is_registered_with_gem():
    import gem

    import roll.pipeline.agentic.env  # noqa: F401  (registers the environments)

    env = gem.make("sokoban_runta_sandbox", memory_mib=4096)
    assert isinstance(env.unwrapped if hasattr(env, "unwrapped") else env, SokobanRuntaSandboxEnv)


def test_construction_does_not_create_a_runtime(runta):
    SokobanRuntaSandboxEnv()
    assert runta.creates == []


def test_reset_starts_runtime_container_and_server_then_returns_game_state(runta):
    env = SokobanRuntaSandboxEnv(vcpus=2, memory_mib=4096, max_lifetime_seconds=900, lease_seconds=120)

    obs, info = env.reset(seed=3)

    name, kwargs = runta.creates[0]
    assert name.startswith("roll-sokoban-")
    assert kwargs == {
        "vcpus": 2,
        "memory_mib": 4096,
        "auto_shutdown_after_secs": 120,  # the lease, not the lifetime
        "auto_delete_after_secs": 600,
        "wait": False,
    }
    assert runta.waits == [600]
    assert runta.commands[0] == [
        "docker",
        "run",
        "-d",
        "--name",
        "roll-sokoban",
        "--network",
        "host",
        SOKOBAN_SANDBOX_IMAGE,
        "sleep",
        "infinity",
    ]
    assert runta.commands[1][:4] == ["docker", "exec", "-d", "roll-sokoban"]
    assert "python server.py" in runta.commands[1][-1]
    assert runta.commands[-1] == [
        "docker",
        "exec",
        "roll-sokoban",
        "bash",
        "-c",
        "cd /app && python client.py reset --seed 3 --json",
    ]
    assert obs == "push the boxes"
    assert info == RESET_STATE["info"]
    assert env.render() == "###\n#P#\n###"


def test_step_quotes_action_and_parses_transition(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step("<answer>Up</answer>\n; rm -rf /")

    assert runta.commands[-1][-1] == "cd /app && python client.py action '<answer>Up</answer> ; rm -rf /' --json"
    assert (obs, reward, terminated, truncated) == ("At turn 1, you moved Up.", 1.0, True, False)
    assert env.render() == "###\n#S#\n###"


def test_runtime_is_reused_across_episodes(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset(seed=1)
    env.reset(seed=2)
    assert len(runta.creates) == 1


def test_non_json_output_ends_episode_with_error_info(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()
    runta.responses["client.py action"] = result(stdout="Traceback: server crashed")

    obs, reward, terminated, truncated, info = env.step("<answer>Up</answer>")

    assert terminated is True and reward == 0
    assert info == {"error": "JSONDecodeError", "raw_output": "Traceback: server crashed"}


def test_close_deletes_runtime_and_next_reset_creates_a_new_one(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()

    env.close()
    env.close()
    assert (runta.deletes, runta.closes) == (1, 1)

    env.reset()
    assert len(runta.creates) == 2


def test_runtime_is_created_with_the_default_lease(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()

    assert runta.creates[0][1]["auto_shutdown_after_secs"] == 600
    assert env._lease.interval == 200
    assert env._lease.runtime_id == "runtime-1"
    assert env._lease._thread.daemon  # the heartbeat never keeps the process alive


def test_lease_longer_than_the_lifetime_is_capped_at_creation(runta):
    env = SokobanRuntaSandboxEnv(max_lifetime_seconds=300, lease_seconds=900)
    env.reset()

    assert runta.creates[0][1]["auto_shutdown_after_secs"] == 300


def test_lease_that_would_disable_the_shutdown_policy_is_rejected(runta):
    with pytest.raises(ValueError, match="lease_seconds"):
        SokobanRuntaSandboxEnv(lease_seconds=0)


def test_heartbeat_renews_the_lease_on_its_own_client(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()

    runta.now += 200
    assert env._lease.beat() is True

    assert runta.runtimes[0].policy_updates == [(600, None)]  # only the shutdown clock is re-armed
    env_client, lease_client = runta.clients
    assert env.client is env_client
    assert lease_client.kwargs == {"retry_max_attempts": 1}  # a failed renewal waits for the next beat


def test_heartbeat_client_uses_the_env_endpoint(runta):
    env = SokobanRuntaSandboxEnv(endpoint="https://runta.example")
    env.reset()

    env._lease.beat()

    assert runta.clients[1].kwargs == {"retry_max_attempts": 1, "endpoint": "https://runta.example"}


def test_heartbeat_never_renews_beyond_max_lifetime(runta):
    env = SokobanRuntaSandboxEnv(max_lifetime_seconds=1000, lease_seconds=600)
    env.reset()
    lease, runtime = env._lease, runta.runtimes[0]

    runta.now += 200  # 800 s of lifetime left
    assert lease.beat() is True
    runta.now += 400  # 400 s left: less than a lease
    assert lease.beat() is True
    runta.now += 399.5  # half a second left: nothing to ask for, and 0 would disable the policy
    assert lease.beat() is False
    runta.now += 200  # past the cap
    assert lease.beat() is False

    assert runtime.policy_updates == [(600, None), (400, None)]


def test_failed_renewal_is_logged_and_retried_on_the_next_beat(runta, caplog):
    env = SokobanRuntaSandboxEnv()
    env.reset(seed=1)
    runta.policy_errors = [FakeApiError("runtime revision conflict: expected 3", status_code=409), OSError("boom")]

    with caplog.at_level(logging.WARNING):
        assert env._lease.beat() is True
        assert env._lease.beat() is True
    assert runta.runtimes[0].policy_updates == []
    assert [r.levelname for r in caplog.records if "Could not renew the lease" in r.getMessage()] == ["WARNING"] * 2

    # The episode is unaffected, and the next beat renews.
    obs, reward, terminated, truncated, info = env.step("<answer>Up</answer>")
    assert reward == 1.0
    assert env._lease.beat() is True
    assert runta.runtimes[0].policy_updates == [(600, None)]


def test_close_stops_the_heartbeat(runta, caplog):
    env = SokobanRuntaSandboxEnv()
    env.reset()
    lease = env._lease
    lease.beat()

    with caplog.at_level(logging.WARNING):
        env.close()

    assert env._lease is None
    assert lease.stopped
    assert lease.join(timeout=5)
    assert runta.clients[1].closed  # the heartbeat's own client is released too
    assert lease.beat() is False
    assert runta.runtimes[0].policy_updates == [(600, None)]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_renewal_that_loses_the_race_with_close_is_not_reported(runta, caplog):
    env = SokobanRuntaSandboxEnv()
    env.reset()
    lease, runtime = env._lease, runta.runtimes[0]

    def deleted_meanwhile(**kwargs):
        env.close()
        raise FakeApiError("runtime not found", status_code=404)

    runtime.update_timeout_policies = deleted_meanwhile

    with caplog.at_level(logging.WARNING):
        assert lease.beat() is False
    assert [r for r in caplog.records if "renew" in r.getMessage()] == []


def test_replacing_a_stopped_runtime_stops_its_heartbeat_and_starts_a_new_one(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset(seed=1)
    old_lease = env._lease
    runta.runtimes[0].status_value = FakeRuntimeStatus.SHUTDOWN  # the lease lapsed

    env.step("<answer>Up</answer>")

    assert old_lease.stopped and old_lease.join(timeout=5)
    assert env._lease is not old_lease
    assert env._lease.runtime_id == "runtime-2"
    assert not env._lease.stopped
    assert runta.creates[1][1]["auto_shutdown_after_secs"] == 600
    env._lease.beat()
    assert runta.runtimes[0].policy_updates == []
    assert runta.runtimes[1].policy_updates == [(600, None)]


def test_failed_initialization_stops_the_heartbeat(runta):
    runta.wait_errors = [RuntimeError("runtime runtime-1 entered error state")]

    with pytest.raises(RuntimeError, match="critical error"):
        SokobanRuntaSandboxEnv().reset()

    assert [lease.stopped for lease in runta.leases] == [True]
    assert runta.leases[0].join(timeout=5)


def test_heartbeat_stops_when_the_env_is_garbage_collected(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()
    lease = env._lease

    del env
    gc.collect()

    assert lease.beat() is False
    assert runta.runtimes[0].policy_updates == []


def test_heartbeat_thread_renews_until_stopped(runta):
    owner = SokobanRuntaSandboxEnv()
    owner.reset()
    lease = RuntaLease(lease_seconds=600, max_lifetime_seconds=7200, owner=owner, interval=0.01)

    lease.start("runtime-1")
    assert runta.renewed.wait(timeout=5)
    lease.stop()

    assert lease.join(timeout=5)  # the thread ended, so nothing renews after stop()
    assert set(runta.runtimes[0].policy_updates) == {(600, None)}
    assert runta.clients[-1].closed


def test_exit_hook_closes_live_envs(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()
    assert env in env_module._LIVE_ENVS

    env_module._close_live_envs()

    assert runta.runtimes[0].deleted
    assert (runta.deletes, runta.closes) == (1, 1)
    assert env.runtime is None
    assert env not in env_module._LIVE_ENVS


def test_exit_hook_deletes_runtime_interrupted_while_booting(runta):
    runta.wait_errors = [KeyboardInterrupt()]
    env = SokobanRuntaSandboxEnv()

    with pytest.raises(KeyboardInterrupt):
        env.reset()
    env_module._close_live_envs()

    assert runta.runtimes[0].deleted


def test_reset_replaces_runtime_that_was_shut_down(runta, caplog):
    env = SokobanRuntaSandboxEnv()
    env.reset(seed=1)
    env.step("<answer>Up</answer>")
    old_runtime = runta.runtimes[0]
    old_runtime.status_value = FakeRuntimeStatus.SHUTDOWN  # e.g. max_lifetime_seconds was reached

    with caplog.at_level(logging.WARNING):
        obs, info = env.reset(seed=2)

    assert (obs, info) == ("push the boxes", RESET_STATE["info"])
    assert old_runtime.deleted
    assert len(runta.runtimes) == 2
    # A new episode starts on the new runtime: nothing of the previous one is replayed.
    assert client_commands(runta.runtimes[1]) == ["python client.py reset --seed 2 --json"]
    assert [r.levelname for r in caplog.records if "no longer running" in r.getMessage()] == ["WARNING"]


def test_step_replaces_deleted_runtime_and_replays_the_episode(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset(seed=3)
    env.step("<answer>Up</answer>")
    runta.runtimes[0].deleted = True  # deleted behind the env's back

    obs, reward, terminated, truncated, info = env.step("<answer>Left</answer>")

    assert (obs, reward, terminated, truncated) == ("At turn 1, you moved Up.", 1.0, True, False)
    assert len(runta.runtimes) == 2
    assert client_commands(runta.runtimes[1]) == [
        "python client.py reset --seed 3 --json",
        "python client.py action '<answer>Up</answer>' --json",
        "python client.py action '<answer>Left</answer>' --json",
    ]


def test_crashed_runtime_is_replaced(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset(seed=4)
    runta.runtimes[0].status_value = FakeRuntimeStatus.CRASHED

    env.step("<answer>Up</answer>")

    assert runta.runtimes[0].deleted
    assert len(runta.runtimes) == 2


def test_unseeded_episode_is_not_replayed_on_a_new_runtime(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()  # random map: it cannot be restored on another runtime
    runta.runtimes[0].status_value = FakeRuntimeStatus.SHUTDOWN

    with pytest.raises(RuntimeError, match="reset without a seed"):
        env.step("<answer>Up</answer>")

    assert runta.runtimes[0].deleted
    assert len(runta.runtimes) == 1

    # The next episode starts cleanly on a new runtime.
    env.reset(seed=5)
    assert client_commands(runta.runtimes[1]) == ["python client.py reset --seed 5 --json"]


def test_exec_failure_on_a_running_runtime_is_raised(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()
    runta.responses["client.py action"] = FakeApiError("exec stream broke")

    with pytest.raises(FakeApiError, match="exec stream broke"):
        env.step("<answer>Up</answer>")

    assert len(runta.runtimes) == 1
    assert not runta.runtimes[0].deleted


def test_failure_that_outlives_the_replacement_runtime_is_raised(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset(seed=6)
    runta.runtimes[0].status_value = FakeRuntimeStatus.SHUTDOWN
    runta.responses["client.py action"] = FakeApiError("still broken")

    with pytest.raises(FakeApiError, match="still broken"):
        env.step("<answer>Up</answer>")

    assert len(runta.runtimes) == 2


def test_stopped_runtime_that_cannot_be_replaced_is_raised(runta):
    env = SokobanRuntaSandboxEnv()
    env.reset()
    runta.runtimes[0].status_value = FakeRuntimeStatus.SHUTDOWN
    runta.create_errors = [FakeApiError("quota exceeded")] * 3

    with pytest.raises(RuntimeError, match="multiple Runta API errors"):
        env.reset()

    assert runta.runtimes[0].deleted


def test_transient_api_error_is_retried(runta):
    runta.create_errors = [FakeApiError("503 unavailable")]
    env = SokobanRuntaSandboxEnv()

    env.reset()

    assert len(runta.creates) == 2
    assert runta.closes == 1  # the failed attempt's client was released


def test_persistent_api_errors_fail_after_three_attempts(runta):
    runta.create_errors = [FakeApiError("down")] * 3

    with pytest.raises(RuntimeError, match="multiple Runta API errors"):
        SokobanRuntaSandboxEnv().reset()

    assert len(runta.creates) == 3


def test_runtime_that_never_becomes_ready_is_deleted_before_each_retry(runta):
    runta.wait_errors = [TimeoutError("runtime did not become ready")] * 3

    with pytest.raises(RuntimeError, match="multiple Runta API errors"):
        SokobanRuntaSandboxEnv().reset()

    assert len(runta.runtimes) == 3
    assert all(runtime.deleted for runtime in runta.runtimes)


def test_runtime_that_fails_to_boot_is_deleted_without_retry(runta):
    runta.wait_errors = [RuntimeError("runtime runtime-1 entered error state")]

    with pytest.raises(RuntimeError, match="critical error"):
        SokobanRuntaSandboxEnv().reset()

    assert len(runta.runtimes) == 1
    assert runta.runtimes[0].deleted


def test_failed_image_start_deletes_runtime_without_retry(runta):
    runta.responses["docker run"] = result(exit_code=125, stderr="manifest unknown")

    with pytest.raises(RuntimeError, match="critical error"):
        SokobanRuntaSandboxEnv(image="missing:tag").reset()

    assert len(runta.creates) == 1
    assert runta.deletes == 1


def test_server_that_never_answers_fails_initialization(runta, monkeypatch):
    runta.responses["curl"] = result(stdout="000")
    clock = iter(range(0, 10_000, 10))
    monkeypatch.setattr("roll.pipeline.agentic.env.sandbox.runta_sandbox_env.time.time", lambda: next(clock))

    with pytest.raises(RuntimeError, match="critical error"):
        SokobanRuntaSandboxEnv(server_start_timeout=30).reset()

    assert runta.deletes == 1


@pytest.mark.skipif(not os.environ.get("RUNTA_TOKEN"), reason="RUNTA_TOKEN is required")
def test_live_game_round_trip():
    env = SokobanRuntaSandboxEnv(max_lifetime_seconds=900)
    try:
        obs, info = env.reset(seed=1)
        assert "push all boxes" in obs
        assert "#" in env.render()
        obs, reward, terminated, truncated, info = env.step("<answer>Up</answer>")
        assert isinstance(reward, float)
        assert "metrics" in info
    finally:
        env.close()
