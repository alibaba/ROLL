"""Tests for the Runta-hosted Sokoban sandbox environment.

The unit tests run against a fake ``runta`` module. ``test_live_game_round_trip``
talks to the real service and is skipped unless ``RUNTA_TOKEN`` is set.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from roll.pipeline.agentic.env.sandbox.runta_sandbox_env import (
    SOKOBAN_SANDBOX_IMAGE,
    SokobanRuntaSandboxEnv,
)


class FakeApiError(Exception):
    pass


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
    """Install a fake ``runta`` module whose runtime answers scripted commands."""
    state = SimpleNamespace(creates=[], commands=[], deletes=0, closes=0, create_errors=[], responses={})

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
        id = "runtime-1"

        def exec(self, argv, timeout=None):
            state.commands.append(argv)
            response = respond(argv)
            if isinstance(response, Exception):
                raise response
            return response

        def delete(self):
            state.deletes += 1

    class Runta:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.runtimes = SimpleNamespace(create=self._create)

        def _create(self, name, **kwargs):
            state.creates.append((name, kwargs))
            if state.create_errors:
                raise state.create_errors.pop(0)
            return FakeRuntime()

        def close(self):
            state.closes += 1

    monkeypatch.setitem(sys.modules, "runta", SimpleNamespace(Runta=Runta, ApiError=FakeApiError))
    monkeypatch.setattr("roll.pipeline.agentic.env.sandbox.runta_sandbox_env.time.sleep", lambda _: None)
    return state


def test_env_is_registered_with_gem():
    import gem

    import roll.pipeline.agentic.env  # noqa: F401  (registers the environments)

    env = gem.make("sokoban_runta_sandbox", memory_mib=4096)
    assert isinstance(env.unwrapped if hasattr(env, "unwrapped") else env, SokobanRuntaSandboxEnv)


def test_construction_does_not_create_a_runtime(runta):
    SokobanRuntaSandboxEnv()
    assert runta.creates == []


def test_reset_starts_runtime_container_and_server_then_returns_game_state(runta):
    env = SokobanRuntaSandboxEnv(vcpus=2, memory_mib=4096, max_lifetime_seconds=900)

    obs, info = env.reset(seed=3)

    name, kwargs = runta.creates[0]
    assert name.startswith("roll-sokoban-")
    assert kwargs == {
        "vcpus": 2,
        "memory_mib": 4096,
        "auto_shutdown_after_secs": 900,
        "auto_delete_after_secs": 600,
        "timeout": 600,
    }
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
