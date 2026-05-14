import logging
import time

import httpx

from gem import Env
from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func

logger = logging.getLogger(__name__)


class CliEnv(Env):
    """CLI environment backed by ROCK sandbox.

    The LLM agent sends shell commands as actions, receives stdout as observations.
    Lifecycle: reset() starts a fresh sandbox, step() runs commands.
    Cleanup is handled via reset() (stops previous sandbox) and __del__ (final cleanup).
    """

    def __init__(
        self,
        sandbox_base_url: str = "http://localhost:8080",
        sandbox_type: str = "minisandbox",
        sandbox_image: str = "",
        memory: str = "8g",
        cpus: float = 2.0,
        auto_clear_seconds: int = 1200,
        workspace_dir: str = "/tmp/cli_workspace",
        max_steps: int = 30,
        format_penalty: float = -0.1,
        action_pattern: str = "<answer>(.*?)</answer>",
        special_token_list: tuple = ("<|im_start|>", "<|im_end|>"),
        env_instruction: str | None = None,
        **kwargs,
    ):
        self.sandbox_base_url = sandbox_base_url.rstrip("/")
        self._api_url = f"{self.sandbox_base_url}/apis/envs/sandbox/v1"
        self.sandbox_type = sandbox_type
        self.sandbox_image = sandbox_image
        self.memory = memory
        self.cpus = cpus
        self.auto_clear_seconds = auto_clear_seconds
        self.workspace_dir = workspace_dir
        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.env_instruction = env_instruction or (
            "You are a CLI assistant. Execute shell commands to complete tasks. "
            "Output format: <answer>COMMAND</answer> where COMMAND is a valid shell command."
        )

        self._client = httpx.Client(timeout=300.0)
        self._sandbox_id: str | None = None
        self._session: str | None = None
        self._step_count: int = 0

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        if self._sandbox_id:
            try:
                self._post("stop", {"sandbox_id": self._sandbox_id})
            except Exception:
                logger.warning("Failed to stop sandbox during cleanup, ignoring")
            self._sandbox_id = None
        try:
            if hasattr(self, "_client") and self._client is not None:
                self._client.close()
        except Exception:
            pass

    def _post(self, path: str, data: dict) -> dict:
        response = self._client.post(f"{self._api_url}/{path}", json=data)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "Success":
            raise RuntimeError(f"API error at {path}: {body}")
        return body.get("result", {})

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = self._client.get(f"{self._api_url}/{path}", params=params)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "Success":
            raise RuntimeError(f"API error at {path}: {body}")
        return body.get("result", {})

    def reset(self, seed=None) -> tuple[str, dict]:
        Env.reset(self, seed)

        # Stop existing sandbox if any
        if self._sandbox_id:
            try:
                self._post("stop", {"sandbox_id": self._sandbox_id})
            except Exception:
                logger.warning("Failed to stop previous sandbox, ignoring")

        # Start new sandbox
        start_result = self._post("start_async", {
            "type": self.sandbox_type,
            "image": self.sandbox_image,
            "memory": self.memory,
            "cpus": self.cpus,
            "auto_clear_time": self.auto_clear_seconds / 60,
            "auto_clear_time_minutes": self.auto_clear_seconds / 60,
            "deployment_config": {},
        })
        self._sandbox_id = start_result["sandbox_id"]

        # Poll until alive
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            alive_result = self._get("is_alive", {"sandbox_id": self._sandbox_id})
            if alive_result.get("is_alive"):
                break
            time.sleep(2)
        else:
            raise TimeoutError(f"Sandbox {self._sandbox_id} did not become alive within 120s")

        # Create bash session
        session_name = f"cli-{int(time.time_ns())}"
        self._post("create_session", {
            "sandbox_id": self._sandbox_id,
            "session": session_name,
        })
        self._session = session_name

        self._step_count = 0

        # Prepare workspace
        obs = ""
        try:
            obs = self._run_command(f"mkdir -p {self.workspace_dir} && cd {self.workspace_dir} && pwd")
        except Exception:
            obs = "Ready."

        return obs, {"env_instruction": self.env_instruction}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        self._step_count += 1

        action_info = default_parser_action_func(
            action, self.action_pattern, None, self.special_token_list
        )
        command = action_info.get("action_content", "").strip()

        metrics_agg_mode = {
            "action_is_valid": "mean",
            "format_penalty": "mean",
        }

        if not command:
            metrics = {"action_is_valid": False, "format_penalty": self.format_penalty}
            return (
                "Invalid action: could not parse a command from your response.",
                self.format_penalty,
                False,
                self._step_count >= self.max_steps,
                {"metrics": metrics, "metrics_agg_mode": metrics_agg_mode, **action_info},
            )

        try:
            output = self._run_command(command)
        except Exception as e:
            metrics = {"action_is_valid": True, "format_penalty": self.format_penalty}
            return (
                f"Command execution failed: {e}",
                self.format_penalty,
                False,
                self._step_count >= self.max_steps,
                {"metrics": metrics, "metrics_agg_mode": metrics_agg_mode, **action_info},
            )

        metrics = {"action_is_valid": True, "format_penalty": 0.0}
        truncated = self._step_count >= self.max_steps
        return output, 0.0, False, truncated, {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            **action_info,
        }

    def _run_command(self, command: str, timeout: int = 30) -> str:
        result = self._post("run_in_session", {
            "action_type": "bash",
            "sandbox_id": self._sandbox_id,
            "session": self._session,
            "command": command,
            "timeout": timeout,
            "check": "ignore",
        })
        output = result.get("output", "")
        exit_code = result.get("exit_code", -1)
        if exit_code not in (None, 0):
            failure = result.get("failure_reason", "")
            output = f"{output}\n[exit code: {exit_code}] {failure}".strip()
        return output

    def sample_random_action(self) -> str:
        return "<answer>ls</answer>"

    def close(self):
        self._cleanup()
