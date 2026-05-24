"""OpenReward environment for ROLL agentic training.

Wraps the OpenReward sync SDK to implement the ``gem.Env`` interface expected by
:class:`AgentNativeStepEnvManager`.  Each episode opens an OpenReward session for
one task, collects tool-call interactions, and returns the terminal reward.

Usage in YAML config::

    custom_envs:
      MyEnvTag:
        env_type: "openreward_env"
        env_config:
          environment_name: "kanishk/EndlessTerminals"
          split: "train"
          max_steps: 16
"""
import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

from gem import Env

from roll.pipeline.agentic.env.openreward.tool_utils import (
    openreward_spec_to_qwen_tool,
    parse_tool_call,
    reduce_rewards,
)
from roll.utils.constants import EpisodeStopReason

logger = logging.getLogger(__name__)


class OpenRewardEnv(Env):
    """ROLL environment backed by the OpenReward SDK.

    The model generates ``<tool_call>`` XML blocks.  This class parses them,
    forwards the call to :pymethod:`session.call_tool`, and wraps the result
    in ``<tool_response>`` for the next model turn.

    Args:
        environment_name: Fully-qualified OpenReward environment name
            (e.g. ``"kanishk/EndlessTerminals"``).
        split: Dataset split to draw tasks from (``"train"`` or ``"test"``).
        mode: ``"train"`` or ``"val"`` — mirrors the ROCK env convention.
        max_steps: Maximum tool-call turns per episode.
        system_prompt_template: Override the default system prompt template.
            Must contain a ``{tools}`` placeholder.
        reward_reduction: How to reduce per-step rewards (``"sum"``, ``"mean"``,
            ``"max"``, ``"min"``).
        nonterminal_reward: Penalty added when the episode truncates without
            reaching a terminal state.  ``None`` means no penalty.
        retry_max_attempts: Number of session-creation retries on transient errors.
        retry_backoff_seconds: Base backoff between retries (doubles each attempt).
    """

    def __init__(
        self,
        environment_name: str,
        split: str = "train",
        mode: str = "train",
        max_steps: int = 16,
        system_prompt_template: Optional[str] = None,
        reward_reduction: str = "sum",
        nonterminal_reward: Optional[float] = None,
        retry_max_attempts: int = 3,
        retry_backoff_seconds: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._environment_name = environment_name
        self._split = split
        self._mode = mode
        self._max_steps = max_steps
        self._system_prompt_template = system_prompt_template
        self._reward_reduction = reward_reduction
        self._nonterminal_reward = nonterminal_reward
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds

        # --- SDK handles (lazy: created once in __init__) ---
        from openreward import OpenReward

        api_key = os.environ.get("OPENREWARD_API_KEY", "")
        self._client = OpenReward(api_key=api_key) if api_key else OpenReward()
        self._or_env = self._client.environments.get(name=environment_name)
        self._num_tasks: int = self._or_env.num_tasks(split)
        logger.info(
            "[OpenRewardEnv] Connected to %s — %d %s tasks",
            environment_name, self._num_tasks, split,
        )

        # --- Episode state (reset each episode) ---
        self._session: Any = None
        self._message_history: List[Dict[str, str]] = []
        self._step_rewards: List[float] = []
        self._task_index: int = -1
        self.current_step: int = 0
        self._num_tool_calls: int = 0
        self._num_failed_tool_calls: int = 0
        self._finished: bool = False

        # --- Flags read by the env manager ---
        self.env_reset_failed: bool = False
        self.env_timeout: bool = False

    # ------------------------------------------------------------------
    # gem.Env interface
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Open an OpenReward session and return the initial conversation.

        Args:
            seed: Used to deterministically select a task index.

        Returns:
            ``(observation, info)`` where *observation* is a list of message
            dicts (system + user prompt) and *info* contains ``tools``,
            ``error_msg``, and ``failure_mode`` keys.
        """
        super().reset(seed=seed)
        self._clean_state()

        # Derive a deterministic task index from the seed
        if seed is not None:
            self._task_index = seed % self._num_tasks
        else:
            self._task_index = 0

        # Open session with retry logic
        if not self._open_session():
            # Session creation failed — signal to env manager
            return [], {
                "tools": [],
                "error_msg": "Session creation failed after retries",
                "failure_mode": "session_creation_failed",
            }

        # Fetch tools and prompt from the live session
        try:
            raw_tools = self._session.list_tools()
            prompt_blocks = self._session.get_prompt()
        except Exception as exc:
            logger.error("[OpenRewardEnv] Failed to get tools/prompt: %s", exc)
            self.env_reset_failed = True
            self._close_session()
            return [], {
                "tools": [],
                "error_msg": str(exc),
                "failure_mode": "tools_or_prompt_failed",
            }

        # Convert OpenReward tool specs to Qwen-native dict format.
        # These are passed via info["tools"] → env_manager → tokenizer.apply_chat_template(tools=...)
        # The tokenizer's Jinja2 template builds the system prompt automatically with the correct
        # tool-call format that the model was trained on (<function=name><parameter=key>...</parameter>).
        self._qwen_tools = [openreward_spec_to_qwen_tool(t) for t in raw_tools]

        user_text = "".join(
            b.text for b in prompt_blocks if b.type == "text"
        )

        # No manual system prompt needed — the tokenizer builds it from tools=.
        # We only provide the user message with the task prompt.
        self._message_history = [
            {"role": "user", "content": user_text},
        ]

        # --- Observability: log reset ---
        logger.info(
            "[OBSERVE][ENV_RESET] task_index=%d tools=[%s] prompt=%.200s",
            self._task_index,
            ", ".join(t["function"]["name"] for t in self._qwen_tools),
            user_text[:200].replace("\n", "\\n"),
        )

        return copy.deepcopy(self._message_history), {
            "tools": self._qwen_tools,
            "error_msg": "",
            "failure_mode": "",
            "task_name": f"{self._environment_name}:{self._split}:{self._task_index}",
        }

    def step(
        self, action: Union[str, Any],
    ) -> Tuple[List[Dict[str, str]], SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one tool-call turn.

        Args:
            action: The model's decoded text output, or an
                :class:`EpisodeStopReason` for forced termination.

        Returns:
            ``(observation, reward, terminated, truncated, info)``
        """
        self.current_step += 1

        # Handle forced-termination actions from the env manager
        if isinstance(action, EpisodeStopReason):
            reward = self._compute_final_reward(reached_terminal=False)
            info = self._build_info(stop_reason=action.value)
            self._close_session()
            return copy.deepcopy(self._message_history), reward, True, True, info

        # Enforce max_steps internally (env manager expects this)
        if self.current_step > self._max_steps:
            reward = self._compute_final_reward(reached_terminal=False)
            info = self._build_info(stop_reason="max_steps")
            self._close_session()
            return copy.deepcopy(self._message_history), reward, True, True, info

        # Clean trailing special tokens from the model output
        clean_action = action.replace("<|im_end|>", "").rstrip()

        # --- Observability: log model action ---
        logger.info(
            "[OBSERVE][ENV_ACTION] step=%d has_tool_call=%s has_close_tag=%s action=%.500s",
            self.current_step, "<tool_call>" in clean_action, "</tool_call>" in clean_action,
            clean_action.replace("\n", "\\n"),
        )

        # Append the assistant's message to history
        self._message_history.append({"role": "assistant", "content": clean_action})

        # Parse the tool call
        tc = parse_tool_call(clean_action)

        if tc is None:
            # No tool call found — nudge the model
            nudge = (
                "No tool call detected in your response. "
                "Please use the provided tools with <tool_call>...</tool_call> format "
                "to complete the task."
            )
            self._message_history.append({"role": "user", "content": nudge})
            info = self._build_info(stop_reason="no_tool_call")
            return copy.deepcopy(self._message_history), 0.0, False, False, info

        if tc["type"] == "error":
            # Parse error — nudge with the error message
            error_nudge = (
                f"Tool call parse error: {tc['error']}. "
                "Please ensure arguments are valid JSON within "
                "<tool_call>...</tool_call> tags."
            )
            self._message_history.append({"role": "user", "content": error_nudge})
            self._num_failed_tool_calls += 1
            info = self._build_info(stop_reason="parse_error")
            return copy.deepcopy(self._message_history), 0.0, False, False, info

        # Valid tool call — execute it
        self._num_tool_calls += 1

        # --- Observability: log tool call request ---
        logger.info(
            "[OBSERVE][TOOL_CALL] step=%d name=%s arguments=%s",
            self.current_step, tc["name"],
            str(tc["arguments"])[:300].replace("\n", "\\n"),
        )

        tool_text, finished = self._execute_tool_call(tc["name"], tc["arguments"])

        # --- Observability: log tool response ---
        logger.info(
            "[OBSERVE][TOOL_RESPONSE] step=%d name=%s finished=%s reward=%s output=%.200s",
            self.current_step, tc["name"], finished,
            self._step_rewards[-1] if self._step_rewards else None,
            tool_text[:200].replace("\n", "\\n"),
        )

        # Append tool response as user message
        self._message_history.append({
            "role": "user",
            "content": f"<tool_response>\n{tool_text}\n</tool_response>",
        })

        # Determine termination and reward
        terminated = finished
        reward = 0.0
        if terminated:
            reward = self._compute_final_reward(reached_terminal=True)
            self._close_session()

        info = self._build_info(
            stop_reason="finished" if terminated else "continue",
        )
        return copy.deepcopy(self._message_history), reward, terminated, False, info

    def close(self) -> None:
        """Release the OpenReward session if still open."""
        self._close_session()

    @property
    def env_info(self) -> Dict[str, Any]:
        """Task metadata used by ``formulate_rollouts`` for trajectory logging."""
        return {
            "environment_name": self._environment_name,
            "task_index": self._task_index,
            "split": self._split,
            "current_step": self.current_step,
            "num_tool_calls": self._num_tool_calls,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_state(self) -> None:
        """Reset all per-episode state."""
        self._close_session()
        self._message_history = []
        self._step_rewards = []
        self._task_index = -1
        self.current_step = 0
        self._num_tool_calls = 0
        self._num_failed_tool_calls = 0
        self._finished = False
        self.env_reset_failed = False
        self.env_timeout = False

    def _open_session(self) -> bool:
        """Open an OpenReward session with retry + exponential backoff.

        Returns:
            ``True`` if a session was opened, ``False`` on failure.
        """
        backoff = self._retry_backoff_seconds
        for attempt in range(self._retry_max_attempts + 1):
            try:
                self._session = self._or_env.session(
                    split=self._split, index=self._task_index,
                )
                self._session.__enter__()
                logger.info(
                    "[OpenRewardEnv] Session opened: %s split=%s index=%d (attempt %d)",
                    self._environment_name, self._split, self._task_index, attempt,
                )
                return True
            except Exception as exc:
                logger.warning(
                    "[OpenRewardEnv] Session creation failed (attempt %d/%d): %s",
                    attempt + 1, self._retry_max_attempts + 1, exc,
                )
                if attempt < self._retry_max_attempts:
                    time.sleep(backoff)
                    backoff *= 2

        self.env_reset_failed = True
        return False

    def _close_session(self) -> None:
        """Safely close the active session."""
        if self._session is not None:
            try:
                self._session.__exit__(None, None, None)
            except Exception as exc:
                logger.debug("[OpenRewardEnv] Error closing session: %s", exc)
            self._session = None

    def _execute_tool_call(
        self, name: str, arguments: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """Call a tool on the OpenReward session.

        Returns:
            ``(tool_output_text, finished)``
        """
        try:
            tool_out = self._session.call_tool(name, arguments)
            tool_text = "".join(
                b.text for b in tool_out.blocks if b.type == "text"
            )
            if tool_out.reward is not None:
                self._step_rewards.append(tool_out.reward)
            return tool_text, tool_out.finished
        except Exception as exc:
            logger.warning("[OpenRewardEnv] Tool call failed (%s): %s", name, exc)
            self._num_failed_tool_calls += 1
            return f"Error executing tool '{name}': {exc}", False

    def _compute_final_reward(self, reached_terminal: bool) -> float:
        """Compute the episode reward from collected step rewards."""
        rewards = list(self._step_rewards)
        if not reached_terminal and self._nonterminal_reward is not None:
            rewards.append(self._nonterminal_reward)
        return reduce_rewards(rewards, self._reward_reduction)

    def _build_info(self, stop_reason: str = "") -> Dict[str, Any]:
        """Build the info dict returned by ``step()``."""
        metrics = {
            "env_timeout": self.env_timeout,
            "env_reset_failed": self.env_reset_failed,
            "success": any(r > 0 for r in self._step_rewards),
            "raw_reward": self._compute_final_reward(reached_terminal=True),
            "current_step": self.current_step,
            "num_tool_calls": self._num_tool_calls,
            "num_failed_tool_calls": self._num_failed_tool_calls,
        }
        metrics_agg_mode = {
            "success": "last",
            "raw_reward": "last",
        }
        return {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "failure_mode": "",
            "error_messages": [],
            "stop_reason": stop_reason,
            "test_output": "",
        }
