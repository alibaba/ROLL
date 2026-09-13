from typing import Any, Dict, List, Union

import httpx
from omegaconf import DictConfig

from roll.pipeline.agentic.agent_runner.base import AgentRunner, EpisodeResult
from roll.pipeline.agentic.env import gem
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field


class GEMRunner(AgentRunner):
    """AgentRunner for local gem.Env environments.

    The gem.Env (e.g. SokobanEnv) returns raw text observations and env
    instructions.  GEMRunner is responsible for constructing the OpenAI
    message history and driving the interaction loop:

        env.reset(seed) -> (obs_text, info)
        loop:
            build messages from history
            LLM request -> response
            env.step(action_text) -> (obs_text, reward, terminated, truncated, info)

    Applicable to: Sokoban, FrozenLake, and other gym-like environments.
    """

    def __init__(self, base_url: str, env_id: int, env_config: DictConfig, **kwargs):
        super().__init__(base_url, env_id, env_config, **kwargs)
        self.logger = get_logger()
        self.max_steps: int = env_config.get("max_steps", 10)
        self.http_timeout: float = float(env_config.get("http_timeout", 3600.0))
        self.system_template: str = env_config["agent_system_template"]
        self.agent_template = env_config["agent_template"]
        self.setup()

    def setup(self) -> None:
        """Create a gem.Env instance from config.

        - ``env_type`` comes from the top-level env_config (e.g. "sokoban").
        - env constructor params come from ``env_config["config"]``
          (e.g. dim_room, num_boxes, search_depth).
        - If ``tool_wrapper`` is configured, wraps the env for text-based tool support.
        """
        env_type = self.env_config.get("env_type", "sokoban")
        self.env = gem.make(env_id=env_type, **self.env_params)
        if "tool_wrapper" in self.env_config:
            from roll.pipeline.agentic.tools.tool_env_wrapper import tool_wrapper
            self.env = tool_wrapper(
                self.env,
                wrapper_args=self.env_config.tool_wrapper.wrapper_args,
                tool_configs=self.env_config.tool_wrapper.tool_configs,
            )

    def _render_observation(self, obs_text: str, turn_idx: int, actions_left: int) -> str:
        """Render observation text through the agent_template with optional fields."""
        render_dict: Dict[str, Any] = {"observation": obs_text}
        if contains_renderable_field(self.agent_template, "turn_idx"):
            render_dict["turn_idx"] = turn_idx
        if contains_renderable_field(self.agent_template, "suffix"):
            render_dict["suffix"] = ""
        if contains_renderable_field(self.agent_template, "actions_left"):
            render_dict["actions_left"] = actions_left
        if contains_renderable_field(self.agent_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config.get("max_tokens_per_step", 128)
        return self.agent_template.format(**render_dict)

    def run_job(self, seed: int) -> EpisodeResult:
        """Run a complete text-mode episode with the local gem.Env.

        If ``tool_wrapper`` is configured, the wrapper transparently intercepts
        text actions and executes tools — no change to this loop.
        """
        obs_text, info = self.env.reset(seed=seed)
        if obs_text is None:
            return EpisodeResult(status="NoData", score=0.0)

        env_instruction = info.get("env_instruction", "")

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (f"{self.system_template}\n\n{env_instruction}"
                            if env_instruction else self.system_template),
            },
            {
                "role": "user",
                "content": self._render_observation(obs_text, turn_idx=1, actions_left=self.max_steps),
            },
        ]

        rewards: List[float] = []
        try:
            with httpx.Client(timeout=self.http_timeout) as client:
                for turn_index in range(1, self.max_steps+1):
                    resp_json = self._llm_request(client, messages)

                    if "error" in resp_json:
                        break

                    assistant_content = resp_json["choices"][0]["message"].get("content", "")
                    messages.append({"role": "assistant", "content": assistant_content})

                    obs_text, reward, terminated, truncated, _ = self.env.step(assistant_content)
                    rewards.append(reward)

                    if terminated or truncated:
                        break

                    user_content = self._render_observation(
                        obs_text,
                        turn_idx=turn_index + 1,
                        actions_left=self.max_steps - turn_index,
                    )
                    messages.append({"role": "user", "content": user_content})
        except httpx.HTTPError as e:
            self.logger.error(f"[GEMRunner] HTTP error at step {len(rewards)}: {e}")
            return EpisodeResult(
                status="Failed",
                score=0.0,
                step_scores=rewards,
                agent_exit_reason=f"HTTP error: {e}",
            )

        return EpisodeResult(
            status="Finished",
            score=float(sum(rewards)),
            step_scores=rewards,
        )

    def teardown(self) -> None:
        if self.env is not None:
            self.env.close()


class ToolCallRunner(GEMRunner):
    """AgentRunner for gem.Env environments using OpenAI function-calling protocol.

    Unlike GEMRunner (text-in/text-out), this runner:
    - Passes ``tools`` from ``info`` to every LLM request.
    - When the LLM responds with ``tool_calls``, passes the full message dict
      to ``env.step(message)``; otherwise passes plain text.
    - Lets the env manage the conversation history — ``env.step()`` returns
      ``(messages, reward, terminated, truncated, info)``.

    Applicable to: SokobanToolCallEnv and other envs that handle tool
    execution internally.
    """

    def run_job(self, seed: int) -> EpisodeResult:
        """Run a complete tool-call episode with the local gem.Env."""
        messages, info = self.env.reset(seed=seed)
        if messages is None:
            return EpisodeResult(status="NoData", score=0.0)
        assert isinstance(messages, list), (
            f"ToolCallRunner requires env.reset() to return a message list, got {type(messages)}"
        )
        tools = info.get("tools") or None

        rewards: List[float] = []
        try:
            with httpx.Client(timeout=self.http_timeout) as client:
                for _ in range(self.max_steps):
                    resp_json = self._llm_request(client, messages, tools=tools)

                    if "error" in resp_json:
                        break

                    message = resp_json["choices"][0]["message"]

                    # Tool calls → pass full message dict; text → pass content string
                    if message.get("tool_calls"):
                        action: Union[Dict[str, Any], str] = message
                    else:
                        action = message.get("content", "")

                    # env executes tool calls internally and returns updated messages
                    messages, reward, terminated, truncated, _ = self.env.step(action)
                    rewards.append(reward)

                    if terminated or truncated:
                        break
        except httpx.HTTPError as e:
            self.logger.error(f"[ToolCallRunner] HTTP error at step {len(rewards)}: {e}")
            return EpisodeResult(
                status="Failed",
                score=0.0,
                step_scores=rewards,
                agent_exit_reason=f"HTTP error: {e}",
            )

        return EpisodeResult(
            status="Finished",
            score=float(sum(rewards)),
            step_scores=rewards,
        )
