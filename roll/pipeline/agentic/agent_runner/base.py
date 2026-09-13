from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx
from omegaconf import DictConfig

if TYPE_CHECKING:
    from roll.pipeline.agentic.agentic_config import EnvManagerConfig


class EpisodeResult:
    """Structured result from a single episode execution."""

    def __init__(
        self,
        status: str,
        score: float,
        step_scores: Optional[List[float]] = None,
        agent_exit_reason: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.status = status
        self.score = score
        self.step_scores = step_scores or []
        self.agent_exit_reason = agent_exit_reason
        self.metrics = metrics or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "step_scores": self.step_scores,
            "agent_exit_reason": self.agent_exit_reason,
            **self.metrics,
        }


class AgentRunner(ABC):
    """Agent interaction loop abstraction.

    AgentRunner encapsulates how an agent interacts with an environment and
    calls the LLM.  It does NOT concern itself with:
    - Trajectory collection (handled transparently by ProxyServer / MessageTracker)
    - Training sample construction (handled by EnvManager.formulate_rollouts)
    - Episode scheduling (handled by run_rollout_loop outer loop)

    Subclasses implement ``run_job(seed)``: load data, run a full episode,
    return an ``EpisodeResult``.
    """

    def __init__(
        self,
        base_url: str,
        env_id: int,
        env_config: DictConfig,
        worker_config: Optional["EnvManagerConfig"] = None,
        **kwargs,
    ):
        """
        Args:
            base_url: LLM inference service URL.  For Proxy mode this is the
                local ProxyServer address (e.g. http://127.0.0.1:8000).
            env_id: Environment instance ID, fixed for the runner's lifetime.
                Used by ProxyServer for routing (as Authorization Bearer token).
            env_config: Full environment configuration DictConfig.
                Contains top-level keys (env_type, max_steps, agent_runner_cls, ...)
                and nested ``config`` / ``env_config`` dicts.  Subclasses access
                ``env_config.config`` for runner-level settings and
                ``env_config["env_config"]`` for env constructor params.
            worker_config: EnvManager-level configuration (model_args, generating_args, etc.).
        """
        self.base_url = base_url
        self.env_id = env_id
        self.env_config: DictConfig = env_config
        self.worker_config = worker_config
        self.env: Optional[Any] = None
        self.env_params: Dict[str, Any] = {}
        if "config" in self.env_config:
            self.env_params = dict(self.env_config["config"])
        self.mode: str = self.env_params.get("mode", "train")

    @abstractmethod
    def run_job(self, seed: int) -> EpisodeResult:
        """Run a complete episode.

        Args:
            seed: Episode seed for data loading and env initialisation.

        Returns:
            EpisodeResult with score, step_scores, and metrics.
        """
        ...

    def setup(self) -> None:
        """One-time initialisation (create env, establish connections, etc.)."""
        pass

    def teardown(self) -> None:
        """Clean up resources."""
        pass

    def _llm_request(
        self,
        client: httpx.Client,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """Send an OpenAI-compatible chat completion request to ``base_url``.

        The request is intercepted by ProxyServer, which routes it via the
        ``Authorization`` header's ``self.env_id`` to the corresponding
        ``EnvManager.process_request`` for inference + trajectory recording.
        """
        resp = client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "tools": tools or [],
                "tool_choice": tool_choice if tools else "none",
            },
            headers={"Authorization": f"Bearer {self.env_id}"},
        )
        resp.raise_for_status()
        return resp.json()
