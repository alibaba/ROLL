import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from atroposlib.envs.base import BaseEnv, ScoredDataGroup
from atroposlib.envs.server_handling.server_baseline import APIServer, APIServerConfig
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice

logger = logging.getLogger(__name__)

class RolloutTurnBoundary(Exception):
    """Exception raised to signal a turn boundary in the controlled rollout."""
    def __init__(self, observation: Union[str, List[Dict[str, Any]]], metadata: Dict[str, Any] = None):
        self.observation = observation
        self.metadata = metadata or {}

class AtroposExecutionBridge(APIServer):
    """
    Internal execution adapter that bridges ROLL actions into Atropos trajectories.
    It provides the action from ROLL and collects the environment's reaction 
    for the next step.
    """
    def __init__(self, action: str, history: List[Dict[str, Any]], debug: bool = False, reward_config: Dict = None):
        # We don't need a real config for this mock
        super().__init__(APIServerConfig(model_name="mock", base_url="mock", api_key="x"))
        self.action = action
        self.initial_history_len = len(history)
        self.call_count = 0
        self.debug = debug

    async def check_server_status_task(self, chat_completion: bool = True):
        """Mock health check."""
        self.server_healthy = True

    async def _tokens_and_logprobs_completion_wrapper(self, **kwargs) -> Any:
        self.call_count += 1
        prompt = kwargs.get("prompt", "")
        
        if self.debug:
            print(f"[AtroposExecutionBridge] Call {self.call_count} | Prompt len: {len(prompt)}")

        # First call: Provide the response from the ROLL assistant.
        if self.call_count == 1:
            # Generate tokens for the provided action.
            mock_tokens = [ord(c) for c in self.action]
            mock_logprobs = [0.0] * len(mock_tokens)
            if self.debug:
                print(f"[AtroposExecutionBridge] Providing action tokens: {self.action}")
            return ([0], [mock_tokens], [mock_logprobs], ["stop"])
        
        # Subsequent calls: Signal that we've reached a new turn boundary.
        else:
            # Extract the new observation from the prompt.
            observation = prompt
            if self.debug:
                print(f"[AtroposExecutionBridge] Signalling Turn Boundary. New observation captured.")
            raise RolloutTurnBoundary(observation)

    async def _completion_wrapper(self, **kwargs) -> Any:
        raise NotImplementedError("Completion not supported in AtroposExecutionBridge")

    async def _chat_completion_wrapper(self, **kwargs) -> Any:
        """Fallback for non-managed calls."""
        self.call_count += 1
        messages = kwargs.get("messages", [])
        if self.call_count == 1:
            return self._create_chat_completion(self.action)
        else:
            observation = messages[self.initial_history_len:]
            raise RolloutTurnBoundary(observation)

    async def _get_logprobs_wrapper(self, **kwargs) -> Dict[str, Any]:
        return {
            "prompt_tokens": [],
            "prompt_topk_token_ids": [],
            "prompt_topk_logprobs": []
        }

    def _create_chat_completion(self, content: str) -> ChatCompletion:
        import uuid
        import time
        from openai.types.chat.chat_completion import Choice, ChatCompletionMessage
        
        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content=content, role="assistant"),
                )
            ],
            created=int(time.time()),
            model="mock",
            object="chat.completion",
        )

async def execute_controlled_rollout(
    env: BaseEnv,
    item: Any,
    action: str,
    history: List[Dict[str, Any]],
    debug: bool = False, reward_config: Dict = None
) -> Tuple[Union[str, List[Dict[str, Any]]], float, bool, Dict[str, Any]]:
    """
    Executes a controlled segment of an Atropos trajectory.
    
    This function bridges a single ROLL step into the Atropos trajectory-based engine
    by running a rollout until either the trajectory terminates or a new 
    turn boundary is reached.
    """
    
    # 1. Attach the execution bridge
    original_servers = env.server.servers
    execution_bridge = AtroposExecutionBridge(action, history, debug=debug)
    env.server.servers = [execution_bridge]
    
    try:
        if debug:
            logger.info(f"[AtroposBridge] Executing controlled rollout. History: {len(history)}")
            
        # 2. Trigger Atropos environment logic
        result, _ = await env.collect_trajectories(item)
        
        # 3. Trajectory finished naturally — extract Atropos math score
        atropos_reward = 0.0
        if result and isinstance(result, (dict, ScoredDataGroup)) and "scores" in result:
            if len(result["scores"]) > 0:
                atropos_reward = float(result["scores"][0])

        # 4. Compute Universal Bridge Reward
        if atropos_reward > 0:
            reward = atropos_reward
        else:
            format_bonus = 0.0
            
            # Default to reasoning tags if no config (backward compatibility)
            markers = [
                {"marker": "<think>", "reward": 0.2},
                {"marker": "\\boxed{", "reward": 0.3},
            ]
            
            # Override with YAML config if provided
            length_bounty_max = 0.2
            if reward_config:
                markers = reward_config.get("format_markers", markers)
                length_bounty_max = reward_config.get("length_bounty_max", length_bounty_max)

            # Check markers
            for bonus_item in markers:
                if bonus_item["marker"] in action:
                    format_bonus += bonus_item["reward"]
            
            # Continuous Length component (CRITICAL for GRPO variance)
            length_bonus = min(len(action) / 1000.0, length_bounty_max)
            reward = -1.0 + format_bonus + length_bonus

        if debug:
            logger.info(f"[AtroposBridge] Rollout complete (Traj End). Atropos: {atropos_reward}, Final: {reward}")
            
        return "", reward, True, {"result": result}
        
    except RolloutTurnBoundary as e:
        # 4. A new turn boundary was reached
        if debug:
            logger.info(f"[AtroposBridge] Rollout complete (Turn Boundary). Observation captured.")
            
        return e.observation, 0.0, False, e.metadata
        
    except Exception as e:
        logger.error(f"Error during partial trajectory execution: {e}")
        raise
    finally:
        # Restore original servers
        env.server.servers = original_servers
