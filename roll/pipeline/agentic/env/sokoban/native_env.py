from typing import Any, Dict, List, Tuple, SupportsFloat, Union

from roll.pipeline.agentic.env.sokoban.env import SokobanEnv
from roll.utils.constants import EpisodeStopReason
from roll.utils.logging import get_logger


class SokobanNativeEnv(SokobanEnv):
    """
    Sokoban environment for iflow native mode.

    This environment provides Sokoban puzzle functionality using the iflow native
    architecture. It's a simplified implementation that works with AgentNativeStepEnvManager
    without requiring external services like ROCK or iflow.
    """

    def __init__(
        self,
        group_id: int = 0,
        num_env_groups: int = 1,
        max_steps: int = 10,
        mode: str = "train",
        debug: bool = False,
        dim_room: Tuple[int, int] = (6, 6),
        num_boxes: int = 1,
        search_depth: int = 300,
        format_penalty: float = -0.1,
        action_pattern: str = "<answer>(.*?)</answer>",
        system_template: str = None,
        observation_suffix: str = None,
        **kwargs
    ):
        """
        Initialize Sokoban native environment.
        """
        # Store environment parameters
        self.group_id = group_id
        self.num_env_groups = num_env_groups
        self.mode = mode
        self.debug = debug

        # Runtime state
        self.current_step = 0
        self.task_idx = 0
        self.logger = get_logger()
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.env_reset_failed = False
        self.env_timeout = False
        self.failure_mode = ""
        self.stop_reason = ""
        self.error_messages = []
        self.test_output = ""
        self.is_closed = False

        # Message history for conversation
        self.message_history = []

        self.system_template = system_template
        if self.system_template is None:
            self.system_template = "You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game."

        # Initialize parent SokobanEnv
        super().__init__(
            render_mode="text",
            dim_room=dim_room,
            max_steps=max_steps,
            num_boxes=num_boxes,
            search_depth=search_depth,
            format_penalty=format_penalty,
            action_pattern=action_pattern,
            reset=False,
            **kwargs
        )
        self.observation_suffix = observation_suffix
        if self.observation_suffix is None:
            action_lookup_str = "\nYour available actions are:\n" + ", ".join(
                [f"{v}" for k, v in self.ACTION_LOOKUP.items()])
            self.observation_suffix = (f"\n\n<system-reminder>\nIMPORTANT: Ensure that your response is the format of '<answer> [your answer] </answer>',  with no extra text, eg. <answer>Right</answer>."
                                       f"{action_lookup_str}\n. </system-reminder>\n\n"
                                       f"Decide the next action:\n")

    def reset(self, seed=None) -> Tuple[List[Dict], Dict]:
        """
        Reset the environment and return initial observation.

        Returns:
            observation: List of messages for the agent
            info: Dictionary containing tools, error_msg, and failure_mode
        """
        super().reset(seed)
        self._clean_state()

        # Get the text observation from parent
        text_obs, env_info = super().reset(seed)

        # Initialize message history
        self.message_history = [
            {
                "role": "system",
                "content": f"{self.system_template}\n\n{env_info.get('env_instruction', self.get_instructions())}"
            },
            {
                "role": "user",
                "content": f"Here is the current state:\n{text_obs}\n\n{self.observation_suffix}"
            }
        ]

        # Return info with empty tools (Sokoban doesn't use tools)
        info = {
            "tools": [],
            "error_msg": "",
            "failure_mode": self.failure_mode
        }

        return self.message_history, info

    def step(self, action: str) -> Tuple[Union[List[Dict], str], SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action string from the agent

        Returns:
            observation: List of messages containing full conversation history
            reward: Step reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        self.current_step += 1
        # Check for control actions
        if isinstance(action, EpisodeStopReason):
            if action in [EpisodeStopReason.MAX_LENGTH, EpisodeStopReason.ENV_TIMEOUT]:
                self.terminated = True
                self.truncated = True
                self.stop_reason = action.name
                observation = self.message_history  # Return full history
                return observation, self.reward, True, True, {}

        # Add assistant's response to message history
        self.message_history.append({
            "role": "assistant",
            "content": action
        })

        # Execute the action using parent step method
        text_obs, reward, terminated, truncated, info = super().step(action)

        # Update state
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated

        # Add new user message with updated state to message history
        user_content = f"Current state:\n{text_obs}\n\n{self.observation_suffix}"
        if info.get("action_is_valid", False):
            user_content = (f"\n\n<system-reminder>\n(IMPORTANT TIPS: the last action is not valid, your new response *must* strictly adhere to the format according system-reminder.)</system-reminder>\n\n"
                           f"{user_content}")
        user_message = {
            "role": "user",
            "content": user_content
        }
        self.message_history.append(user_message)

        # Add metrics to info
        metrics = info.get("metrics", {})
        metrics.update({
            "env_timeout": self.env_timeout,
            "env_reset_failed": self.env_reset_failed,
            "success": self.boxes_on_target == self.num_boxes,
            "raw_reward": self.reward,
            "task_id": self.task_idx
        })

        metrics_agg_mode = info.get("metrics_agg_mode", {})
        info_new = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "failure_mode": self.failure_mode,
            "error_messages": self.error_messages,
            "stop_reason": self.stop_reason,
            "test_output": self.test_output
        }
        info.update(info_new)

        return self.message_history, self.reward, self.terminated, self.truncated, info

    def _clean_state(self):
        """Clean up state for new episode."""
        self.task_idx += 1
        self.current_step = 0
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.env_reset_failed = False
        self.env_timeout = False
        self.failure_mode = ""
        self.stop_reason = ""
        self.error_messages.clear()
        self.test_output = ""
        self.is_closed = False
        self.message_history = []  # Clear message history for new episode

    def close(self):
        """Close the environment."""
        super().close()
        self.is_closed = True

    @property
    def env_info(self) -> Dict:
        """Return environment information."""
        return {
            "task_idx": self.task_idx,
            "dim_room": self.dim_room,
            "num_boxes": self.num_boxes,
            "max_steps": self.max_steps,
            "current_step": self.current_step,
            "boxes_on_target": self.boxes_on_target,
        }

if __name__ == '__main__':

    env = SokobanNativeEnv(
        dim_room=(6, 6),
        num_boxes=2,
        max_steps=10,
    )

    print("=== SokobanNativeEnv Debug ===")

    # Reset environment
    obs, info = env.reset(seed=42)
    print("\n[Initial Observation]")
    print(f"Number of messages: {len(obs)}")
    print(f"System message: {obs[0]['content']}")
    print(f"User message: {obs[1]['content'][:200]}...")

    # Test some actions
    actions = [
        "<answer>Up</answer>",
        "<answer>Right</answer>",
        "<answer>Down</answer>",
        "<answer>Left</answer>",
        "<answer>Up</answer>",
    ]

    for i, action in enumerate(actions):
        print(f"\n=== Step {i+1} ===")
        print(f"Action: {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Success: {info.get('metrics', {}).get('success', False)}")
        print(f"Current step: {env.current_step}")
        print(f"Boxes on target: {env.boxes_on_target}/{env.num_boxes}")

        # Show last user message
        if obs:
            print(f"\nLatest observation:\n{obs[-1]['content']}")

        if terminated or truncated:
            print(f"\nEpisode ended! Reason: {info.get('stop_reason', 'Unknown')}")
            break

    # Test with invalid action
    print("\n=== Testing Invalid Action ===")
    obs, reward, terminated, truncated, info = env.step("invalid action")
    print(f"Invalid action reward: {reward}")
    print(f"Action valid: {info.get('metrics', {}).get('action_is_valid', False)}")

    # Show final environment info
    print("\n=== Final Environment Info ===")
    env_info = env.env_info
    for key, value in env_info.items():
        print(f"{key}: {value}")

    env.close()
    print("\n=== Debug Complete ===")
