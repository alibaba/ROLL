"""
Tool call variant of SokobanNativeEnv for fork detection testing.

Exposes Sokoban movements as OpenAI-compatible function tools instead of text-based actions.
Supports full_history mode: when enabled, invalid tool calls trigger MessageTracker fork
detection by removing the invalid assistant+tool message pair from history.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, SupportsFloat, Union

from roll.pipeline.agentic.env.sokoban.native_env import SokobanNativeEnv
from roll.utils.logging import get_logger


class SokobanToolCallEnv(SokobanNativeEnv):
    """
    Tool call variant of SokobanNativeEnv with fork detection support.

    This environment exposes Sokoban movements as function tools (OpenAI format) instead
    of text-based <answer> tags. When full_history=True, invalid tool calls are removed
    from message history to trigger MessageTracker fork detection.

    Args:
        full_history: If True, remove invalid tool call rounds from history to trigger
                     MessageTracker fork detection. Default False.
        **kwargs: Passed to SokobanNativeEnv parent class.

    Example:
        >>> env = SokobanToolCallEnv(full_history=True, dim_room=(6, 6), num_boxes=1, max_steps=10)
        >>> messages, info = env.reset(seed=42)
        >>> tools = info['tools']  # OpenAI tool schema
        >>>
        >>> # Valid tool call
        >>> action = {
        ...     "role": "assistant",
        ...     "tool_calls": [{
        ...         "id": "call_1",
        ...         "type": "function",
        ...         "function": {"name": "move_player", "arguments": '{"direction": "right"}'}
        ...     }]
        ... }
        >>> messages, reward, done, trunc, info = env.step(action)
        >>> # messages now includes assistant + tool result
        >>>
        >>> # Invalid tool call (direction not in VALID_DIRECTIONS)
        >>> action = {
        ...     "role": "assistant",
        ...     "tool_calls": [{
        ...         "id": "call_2",
        ...         "type": "function",
        ...         "function": {"name": "move_player", "arguments": '{"direction": "invalid"}'}
        ...     }]
        ... }
        >>> messages, reward, done, trunc, info = env.step(action)
        >>> # With full_history=True, assistant+tool are removed, user warning added
        >>> # This triggers fork detection in MessageTracker
    """

    def __init__(
        self,
        full_history: bool = False,
        **kwargs
    ):
        """
        Initialize tool call environment.

        Args:
            full_history: If True, remove invalid tool call rounds from history.
            **kwargs: Passed to SokobanNativeEnv parent.
        """
        # Use a dummy pattern that won't match (will be overridden anyway)
        if 'action_pattern' not in kwargs:
            kwargs['action_pattern'] = r'__TOOL_CALL_MODE__'
        
        env_instruction = (
            "You are solving the Sokoban puzzle. "
            "You are the player and you need to push all boxes to targets. "
            "When you are right next to a box, you can push it by moving in the same direction. "
            "You cannot push a box through a wall, and you cannot pull a box. "
            "Use the move_player tool to move the player."
        )
        super().__init__(env_instruction=env_instruction, **kwargs)

        self.full_history = full_history
        self.logger = get_logger()

        # Valid directions for parameter-level validation
        self.VALID_DIRECTIONS = ["up", "down", "left", "right"]

        # Tool definitions (returned in reset())
        self._tools = self._create_tool_definitions()

        self.observation_suffix = ""

    def _create_tool_definitions(self) -> List[Dict]:
        """
        Create OpenAI-compatible tool definitions for Sokoban movements.

        Returns:
            List of tool definitions in OpenAI format.
        """
        return [{
            "type": "function",
            "function": {
                "name": "move_player",
                "description": "Move the player in the specified direction in the Sokoban game",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Direction to move: up, down, left, or right"
                        }
                    },
                    "required": ["direction"]
                }
            }
        }]

    def reset(self, seed=None) -> Tuple[List[Dict], Dict]:
        """
        Reset the environment and return initial observation with tools.

        Args:
            seed: Random seed for environment reset.

        Returns:
            Tuple of (messages, info) where:
            - messages: Initial conversation with system + user messages
            - info: Dictionary containing tools and other metadata
        """
        messages, info = super().reset(seed)

        # Override system message to instruct tool usage
        system_content = (
            f"{self.system_template}\n\n"
            f"{info.get('env_instruction', self.get_instructions())}\n\n"
            "IMPORTANT: Use the move_player tool to take actions. "
            "Do NOT use text-based <answer> tags. "
            "Always call the tool with a valid direction parameter."
        )
        messages[0]['content'] = system_content

        # Add tools to info dict for ProxyEnvManager
        info['tools'] = self._tools

        return messages, info

    def step(
        self,
        action: Union[str, Dict]
    ) -> Tuple[Union[List[Dict], str], SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute a tool call action.

        Args:
            action: Either:
                - Dict with OpenAI message format: {"role": "assistant", "tool_calls": [...]}
                - String (fallback for compatibility, will trigger format error)

        Returns:
            Tuple of (messages, reward, terminated, truncated, info):
            - messages: Updated conversation history
            - reward: Step reward (may include format penalty)
            - terminated: Whether episode is done (max steps)
            - truncated: Whether episode was cut short
            - info: Metrics and metadata
        """
        self.current_step += 1

        # Parse tool call from action
        tool_call, parsed_action_text, is_valid_structure = self._parse_tool_call(action)

        if not is_valid_structure:
            # Invalid structure: format penalty, no history modification
            return self._handle_invalid_format(action)

        # Validate direction parameter (parameter-level validation only)
        is_valid_direction = self._is_valid_direction_param(tool_call)

        # Add assistant message with tool call
        assistant_msg = {
            "role": "assistant",
            "content": action.get("content", "") if isinstance(action, dict) else action,
            "tool_calls": [tool_call] if tool_call else []
        }
        self.message_history.append(assistant_msg)

        # Execute the action using parent's game logic (if direction valid)
        if is_valid_direction:
            message_history = self.message_history
            self.message_history = []
            _, reward, terminated, truncated, info = super().step(parsed_action_text)
            # Use action_is_effective (player position changed) rather than action_is_valid (format valid)
            action_is_effective = info.get("metrics", {}).get("action_is_effective", False)
            self.message_history = message_history
            text_obs = self.render(mode="text")
        else:
            # Invalid direction parameter: apply format penalty, no state change
            text_obs = self.render(mode="text")
            reward = self.format_penalty
            terminated = False
            truncated = False
            action_is_effective = False
            info = self._create_invalid_info()

        # Add tool result message
        # is_valid means: direction param valid AND action was effective (position changed)
        tool_result_msg = {
            "role": "tool",
            "tool_call_id": tool_call["id"] if tool_call else "unknown",
            "content": self._format_tool_result(text_obs, info, is_valid_direction and action_is_effective)
        }
        self.message_history.append(tool_result_msg)

        # FORK DETECTION TRIGGER: Remove invalid round if full_history enabled
        if self.full_history and not is_valid_direction:
            self.logger.info(
                f"[SokobanToolCall] Removing invalid tool call round (step {self.current_step}) to trigger fork detection"
            )
            # Remove last 2 messages (assistant + tool)
            self.message_history = self.message_history[:-2]

            # Add warning message to inform LLM
            warning_msg = {
                "role": "user",
                "content": (
                    f"\n\n<system-reminder>\n"
                    f"IMPORTANT: The last tool call was invalid (direction parameter must be one of up/down/left/right). "
                    f"That conversation round has been removed. Please retry with correct parameters.\n"
                    f"</system-reminder>\n\n"
                    f"Current game state:\n{text_obs}\n\n"
                    f"{self.observation_suffix}"
                )
            }
            self.message_history.append(warning_msg)
        else:
            # Normal flow: add new observation
            user_msg = {
                "role": "user",
                "content": f"Current game state:\n{text_obs}\n\n{self.observation_suffix}"
            }
            self.message_history.append(user_msg)

        # Update info with fork detection flag
        if 'metrics' not in info:
            info['metrics'] = {}
        # Add metrics aggregation mode
        if 'metrics_agg_mode' not in info:
            info['metrics_agg_mode'] = {}

        return self.message_history, reward, terminated, truncated, info

    def _parse_tool_call(
        self,
        action: Union[str, Dict]
    ) -> Tuple[Optional[Dict], str, bool]:
        """
        Parse action into tool call dict and text for parent validation.

        Args:
            action: Action from LLM (dict with tool_calls or string)

        Returns:
            Tuple of (tool_call_dict, action_text_for_parent, is_valid_structure):
            - tool_call_dict: Extracted tool call or None
            - action_text_for_parent: Text format for parent's step() method
            - is_valid_structure: Whether the tool call structure is valid
        """
        # Case 1: Dict format (expected)
        if isinstance(action, dict):
            tool_calls = action.get("tool_calls", [])
            if not tool_calls:
                self.logger.warning("[SokobanToolCall] Received dict without tool_calls")
                return None, "", False

            tool_call = tool_calls[0]  # Take first tool call

            # Validate tool call structure
            if not self._validate_tool_call_structure(tool_call):
                return None, "", False

            # Extract direction parameter
            func = tool_call.get("function", {})
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                direction = args.get("direction", "").lower()
            except json.JSONDecodeError:
                self.logger.error(f"[SokobanToolCall] JSON parse failed: {args_str}")
                return tool_call, "", False

            # Map to parent's expected format
            action_text = f"<answer>{direction.capitalize()}</answer>"
            return tool_call, action_text, True

        # Case 2: String format (fallback for backward compatibility)
        elif isinstance(action, str):
            self.logger.warning(
                f"[SokobanToolCall] Received string action (expected dict): {action[:100]}"
            )
            return None, action, False

        return None, "", False

    def _validate_tool_call_structure(self, tool_call: Dict) -> bool:
        """
        Validate tool call has required fields.

        Args:
            tool_call: Tool call dict to validate

        Returns:
            True if structure is valid, False otherwise
        """
        if not isinstance(tool_call, dict):
            return False
        if "id" not in tool_call or "function" not in tool_call:
            self.logger.warning("[SokobanToolCall] tool_call missing required fields (id or function)")
            return False
        func = tool_call["function"]
        if "name" not in func:
            self.logger.warning("[SokobanToolCall] function missing name field")
            return False
        if func["name"] != "move_player":
            self.logger.warning(f"[SokobanToolCall] Unknown tool name: {func['name']}")
            return False
        return True

    def _is_valid_direction_param(self, tool_call: Optional[Dict]) -> bool:
        """
        Check if direction parameter is valid (parameter-level validation only).

        This only checks if direction is in VALID_DIRECTIONS, not whether the move
        is valid in the game state (e.g., hitting walls is OK at this level).

        Args:
            tool_call: Tool call dict

        Returns:
            True if direction parameter is valid, False otherwise
        """
        if not tool_call:
            return False

        func = tool_call.get("function", {})
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            direction = args.get("direction", "").lower()
            is_valid = direction in self.VALID_DIRECTIONS

            if not is_valid:
                self.logger.info(
                    f"[SokobanToolCall] Invalid direction parameter: {direction} "
                    f"(must be one of {self.VALID_DIRECTIONS})"
                )

            return is_valid
        except (json.JSONDecodeError, AttributeError):
            return False

    def _format_tool_result(
        self,
        observation: str,
        info: Dict,
        is_valid: bool
    ) -> str:
        """
        Format tool execution result message.

        Args:
            observation: Current game state text
            info: Step info from parent
            is_valid: Whether the action was valid and effective

        Returns:
            Formatted tool result message
        """
        if is_valid:
            action_desc = info.get("action_desc", "Action executed")
            return f"Success: {action_desc}\n\nNew state:\n{observation}"
        else:
            # Invalid action: either bad parameters or ineffective move (e.g., hit wall)
            action_desc = info.get("action_desc", "This move is invalid.")
            return f"Error: {action_desc}\n\nCurrent state:\n{observation}"

    def _handle_invalid_format(
        self,
        action: Any
    ) -> Tuple[List[Dict], float, bool, bool, Dict]:
        """
        Handle structurally invalid actions (format penalty, no state change).

        Args:
            action: The invalid action

        Returns:
            Tuple of (messages, reward, terminated, truncated, info)
        """
        self.message_history.append({
            "role": "assistant",
            "content": str(action)[:200]  # Truncate for safety
        })

        # Use parent's render to get current state
        text_obs = self.render(mode="text")

        user_msg = {
            "role": "user",
            "content": (
                f"\n\n<system-reminder>\n"
                f"CRITICAL: You must use the move_player tool. "
                f"Do not send text-based actions.\n"
                f"</system-reminder>\n\n"
                f"Current state:\n{text_obs}\n\n"
                f"{self.observation_suffix}"
            )
        }
        self.message_history.append(user_msg)

        info = self._create_invalid_info()

        return self.message_history, self.format_penalty, False, False, info

    def _create_invalid_info(self) -> Dict:
        """
        Create info dict for invalid actions.

        Returns:
            Info dict with invalid action metrics
        """
        metrics = {
            "action_is_valid": False,
            "action_is_effective": False,
            "success": False,
            "format_penalty": self.format_penalty,
        }

        metrics_agg_mode = {
            "action_is_valid": "mean",
            "action_is_effective": "mean",
            "success": "last",
            "format_penalty": "mean",
        }

        return {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": "Invalid action (parameter validation failed)",
        }

    def parse_action(self, text: str) -> Dict:
        """
        Override parent's parse_action to handle tool call format.

        This method is called by parent's step() when we pass action_text.
        We use a special format that includes direction mapping.

        Args:
            text: Action text in format "<answer>Direction</answer>"

        Returns:
            Action info dict with action code
        """
        # Extract direction from <answer> tags (our internal format)
        import re
        match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE)
        if match:
            direction = match.group(1).strip().lower()
            # Map direction to action code
            direction_map = {
                "up": 1,
                "down": 2,
                "left": 3,
                "right": 4
            }
            action_code = direction_map.get(direction)
            return {
                "action": action_code,
                "action_content": direction,
                "think_content": ""
            }

        # Invalid format
        return {
            "action": None,
            "action_content": text[:50],
            "think_content": ""
        }
