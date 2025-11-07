from typing import Coroutine, Any, Tuple, Dict, Optional
import asyncio
import json
import httpx
from gem import Env
from roll.pipeline.agentic.env.mcp.mcp_client import MCPClient
from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func
from roll.utils.logging import get_logger

logger = get_logger()

class SokobanMCPEnv(Env):
    def __init__(
            self, 
            server_url: str,
            max_steps=20,
            action_lookup: Dict[int, str] = {1: "Up", 2: "Down", 3: "Left", 4: "Right"},
            env_instruction: str = None,
            format_penalty: float = -0.1,
            action_pattern="<answer>(.*?)</answer>",
            special_token_list=("<think>", "</think>", "<answer>","</answer>", "<|im_start|>", "<|im_end|>"),
            client: Optional[MCPClient] = None):
        super().__init__()
        self.env_instruction = (
            "You are solving the Sokoban puzzle. "
            "You are the player and you need to push all boxes to targets. "
            "When you are right next to a box, you can push it by moving in the same direction. "
            "You cannot push a box through a wall, and you cannot pull a box. "
            f"The answer must be one of action in a turn, format is <answer>Right</answer>."
        )
        
        self.server_url = server_url
        self.max_steps=max_steps
        self.action_lookup=action_lookup
        if env_instruction is not None:
            self.env_instruction = env_instruction
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern      
        self.special_token_list = special_token_list
        
        self._connected: bool = False
        
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        if client:
            self.client = client
        else:
            self.client = MCPClient(self.server_url) 
        
        self.num_env_steps = 0
        self._last_obs = None  

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Resets the environment by calling the 'reset' tool via the MCP server.
        """
        super().reset(seed)
        logger.info(f"Resetting Sokoban environment with seed={seed}...")
        self.num_env_steps = 0
        tool_name = "reset"
        tool_params = {"seed": seed} if seed is not None else {}
        
        try:
            # Use the helpers from the base class to run the async logic.
            # The base class will handle the call and the parsing pipeline.
            obs, _, _, _ = self._run_async_logic(
                self._execute_and_parse(tool_name, tool_params)
            )
        except (httpx.ReadTimeout, httpx.ConnectError, ConnectionError, ValueError, json.JSONDecodeError) as e:
            error_message = f"Failed to reset the environment due to a server or network issue: {e}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            logger.error(f"FATAL: An unexpected critical error occurred during reset: {e}")
            raise RuntimeError("Environment failed to reset due to an unexpected error.") from e  
        
        self._last_obs = obs

        return self._last_obs, {"env_instruction": self._get_instructions()}

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        """
        This step method mirrors the simple SokobanEnv's logic, but uses tool calls.
        """
        self.num_env_steps += 1
        # 1. Parse the action string to get an action ID.
        action_info = self.parse_action(action)
        action_id = action_info.get("action")
        action_content = action_info.get("action_content", "unknown") # Get clean action name like "Up"

        # 2. Handle parsing failure (e.g., LLM output is wrong)
        if action_id is None:
            # Construct a feedback message for the LLM.
            action_desc = f"At turn {self.num_env_steps}, you provided an invalid action."
            # Report the facts in the metrics dictionary for logging and analysis.
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False, # This is the key signal
                "success": False, # Assuming no success on a failed parse
                "format_penalty": self.format_penalty 
            }
            metrics_agg_mode = {
                "action_is_effective": "mean",
                "action_is_valid": "mean",
                "success": "last",
                "format_penalty": "mean",
            }
            # Construct the final info dictionary. The state ('suffix') does not change.
            info = {
                "metrics": metrics,
                "raw_action_text": action,
                "action_desc": action_desc,
                "metrics_agg_mode": metrics_agg_mode,
            }
            return self._last_obs, self.format_penalty, False, False, info

        # 3. Handle parsing success: execute the action via a tool call
        tool_name = "play"
        tool_params = {"action": action_id}

        try:
            # This is the MCP equivalent of `GymSokobanEnv.step()`
            new_obs, terminated, truncated, info = self._run_async_logic(
                self._execute_and_parse(tool_name, tool_params)
            )
        except Exception as e:
            # Keep our robust error handling for network/server issues
            logger.error(f"Server/Network Error on action '{action}': {e}", exc_info=True)
            info = {
                "metrics": {
                    "action_is_effective": False,
                    "action_is_valid": False, 
                    "success": False,
                    "format_penalty": 0.0
                },
                "metrics_agg_mode": {
                    "action_is_effective": "mean",
                    "action_is_valid": "mean",
                    "success": "last",
                    "format_penalty": "mean",
                },
                "error_details": str(e),
                "action_desc": "System Error"
            }
            # set truncated = true
            return self._last_obs, 0.0, False, True, info
        
        # The new observation text must be stored before calling _get_task_suffix().
        self._last_obs = new_obs
        
        reward_from_server = info.get('reward_from_server', 0.0) 
        metrics = info.get("metrics", {})
        action_effective = metrics.get('action_is_effective', False)

        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }
        
        # Construct high-quality feedback for the LLM based on server's ground truth.
        if not action_effective:
            action_desc = f"At turn {self.num_env_steps}, you tried to move {action_content}, which was not effective."
        else:
            action_desc = f"At turn {self.num_env_steps}, you moved {action_content}, which was effective."
            
        info.update({
            "tool_name": tool_name,
            "tool_params": tool_params,
            "raw_action_text": action,
            "action_desc": action_desc,
            "metrics_agg_mode": metrics_agg_mode
        })
        
        return self._last_obs, reward_from_server, terminated, truncated, info
        
    def sample_random_action(self) -> str:
        """Samples a random valid action from the action space."""
        # This requires seeding.py and calling super().reset(seed)
        action_name = self._np_random.choice(list(self.action_lookup.values()))
        return f"<answer>{action_name}</answer>"
        
    def parse_action(self, text) -> Dict[str, Any]:
        """
        Parses a simple action string like "<answer>Up</answer>" and returns the action ID.
        Returns None if parsing fails.
        """
        return default_parser_action_func(text, self.action_pattern, self.action_lookup, self.special_token_list)

    def get_all_actions(self) -> list[str]:
        return list(self.config.action_lookup.values())    
    
    def render(self, mode="text") -> Any:
        if mode == "text":
            return self._last_obs or "The environment has not been reset yet."
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented")
    
    def close(self):
        """Closes the connection to the MCP server."""
        if self._connected:
            self._run_async_logic(self._disconnect())
        # Ensure the event loop is closed if this class created it.
        if not asyncio.get_event_loop().is_running():
            self._event_loop.close()
    
    async def _connect(self):
        if not self._connected:
            if self.client is None:
                raise RuntimeError("Client has not been initialized.")       
            await self.client.__aenter__()
            self._connected = True
    
    async def _disconnect(self):
        if self.client and self._connected:
            await self.client.__aexit__(None, None, None)
            self._connected = False   
            
    def _run_async_logic(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Runs an async coroutine in the managed event loop."""
        if self._event_loop.is_running():
            # This case handles environments where the outer framework is already async.
            future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
            return future.result()
        else:
            # This case handles a purely synchronous script.
            return self._event_loop.run_until_complete(coro)
        
    async def _execute_and_parse(self, tool_name: str, tool_params: Dict) -> Tuple[Any, bool, Dict]:
        """Async helper to connect, call a tool, and parse its response."""
        if not self._connected:
            await self._connect()
        # It's the subclass's job to parse the raw response.
        raw_res = await self.client.call_tool(tool_name, tool_params)
        return self._parse_tool_response(raw_res)
    
    def _parse_tool_response(self, response: Any) -> Tuple[Any, bool, bool, Dict]:
        """
        Default tool response parser. It extracts the text content from the
        tool's raw response, assumes it's a JSON string, and then calls
        a new abstract method `process_parsed_json` to create the final output.
        This separates the "unwrapping" logic from the "interpreting" logic.
        """
        text_content = next((item.text for item in getattr(response, "content", []) if getattr(item, "type", None) == "text"), None)
        if text_content is None:
            raise ValueError("Tool server response is empty or does not contain 'text' content.")
        
        logger.debug(f"Parsing tool response text: {text_content}")
        
        try:
            parsed_json = json.loads(text_content)
            return self._process_parsed_json(parsed_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse server response. Content was: '{text_content}'") from e
        
    def _process_parsed_json(self, data: Dict) -> Tuple[str, bool, bool, Dict]:
        """
        Processes the JSON data from the Sokoban tool (`reset` or `play`)
        into a text-based observation, a done flag, and an info dict.
        This implementation is based on the keys found in the test file.
        """  
        # Extract raw signals from the server's JSON response
        observation_text = data.get("Observation", "Error: No observation found.")
        server_reward = data.get("Reward", 0.0) # Default to 0.0 if not provided
        game_end = data.get("Game End", False)
        server_info = data.get("info", {})
        server_info['format_penalty'] = 0.0
        
        # Translate raw signals into standard terminated/truncated flags 
        game_success = server_info.get("success", False)    
        is_terminated = game_end and game_success
        is_truncated = game_end and not game_success
        
        info = {
            "metrics": server_info,
            "reward_from_server": server_reward
        }
        
        return observation_text, is_terminated, is_truncated, info
    
    def _get_instructions(self) -> str:
        """Returns the static instructions for the Sokoban task."""
        if not self.action_lookup:
            return self.env_instruction
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{v}" for k, v in self.action_lookup.items()])
        return self.env_instruction + action_lookup_str
    
    def _get_task_suffix(self) -> str:
        """Returns the dynamic part of the observation (the current state)."""
        if self._last_obs:
            return (
                f"Here is the current state of the Sokoban puzzle:\n{self._last_obs}\n\n"
                "Legend: P=Player, X=Box, O=Target, √=Box on Target, #=Wall, _=Empty"
            )
        return "The environment state is not available yet."
