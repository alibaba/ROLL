import asyncio
import logging
import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from gem import Env
from roll.pipeline.agentic.env.atropos.manager import (
    load_atropos_env_class,
    create_atropos_instance,
    safe_get_next_item
)
from roll.pipeline.agentic.env.atropos.executor import execute_controlled_rollout
from roll.utils.constants import EpisodeStopReason

logger = logging.getLogger(__name__)

class AtroposEnv(Env):
    """
    Atropos environment adapter for ROLL.
    
    This adapter treats Atropos as a trajectory-driven black box, bridging 
    its asynchronous rollout engine to ROLL's synchronous step-based interface.
    
    ### Abstraction Boundary
    - Atropos is treated as a black-box trajectory engine. The adapter interacts 
      only with the public `collect_trajectories` API.
    - No assumptions are made about internal environment state or logic.
    - Convergence between trajectory and step interfaces is managed via 
      controlled partial rollout execution.

    ### Limitations and Performance
    - **Replay Cost**: Each `step()` triggers a rollout from the beginning 
      of the trajectory. This ensures correctness by allowing Atropos logic 
      (tools, parsing) to react to the full context, but increases compute cost.
    - **Rewards**: Rewards are typically episodic and returned upon completion 
      of the Atropos trajectory.
    - **Control**: Turn boundaries are detected by the execution bridge 
      detecting new model generation requests.
    """

    def __init__(
        self,
        atropos_env_path: str,
        max_steps: int = 16,
        env_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.atropos_env_path = atropos_env_path
        self.max_steps = max_steps
        self.debug = debug
        self.env_config = env_config or {}
        
        # 1. Dynamic Loading
        self.env_class = load_atropos_env_class(atropos_env_path)
        self.env = create_atropos_instance(self.env_class, self.env_config)
        
        # 2. Async Lifecycle Management (Sync boundary)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # If we are inside an existing loop (e.g. durante testing), 
            # we cannot use run_until_complete in the constructor.
            # In production ROLL, the worker is usually sync, so this won't happen.
            logger.debug("Event loop is already running. Scheduling setup in background.")
            loop.create_task(self.env.setup())
        else:
            loop.run_until_complete(self.env.setup())

        # Episode state
        self.current_item = None
        self.history = []
        self.step_count = 0
        
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment and returns the initial observation.
        """
        loop = asyncio.get_event_loop()
        self.current_item = loop.run_until_complete(safe_get_next_item(self.env))
        
        # Extract the initial prompt from the environment item
        initial_prompt = ""
        try:
            if isinstance(self.current_item, dict):
                initial_prompt = self.current_item.get("question", 
                                 self.current_item.get("problem_statement", 
                                 self.current_item.get("prompt", "")))
            elif isinstance(self.current_item, (list, tuple)) and len(self.current_item) > 0:
                # For complex Atropos environments, the first element is often the context/messages
                first_elem = self.current_item[0]
                if isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                    # If it's a list of messages, the first user message is the prompt
                    msg = first_elem[0]
                    if isinstance(msg, dict):
                        initial_prompt = msg.get("content", msg.get("value", ""))
                    else:
                        initial_prompt = str(msg)
                else:
                    initial_prompt = str(first_elem)
            else:
                initial_prompt = str(self.current_item)
        except Exception as e:
            logger.debug(f"Could not extract prompt from complex item: {e}. Using string fallback.")
            initial_prompt = str(self.current_item)
            
        initial_prompt = str(initial_prompt) or "New Task"
            
        self.history = [{"role": "user", "content": initial_prompt}]
        self.step_count = 0
        
        if self.debug:
            logger.info(f"\n{'='*20} ATROPOS RESET {'='*20}")
            logger.info(f"Task: {initial_prompt[:100]}...")
            
        return initial_prompt, {"item": self.current_item}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Executes one step in the environment.
        action: The assistant's response string.
        """
        self.step_count += 1
        
        # action is typically a string (assistant response)
        assistant_msg = str(action)
        
        if self.debug:
            logger.info(f"\n--- ATROPOS STEP {self.step_count} ---")
            logger.info(f"Action (Assistant): {assistant_msg[:100]}...")

        # Run the controlled partial trajectory execution
        # Delegate execution to the controlled rollout bridge
        loop = asyncio.get_event_loop()
        obs, reward, done, info = loop.run_until_complete(
            execute_controlled_rollout(
                self.env, 
                self.current_item, 
                assistant_msg, 
                self.history, 
                debug=self.debug
            )
        )
        
        # Update history
        self.history.append({"role": "assistant", "content": assistant_msg})
        
        if not done and obs:
            # obs is either a string or a list of messages (reactions)
            if isinstance(obs, list):
                for msg in obs:
                    self.history.append(msg)
            else:
                self.history.append({"role": "user", "content": str(obs)})
        
        # Handle ROLL's truncated/terminated convention
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            done = True
            
        if self.debug:
            logger.info(f"Observation: {str(obs)[:100]}...")
            logger.info(f"Reward: {reward}")
            logger.info(f"Done: {done}")

        # ROLL returns (obs, reward, terminated, truncated, info)
        return obs, float(reward), done, truncated, info

    def render(self):
        pass

    def close(self):
        pass
