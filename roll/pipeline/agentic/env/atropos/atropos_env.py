from datasets import disable_progress_bar; disable_progress_bar()
import concurrent.futures
try:
    import tqdm.contrib.concurrent
    def safe_thread_map(fn, *iterables, **kwargs):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(fn, *iterables))
    tqdm.contrib.concurrent.thread_map = safe_thread_map
except ImportError:
    pass

import asyncio
import logging
import os
import sys
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union

# Attempt to use uvloop if available
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

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
    Ported with critical attributes for TrajEnvManager compatibility.
    """

    def _get_loop(self):
        """Get or create a usable event loop for the current thread.
        
        Handles Ray's ThreadPoolExecutor threads where no default event loop exists
        (Python 3.10+ raises RuntimeError in non-main threads).
        """
        # First try to get an existing, non-closed loop
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                return loop
        except RuntimeError:
            pass
        # Create and install a new loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def _run_async(self, coro):
        """Run an async coroutine synchronously, safe from any thread context.
        
        If an event loop is already running in this thread, spawn a helper thread
        to run the coroutine via asyncio.run(). Otherwise, use run_until_complete().
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside a running loop (e.g. Ray async actor) — delegate to a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=120)
        else:
            # No running loop — use the thread-local loop directly
            local_loop = self._get_loop()
            return local_loop.run_until_complete(coro)

    def __init__(
        self,
        atropos_env_path: str,
        max_steps: int = 16,
        env_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        **kwargs
    ) -> None:
        # Mandatory attributes for ROLL NativeEnvManager
        # We use object.__setattr__ to bypass gem.Env's strict __setattr__ which often fails on late-bound attributes
        object.__setattr__(self, "_env_reset_failed", False)
        object.__setattr__(self, "_env_info", {})
        
        super().__init__()
        
        # Path injection to ensure Atropos and ROLL modules are findable
        for path in ["/workspace/ROLL", "/workspace/atropos"]:
            if path not in sys.path:
                sys.path.append(path)

        self.atropos_env_path = atropos_env_path
        self.max_steps = max_steps
        self.debug = debug
        self.env_config = env_config or {}
        
        # 1. Dynamic Loading
        self.env_class = load_atropos_env_class(atropos_env_path)
        self.env = create_atropos_instance(self.env_class, self.env_config)
        
        # 2. Async Lifecycle Management — always run setup() to completion
        self._run_async(self.env.setup())

        # Episode state
        self.current_item = None
        self.history = []
        self.step_count = 0
        
    @property
    def env_reset_failed(self):
        return getattr(self, "_env_reset_failed", False)

    @property
    def env_info(self):
        return getattr(self, "_env_info", {})

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment and returns the initial observation.
        """
        object.__setattr__(self, "_env_reset_failed", False)
        try:
            self.current_item = self._run_async(safe_get_next_item(self.env))
            
            # Extract the initial prompt from the environment item
            initial_prompt = ""
            if isinstance(self.current_item, dict):
                initial_prompt = self.current_item.get("question", 
                                 self.current_item.get("problem_statement", 
                                 self.current_item.get("prompt", "")))
            else:
                initial_prompt = str(self.current_item) or "New Task"
                
            self.history = [{"role": "user", "content": str(initial_prompt)}]
            self.step_count = 0
            
            if self.debug:
                logger.info(f"\n{'='*20} ATROPOS RESET {'='*20}")
                logger.info(f"Task: {str(initial_prompt)[:100]}...")
            
            object.__setattr__(self, "_env_info", {"item": self.current_item})
            return self.history, self.env_info
        except Exception as e:
            logger.error(f"AtroposEnv reset failed: {e}")
            object.__setattr__(self, "_env_reset_failed", True)
            return "Reset Failed", {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        assistant_msg = str(action)
        
        if self.debug:
            logger.info(f"\n--- ATROPOS STEP {self.step_count} ---")
            logger.info(f"Action: {assistant_msg[:200]}...")
            
        # Delegate execution to the controlled rollout bridge
        try:
            obs, reward, done, info = self._run_async(
                execute_controlled_rollout(
                    self.env, 
                    self.current_item, 
                    assistant_msg, 
                    self.history, 
                    debug=self.debug,
                    reward_config=self.config.get("reward_config")
                )
            )
            
            self.history.append({"role": "assistant", "content": assistant_msg})
            if not done and obs:
                if isinstance(obs, list):
                    for msg in obs: self.history.append(msg)
                else:
                    self.history.append({"role": "user", "content": str(obs)})
            
            truncated = (self.step_count >= self.max_steps)
            if truncated: done = True
                
            return self.history, float(reward), done, truncated, info
        except Exception as e:
            logger.error(f"AtroposEnv step failed: {e}")
            return "Step Failed", 0.0, True, True, {"error": str(e)}

    def render(self): pass
    def close(self): pass
