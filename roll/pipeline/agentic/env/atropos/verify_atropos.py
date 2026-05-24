import os
import sys
import asyncio
import logging
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

# Add current directory and Atropos to path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../atropos"))

# --- MOCK GEM DEPENDENCY ---
if "gem" not in sys.modules:
    gem = ModuleType("gem")
    def register(id, entry_point, **kwargs):
        logging.info(f"[Mock Gem] Registered environment: {id} -> {entry_point}")
    gem.register = register
    
    class Env:
        def __init__(self, *args, **kwargs): pass
        def reset(self, *args, **kwargs): return None, {}
        def step(self, action): return None, 0.0, False, False, {}
    gem.Env = Env
    sys.modules["gem"] = gem

# Try to import from real ROLL first
try:
    from roll.utils.constants import EpisodeStopReason
except ImportError:
    constants = ModuleType("roll.utils.constants")
    class EpisodeStopReason:
        DONE = "done"
        TRUNCATED = "truncated"
    constants.EpisodeStopReason = EpisodeStopReason
    if "roll" not in sys.modules:
        sys.modules["roll"] = ModuleType("roll")
    if "roll.utils" not in sys.modules:
        sys.modules["roll.utils"] = ModuleType("roll.utils")
    sys.modules["roll.utils.constants"] = constants

from roll.pipeline.agentic.env.atropos import AtroposEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AtroposValidation")

def test_gsm8k_integration():
    """Validates the standard single-turn rollout flow."""
    logger.info("\n=== Testing GSM8K Integration (Single-Turn) ===")
    
    env_path = "environments.gsm8k_server:GSM8kEnv"
    
    env = AtroposEnv(
        atropos_env_path=env_path,
        max_steps=5,
        debug=True,
        env_config={
            "group_size": 1,
            "max_token_length": 128
        }
    )

    obs, info = env.reset()
    logger.info(f"Initial Observation: {obs}")
    
    mock_action = "<think> Let's solve this. 1+1=2. </think> The answer is \\boxed{2}"
    obs, reward, terminated, truncated, info = env.step(mock_action)
    
    logger.info(f"Step Result: Reward={reward}, Terminated={terminated}")
    if terminated:
        logger.info("SUCCESS: Single-turn rollout validated.")

def test_multiturn_tool_use():
    """Validates multi-turn history persistence through the Execution Bridge."""
    logger.info("\n=== Testing Multi-Turn Tool Use Integration ===")
    
    env_path = "environments.tool_use_multiturn_server:MultiTurnToolCallingEnv"
    
    # Initialize with default data loading
    env = AtroposEnv(
        atropos_env_path=env_path,
        max_steps=10,
        debug=True,
        env_config={
            "group_size": 1,
            "max_token_length": 256
        }
    )
    
    # reset() will call safe_get_next_item and setup history
    obs, info = env.reset()
    logger.info(f"Initial Observation: {obs}")
    
    # STEP 1: Simulate a tool call action
    # We use a tool call format that the environment expects
    action_1 = "<think>I need a tool.</think><tool_call>{\"name\":\"calc\",\"arguments\":{\"q\":\"5+5\"}}</tool_call>"
    logger.info(f"Step 1 Action: {action_1}")
    
    obs, reward, terminated, truncated, info = env.step(action_1)
    logger.info(f"Step 1 Observation: {obs}")
    
    if not terminated:
        logger.info("SUCCESS: Turn boundary detected after tool call.")
        
        # STEP 2: Finish the episode
        action_2 = "<think>Done.</think>The answer is 10."
        obs, reward, terminated, truncated, info = env.step(action_2)
        logger.info(f"Step 2 Result: Reward={reward}, Terminated={terminated}")

if __name__ == "__main__":
    test_gsm8k_integration()
    try:
        test_multiturn_tool_use()
    except Exception as e:
        logger.error(f"Multi-turn test failed: {e}")
