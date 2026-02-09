import pytest
from unittest.mock import MagicMock
from roll.pipeline.agentic.env.mcp.mcp_client import MCPClient
from roll.pipeline.agentic.env.mcp import SokobanMCPEnv

# Configuration
SERVER_URL = "http://sokoban-mcp.alibaba-inc.com/sse"
MOCK_SERVER_URL = "http://mock-sokoban-server.test"
TEST_SEED = 2
TEST_ACTION_STR = "Left" 

MOCK_ENV_INSTRUCTION = "Solve the puzzle."
MOCK_ACTION_LOOKUP = {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
MOCK_FORMAT_PENALTY = -0.15

# =============================================================================
# / Pytest Fixtures                                                           /
# =============================================================================
@pytest.fixture(scope="function")
def real_sokoban_env():
    """
    Provides a SokobanMCPEnv instance connected to the REAL server.
    Use this fixture ONLY for integration tests.
    """
    print("\n[Fixture Setup] Creating SokobanMCPEnv instance for integration test...")
        
    env = SokobanMCPEnv(
        server_url=SERVER_URL,
        env_instruction=MOCK_ENV_INSTRUCTION,
        action_lookup=MOCK_ACTION_LOOKUP,
        format_penalty=MOCK_FORMAT_PENALTY,
    )
    yield env
    
@pytest.fixture
def isolated_mock_env():
    """
    Provides a mocked env where the automatic __init__ async logic is disabled,
    allowing for isolated testing of individual methods like step() and reset().
    """
    env = SokobanMCPEnv(
        server_url=MOCK_SERVER_URL,
        env_instruction=MOCK_ENV_INSTRUCTION,
        action_lookup=MOCK_ACTION_LOOKUP,
        format_penalty=MOCK_FORMAT_PENALTY,
        client=MagicMock(spec_set=MCPClient),
    )
    env._last_obs = "A previous observation state."
    yield env

# =============================================================================
# / Integration Tests (Requires Real Server)                                  /
# =============================================================================
def test_sokoban_mcp_env_with_valid_action(real_sokoban_env: SokobanMCPEnv):
    """Integration test for SokobanMCPEnv with real server connection"""   
    # 1. Test environment reset
    obs, info = real_sokoban_env.reset(seed=TEST_SEED)
    print(f"Initial state (seed={TEST_SEED}):\n{obs}")
    
    # Validate initial state
    assert "Solve the puzzle" in obs, "Observation should contain the instruction string."
    assert "######\n#_#_P#" in info['suffix'], "Initial state in 'suffix' mismatch"
    
    # 2. Test single action execution
    llm_output_action = f"<answer>{TEST_ACTION_STR}</answer>"
    
    # ACT: Pass the full, tagged string to the step function.
    obs, reward, terminated, truncated, info = real_sokoban_env.step(llm_output_action)
    print(f"After action {TEST_ACTION_STR}:\n{obs}")
    print(f"Reward: {reward}, Terminated: {terminated}, Success: {info.get('success', False)}")
    
    # Validate post-action state
    assert f"you moved {TEST_ACTION_STR}" in obs, "Feedback text should confirm the executed action."
    assert "######\n#_#P_#" in info['suffix'], "Post-action state in 'suffix' mismatch"
    assert reward == -0.1, "Reward value mismatch"
    assert not terminated, "Game should not be terminated after one action"
    assert not truncated, "Game should not be truncated after one action"
    assert not info['metrics']["success"], "Game should not be successful after one action"    

# =============================================================================
# / Unit Tests - Environment Interaction (`reset`, `step`)                    /
# =============================================================================
def test_reset_wraps_connection_error_in_runtime_error(isolated_mock_env: SokobanMCPEnv):
    """
    Tests that reset raises an error if the connection fails during its execution.
    """
    # ARRANGE
    env = isolated_mock_env
    
    # This mock is for the call inside reset's _run_async_logic
    env._run_async_logic = MagicMock(side_effect=ConnectionError("Server is down!"))
    # We expect reset() to catch ConnectionError and raise RuntimeError
    with pytest.raises(RuntimeError, match="Failed to reset the environment due to a server or network issue"):
        env.reset(seed=TEST_SEED)

def test_step_handles_invalid_action(isolated_mock_env: SokobanMCPEnv):
    """
    Tests that the step() method's first error handling block correctly catches
    ANY ValueError raised by the parse_action method and calls the error handler.
    """
    # ARRANGE
    env = isolated_mock_env
    
    env.parse_action = MagicMock(return_value={"action": None, "action_content": "Go Up"})
    
    obs, reward, terminated, truncated, info = env.step("<answer>Go Up</answer>")
        
    # Check the final output to confirm the error handling flow completed.
    assert "provided an invalid action" in obs
    assert reward == MOCK_FORMAT_PENALTY, "Reward should be the format penalty"
    assert not terminated
    assert not truncated
    assert info["metrics"]["action_is_valid"] is False
    assert info["metrics"]["action_is_effective"] is False
    assert info["metrics"]["success"] is False
    assert "The game state has not changed. Please provide a valid action in the correct format." in info["suffix"], "Suffix should contain the old state"
    
# ============================================================================
# / Unit Tests - Pure Functions and Parsers                                   /
# =============================================================================    
def test_parse_action_simple_logic(isolated_mock_env: SokobanMCPEnv):
    """Tests the generic parse_action method from the MCPEnv base class."""
    env = isolated_mock_env
    # --- Path 1: SUCCESS (Valid action) ---
    action_info = env.parse_action("<answer>Up</answer>")
    assert action_info["action"] == 1
    assert action_info["action_content"] == "Up"
    
    # === BASIC FORMATTING FAILURES ===
    
    # --- Path 2: FAILURE (No <answer> tags) ---
    action_info = env.parse_action("Up")
    assert action_info["action"] is None
        
    # --- Path 3: FAILURE (Content is not valid) ---
    action_info = env.parse_action("<answer>move left</answer>")
    assert action_info["action"] is None
        
def test_process_parsed_json_logic(isolated_mock_env: SokobanMCPEnv):
    """
    Unit test for the game-specific process_parsed_json method.
    """
    isolated_mock_env._last_obs = "Previous state"
    
    success_response = {
        "Observation": "New state",
        "Reward": 1.0,
        "Game End": True,
        "info": {"success": True, "action_is_effective": True}
    }
    obs, terminated, truncated, info = isolated_mock_env._process_parsed_json(success_response)
    
    assert "New state" in obs
    assert terminated
    assert not truncated
    assert info["metrics"]["success"]
    assert info["metrics"]["action_is_effective"]
    assert info["metrics"]["format_penalty"] == 0.0
    assert info["reward_from_server"] == 1.0