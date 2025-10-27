import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from jsonschema.exceptions import ValidationError
import re
import json
from roll.pipeline.agentic.env.mcp.mcp_client import MCPClient
from roll.pipeline.agentic.tools.mcp_tool import MCPTool

class MockContentBlock:
    def __init__(self, text, type="text"):
        self.text = text
        self.type = type

class MockToolObject:
    def __init__(self, name, description, inputSchema):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

class MockCallToolResult:
    def __init__(self, isError, content, structuredContent=None):
        self.isError = isError
        self.content = content
        self.structuredContent = structuredContent        

@pytest.fixture
def mock_play_tool_metadata() -> list:
    """Provides a standard, reusable definition for the 'play' tool's metadata."""
    return [
        {
            "name": "play",
            "description": "Performs a play action.",
            "inputSchema": {
                "type": "object",
                "properties": {"action": {"type": "integer"}},
                "required": ["action"]
            }
        }
    ]

@pytest.fixture
def mock_nested_tool_metadata() -> list:
    """
    Provides metadata for a tool with a simple, two-level nested object.
    This is ideal for testing the basic recursive capability without unnecessary complexity.
    """
    return [
        {
            "name": "set_user_contact",
            "description": "Sets a user's contact information.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The unique identifier for the user."
                    },
                    "contact_info": { 
                        "type": "object",
                        "description": "The user's contact details.",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"}
                        },
                        "required": ["email"]
                    }
                },
                "required": ["user_id", "contact_info"]
            }
        }
    ]

@pytest.fixture
def mock_tool() -> MCPTool:
    """Provides a MCPTool instance with a mocked client. The tool is NOT connected."""
    mock_client = MagicMock(spec_set=MCPClient)
    
    # IMPORTANT: Since MCPClient has async methods, we must configure
    # our mock to handle them. We'll use AsyncMock for this.
    mock_client.__aenter__ = AsyncMock()
    mock_client.__aexit__ = AsyncMock()
    mock_client.tools = AsyncMock()
    mock_client.call_tool = AsyncMock()
    
    tool = MCPTool(client=mock_client)
    return tool

@pytest.fixture
def connected_mock_tool(mock_tool: MCPTool, mock_play_tool_metadata: list) -> MCPTool:
    """Provides a "connected" MCPTool with pre-loaded mock tool metadata."""
    # Convert the dictionary metadata to MockToolObject instances
    mock_tool_objects = [
        MockToolObject(
            name=meta['name'],
            description=meta['description'],
            inputSchema=meta['inputSchema']
        )
        for meta in mock_play_tool_metadata
    ]
    
    # Configure the mock client's `tools` method to return our data
    # when it's awaited inside `tool.connect()`.
    mock_tool._client.tools.return_value = mock_tool_objects
    
    asyncio.run(mock_tool._async_connect_and_fetch())
    
    mock_tool._is_connected_and_ready = True
    
    return mock_tool

def test_instruction_string_uses_custom_prompt_when_provided(
    mock_tool: MCPTool,
    mock_play_tool_metadata: list):
    """
    Tests that if a custom_prompt_template is provided during initialization,
    the async instruction_string method returns that exact template.
    """
    # ARRANGE
    my_custom_prompt = "This is a completely custom prompt. Use tool 'play' with param 'action'."
    
    mock_tool._custom_prompt = my_custom_prompt
    mock_tool._is_connected_and_ready = True
    mock_tool._tool_metadata = mock_play_tool_metadata
        
    # ACT
    prompt = mock_tool.instruction_string()
    
    # ASSERT
    assert prompt == my_custom_prompt
    assert "## AVAILABLE TOOLS" not in prompt

def test_generate_prompt_formats_correctly(mock_tool: MCPTool, mock_play_tool_metadata: list):
    """Tests that the prompt generation is correct and well-formatted."""
    # ARRANGE
    # Manually set the tool metadata using the shared fixture.
    mock_tool._tool_metadata = mock_play_tool_metadata
    
    # ACT
    prompt = mock_tool._generate_prompt_from_cached_tools()
    
    # ASSERT
    assert "## AVAILABLE TOOLS" in prompt
    assert '"name": "play"' in prompt
    assert "<tool_call>" in prompt # Check for the correct tag
    assert prompt == prompt.strip()

def test_prompt_example_handles_nested_objects_correctly(mock_tool: MCPTool, mock_nested_tool_metadata: list):
    """
    Tests that the generated example in the prompt correctly handles
    tools with nested object parameters.
    """
    # ARRANGE
    mock_tool._tool_metadata = mock_nested_tool_metadata
    
    # ACT
    example_json_str = mock_tool._create_example_action_json(mock_nested_tool_metadata[0])
    
    # ASSERT
    data = json.loads(example_json_str)
    assert data["tool_name"] == "set_user_contact"
    params = data["tool_params"]
    
    # 1st layer
    assert "user_id" in params
    assert isinstance(params["user_id"], str)
    
    assert "contact_info" in params
    assert isinstance(params["contact_info"], dict)
    
    # 2rd layer
    contact_info = params["contact_info"]
    assert "email" in contact_info
    assert isinstance(contact_info["email"], str)
    assert "phone" in contact_info
    assert isinstance(contact_info["phone"], str)
    
def test_parse_action_extracts_tag_correctly(mock_tool: MCPTool):
    """Tests that _parse_action correctly finds and extracts content from tags."""
    # Case 1: Success
    action_str = '<think>some thought</think><tool_call>{"json": "content"}</tool_call>'
    content, parsed_segment, is_parsed = mock_tool._parse_action(action_str)
    assert is_parsed is True
    assert content == '{"json": "content"}'
    assert parsed_segment == action_str

    # Case 2: Failure (no tag)
    action_str_no_tag = "some raw text"
    content, parsed_segment, is_parsed = mock_tool._parse_action(action_str_no_tag)
    assert is_parsed is False
    assert content == ""
    assert parsed_segment == ""    
    
def test_validate_tool_call_with_json_schema(connected_mock_tool: MCPTool):
    """Tests the validation logic which now uses JSON Schema."""
    tool: MCPTool = connected_mock_tool 
    
    # --- SUCCESS Path ---
    try:
        tool._validate_tool_call("play", {"action": 1})
    except (ValueError, ValidationError):
        pytest.fail("Validation failed unexpectedly for a valid call.")

    # --- FAILURE Paths ---
    # Unknown tool name
    with pytest.raises(ValueError, match="Unknown tool_name: 'fly'"):
        tool._validate_tool_call("fly", {})
        
    # Invalid parameter type
    with pytest.raises(ValidationError, match="'up' is not of type 'integer'"):
        tool._validate_tool_call("play", {"action": "up"})

    # Missing required parameter
    with pytest.raises(ValidationError, match="'action' is a required property"):
        tool._validate_tool_call("play", {})    

def test_execute_action_handles_validation_error(connected_mock_tool: MCPTool):
    """
    Tests that execute_action correctly catches a ValidationError from the
    validation step and returns a proper error tuple.
    """
    # ARRANGE
    tool: MCPTool = connected_mock_tool
    
    any_action = '<tool_call>{"tool_name": "play", "tool_params": {}}</tool_call>'
    
    # We use patching to replace `_validate_tool_call` for the duration of this test.
    # We configure its side_effect to be raising a specific ValidationError.
    # This simulates the "validation failed" event directly.
    mocked_error = ValidationError("A mocked validation error.")
    
    with patch.object(tool, '_validate_tool_call', side_effect=mocked_error) as mock_validator:
        # ACT
        is_parsed, is_valid, observation, parsed_action = tool.execute_action(any_action)

    # ASSERT
    # 1. Did we actually call the validator? Yes.
    mock_validator.assert_called_once_with("play", {})
    
    # 2. Did execute_action handle the exception correctly?
    assert is_parsed is True
    assert is_valid is False
    assert "Validation Error: The tool call format is incorrect. " in observation
    assert parsed_action == any_action

def test_execute_action_handles_server_execution_error(connected_mock_tool: MCPTool):
    """
    Tests that execute_action correctly handles an error raised by the client
    during the `call_tool` execution.
    """
    tool: MCPTool = connected_mock_tool 
    valid_action = '<tool_call>{"tool_name": "play", "tool_params": {"action": 1}}</tool_call>'
    
    # ARRANGE: Configure the mock client's `call_tool` to raise an exception
    # for this specific test case.
    tool._client.call_tool.side_effect = ConnectionRefusedError("Server refused connection")
    
    # ACT
    is_parsed, is_valid, observation, _ = tool.execute_action(valid_action)
    
    # ASSERT
    assert is_parsed is True
    assert is_valid is False
    assert "[Execution Error: Server refused connection]" in observation  
    
def test_execute_action_handles_server_business_logic_error(connected_mock_tool: MCPTool):
    """
    Tests that execute_action correctly handles a business logic error
    returned by the server (i.e., the result object has isError=True).
    """
    # ARRANGE
    tool: MCPTool = connected_mock_tool
    valid_action = '<tool_call>{"tool_name": "play", "tool_params": {"action": 9}}</tool_call>'
    
    # Configure the mock client's call_tool to return a "successful" call
    # that encapsulates a business logic failure.
    server_error_response = MockCallToolResult(
        isError=True,
        content=[MockContentBlock("Error executing tool play:")]
    )
    tool._client.call_tool.return_value = server_error_response
    
    # ACT
    is_parsed, is_valid, observation, _ = tool.execute_action(valid_action)
    
    # ASSERT
    assert is_parsed is True
    assert is_valid is False # Crucially, this must be False
    
    # The observation should contain the specific error message from the server content.
    assert observation == "[Execution Error: Error executing tool play:]"
    
    # Verify that the tool was still called correctly.
    tool._client.call_tool.assert_called_once_with("play", {"action": 9})  

def test_mcp_tool_end_to_end_with_sokoban_mcp_server():
    """
    Tests the full lifecycle of MCPTool against a running MCP server.
    It covers connection, instruction generation, valid action execution,
    business logic error handling, and closing the connection.
    """
    # --- 1. Initialization and Connection ---
    print(f"\nConnecting to MCP server at https://sokoban-mcp-center-share.alibaba-inc.com/sse...")
    tool = MCPTool(server_url="https://sokoban-mcp-center-share.alibaba-inc.com/sse")
    
    # --- 2. Test Instruction String Generation ---
    instructions = tool.instruction_string()
    print(f"Generated instructions:\n---\n{instructions}\n---")
    
    # Validate that the instructions seem correct
    assert "## AVAILABLE TOOLS" in instructions
    assert '"name": "play"' in instructions # Assumes server has a 'play' tool
    assert "## CRITICAL USAGE INSTRUCTIONS" in instructions
    
    # --- 3. Test a Valid Action Execution (Happy Path) ---
    # --- Phase 1: Test the 'reset' tool call ---
    # This ensures the environment can be correctly initialized to a known state.
    reset_action_str = '<tool_call>{"tool_name": "reset", "tool_params": {"seed": 2}}</tool_call>'
    print(f"Executing reset action: {reset_action_str}")
    
    is_parsed, is_valid, observation, _ = tool.execute_action(reset_action_str)
    print(f"Observation from reset action:\n{observation}")
    
    # ASSERT: Validate the outcome of the reset action
    assert is_parsed, "The reset action string should have been parsed."
    assert is_valid, "The reset tool call should be valid and successful."
    
    # Extract and parse the JSON content from the <information> tag
    match = re.search(r"<information>(.*)</information>", observation, re.DOTALL)
    assert match, "Reset observation did not contain the <information> tag with content."
    reset_json_content = match.group(1).strip()           
    reset_data = json.loads(reset_json_content)
    
    # Assert the specific state of the board after resetting with seed 2
    expected_initial_state = "######\n#_#_P#\n#_#X_#\n#___O#\n#____#\n######"
    assert reset_data["Observation"] == expected_initial_state, "The initial board state after reset is incorrect."

    # --- Phase 2: Test a 'play' action from the known state ---
    # Construct a valid action based on the server's schema for the 'play' tool
    valid_action_str = '<tool_call>{"tool_name": "play", "tool_params": {"action": 3}}</tool_call>'
    print(f"Executing valid play action: {valid_action_str}")
    
    is_parsed, is_valid, observation, _ = tool.execute_action(valid_action_str)        
    print(f"Observation from valid action:\n{observation}")
    
    # Validate the happy path result
    assert is_parsed is True
    assert is_valid is True
    # Get real response
    match = re.search(r"<information>(.*)</information>", observation, re.DOTALL)
    assert match, "Observation did not contain the <information> tag with content."
    json_content = match.group(1).strip()           
    data = json.loads(json_content)
    # assert "Action successful" in observation
    assert data["Reward"] == -0.1
    assert data["Game End"] is False
    assert data["Game Success"] is False
    assert "######\n#_#P_#\n#_#X_#\n#___O#\n#____#\n######" in data["Observation"]

    # --- 4. Test a Business Logic Error ---
    # Construct an action that is syntactically valid but should trigger a business error on the server
    business_error_action_str = '<tool_call>{"tool_name": "play", "tool_params": {"action": 999}}</tool_call>' # Assume 999 is an invalid move
    print(f"Executing action designed to cause a business logic error: {business_error_action_str}")

    is_parsed, is_valid, observation, _ = tool.execute_action(business_error_action_str)

    print(f"Observation from business error:\n{observation}")

    # Validate the business logic error result
    assert is_parsed is True
    assert is_valid is False
    assert "[Execution Error: Error executing tool play:]" in observation
    # Assert based on expected server error message, e.g.:
    # assert "Invalid action ID" in observation
    
    print("Tool connection closed.")   
    
    tool.close()
    
def test_calculator_tool_with_subset_of_tools():
    """
    Integration test for MCPTool using a real calculator server.
    It verifies that the `tool_subset` feature correctly limits the tool's
    capabilities and prompt generation.
    """
    # ARRANGE
    calculator_subset = ["add", "modulo"]
    
    tool = MCPTool(
        server_url="https://calculator-mcp-center-share.alibaba-inc.com/sse",
        tool_names_subset=calculator_subset
    )
    
    # ACT & ASSERT
    prompt = tool.instruction_string()
    
    assert '"name": "add"' in prompt
    assert '"name": "modulo"' in prompt
    
    assert '"name": "subtract"' not in prompt
    assert '"name": "multiply"' not in prompt
    print("Prompt correctly contains only 'add' and 'modulo'.")
    
    # --- Success Path ---
    print("\n--- Verifying execution of included tools ---")
    # test 'add'
    add_action = '<tool_call>{"tool_name": "add", "tool_params": {"firstNumber": 10, "secondNumber": 5}}</tool_call>'
    is_parsed, is_valid, observation, _ = tool.execute_action(add_action)
    assert is_parsed is True
    assert is_valid is True
    assert observation == '<information>15</information>' 
    print("'add' tool executed successfully.")
    
    # test 'modulo'
    modulo_action = '<tool_call>{"tool_name": "modulo", "tool_params": {"dividend": 10, "divisor": 3}}</tool_call>'
    is_parsed, is_valid, observation, _ = tool.execute_action(modulo_action)
    assert is_parsed is True
    assert is_valid is True
    assert observation == '<information>1</information>'
    print("'modulo' tool executed successfully.")

    # --- Failure Path ---
    print("\n--- Verifying rejection of excluded tools ---")
    subtract_action = '<tool_call>{"tool_name": "subtract", "tool_params": {"minuend": 10, "subtrahend": 5}}</tool_call>'
    is_parsed, is_valid, observation, _ = tool.execute_action(subtract_action)
    
    assert is_parsed is True
    assert is_valid is False
    
    assert "[Validation Error: The tool call format is incorrect. Reason: Unknown tool_name: 'subtract'. Available tools are: ['add', 'modulo']]" in observation
    print("'subtract' tool was correctly rejected as it's not in the subset.")    
    
    tool.close()