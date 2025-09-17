from typing import Any, Coroutine, Tuple, Dict, List, Optional
import asyncio
import re
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError

import mcp.types as types
from roll.pipeline.agentic.env.mcp.mcp_client import MCPClient

from gem.tools.base_tool import BaseTool
from roll.utils.logging import get_logger

logger = get_logger()

class MCPTool(BaseTool):
    """
    A tool that interacts with an MCP server.

    It connects to a server, discovers available tools, generates a dynamic
    prompt for an AI agent, and executes tool calls based on the agent's
    formatted responses.
    """
    tool_type = "mcp"

    def __init__(self, 
                num_workers=1, 
                server_url: Optional[str] = None, 
                client: Optional[MCPClient] = None,
                tool_names_subset: Optional[List[str]] = None,
                custom_prompt: Optional[str] = None):
        super().__init__(num_workers)
        
        if not client and not server_url:
            raise ValueError("Either 'client' or 'server_url' must be provided.")

        self._client = client or MCPClient(server_url)
        self._tool_metadata: List[Dict] = []
        self._tool_names_subset = tool_names_subset 
        self._custom_prompt = custom_prompt
        self._is_connected_and_ready = False
        
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

    def instruction_string(self) -> str:
        """
        Returns the instruction string for the agent.
        
        If a `custom_prompt` was not provided during initialization, it generates a prompt 
        based on the configured tools.

        Raises:
            RuntimeError: If the tool is not connected before calling.
        """
        self._ensure_connected()
        
        if self._custom_prompt:
            return self._custom_prompt
        
        return self._generate_prompt_from_cached_tools() 

    def execute_action(self, action: str) -> Tuple[bool, bool, str, str]:
        """
        Parses, validates, and executes a tool call from the agent's action string.

        Args:
            action: The raw action string, expected to contain a JSON object with
                tool call information within <tool_call>...</tool_call> tags.

        Returns:
            A tuple (is_parsed, is_valid, observation, parsed_action):
            - is_parsed (bool): True if the action has the <tool_call> tag, False otherwise.
            - is_valid (bool): If parsed, True only if the entire call was successful.
            - observation (str): The result to be returned to the agent (either a 
                                success message or a specific error).
            - parsed_action (str): The relevant segment of the action string for logging.           
        """
        self._ensure_connected()
        
        json_content, parsed_action, is_parsed = self._parse_action(action)
        
        if not is_parsed:
            # The action is not intended for this tool.
            # Return (False, False, ...) to signal the wrapper to try other tools.
            return (False, False, "", action)
        
        # --- STAGE 1: VALIDATION BLOCK ---
        # This block validates the agent's command *before* execution.
        # It checks for JSON errors, missing keys, and schema mismatches.
        try:
            data = json.loads(json_content)
            if not isinstance(data, dict):
                raise ValueError(f"Parsed JSON is not a dictionary, but {type(data)}")
            
            tool_name = data.get("tool_name")
            tool_params = data.get("tool_params", {}) 
            
            if not isinstance(tool_name, str) or not isinstance(tool_params, dict):
                raise ValueError("JSON must contain a 'tool_name' (string) and 'tool_params' (dict).")
            
            # Validate the parameters against the tool's specific JSON schema.
            self._validate_tool_call(tool_name, tool_params)
            
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            # The content was malformed or invalid.
            # The action was parsed, but it's not a valid call.
            error_msg = f"[Validation Error: The tool call format is incorrect. Reason: {e}]"
            return (True, False, error_msg, parsed_action)

        # --- STAGE 2: EXECUTION BLOCK ---
        # This block handles the actual remote call and its outcome.
        # It catches unexpected runtime errors like network failures.
        try:
            result = self._run_async_logic(self._client.call_tool(tool_name, tool_params))
            
            # Process the server's response. This response can indicate either
            # a business-level success or a business-level failure (e.g., "Error executing tool").
            is_success, observation_string = self._process_server_response(result)
            
            # The final validity (`is_valid`) depends directly on the server's response.
            # A business logic error from the server means the action was not ultimately "valid" or "successful".
            return (True, is_success, observation_string, parsed_action)
        
        except Exception as e:
            # An error occurred during the remote call.
            # The action was parsed and the call format was valid, but execution failed.
            error_msg = f"[Execution Error: {e}]"
            return (True, False, error_msg, parsed_action)
        
    def close(self):
        """
        Closes the underlying client connection.
        
        It is highly recommended to call this method manually before your
        application exits to ensure all network resources are properly released.
        """
        if self._client and self._is_connected_and_ready:
            if self._event_loop and not self._event_loop.is_closed():
                logger.debug("MCPTool: Closing client connection...")
                try:
                    self._run_async_logic(self._client.__aexit__(None, None, None))
                    self._is_connected_and_ready = False
                    logger.debug("MCPTool: Connection closed.")
                except Exception as e:
                    print(f"MCPTool: Error during close: {e}")
                    
    def _run_async_logic(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """
        Executes an async coroutine within the managed event loop.

        This method acts as a bridge between the synchronous public methods
        and the asynchronous internal operations. It detects if an event loop
        is already running (e.g., in a FastAPI or Jupyter environment) and
        uses a thread-safe approach, or runs the coroutine to completion
        in a standard synchronous script.

        Args:
            coro: The asynchronous coroutine to execute.

        Returns:
            The result of the executed coroutine.
        """
        if self._event_loop.is_running():
            # This case handles environments where the outer framework is already async.
            future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
            return future.result()
        else:
            # This case handles a purely synchronous script.
            return self._event_loop.run_until_complete(coro)
            
    def _ensure_connected(self):
        """
        Ensures the tool is connected to the server before any operation.
        """   
        if not self._is_connected_and_ready:
            logger.debug("MCPTool: First use detected. Connecting and fetching tools...")
            self._run_async_logic(self._async_connect_and_fetch())
            self._is_connected_and_ready = True
            logger.debug("MCPTool: Connection successful.")
            
    async def _async_connect_and_fetch(self):
        """
        Performs the actual asynchronous connection and tool metadata fetching.
        """ 
        await self._client.__aenter__()  
        tools = await self._client.tools()
        if not tools:
            self._tool_metadata = []
            return
        
        def tool_to_dict(tool_obj):       
            return {
                "name": getattr(tool_obj, "name", "unnamed_tool"),
                "description": getattr(tool_obj, "description", "No description."),
                "inputSchema": getattr(tool_obj, "inputSchema", {})
            }
        
        all_tools_as_dicts = [tool_to_dict(t) for t in tools]
        
        if self._tool_names_subset:
            self._tool_metadata = [
                tool for tool in all_tools_as_dicts
                if tool.get("name") in self._tool_names_subset
            ]
        else:
            self._tool_metadata = all_tools_as_dicts
    
    def _generate_prompt_from_cached_tools(self) -> str:
        """Generates a comprehensive prompt using the cached tool metadata."""
        if not self._tool_metadata:
            return "No tools are available from the server." # Graceful handling of empty tool list
        
        tools_json_string = json.dumps(self._tool_metadata, indent=2, ensure_ascii=False)
        
        example_json_string = self._create_example_action_json(
            self._tool_metadata[0], 
            indent=2
        )
        
        example_tool_name = self._tool_metadata[0].get("name", "example_tool")
        
        prompt_template = f"""
            You are a precise, computer-like agent. You can use a list of tools to solve the problem.  
            ## AVAILABLE TOOLS
            Here is a list of available tools in JSON format. You **MUST** use them to interact with the server.
            ```json
            {tools_json_string}
            ```
            ## CRITICAL USAGE INSTRUCTIONS
            **Your response MUST follow these rules EXACTLY, or it will be REJECTED:**
            1.  You **MUST** respond with a single, valid JSON object.
            2.  This JSON object **MUST** be enclosed within `<tool_call>` and `</tool_call>` tags.
            3.  **ABSOLUTELY NO OTHER TEXT, EXPLANATIONS, OR PUNCTUATION** outside the `<tool_call>` tags.
            4.  The JSON object **MUST** have two keys: `"tool_name"` and `"tool_params"`.
            5.  `"tool_name"` **MUST** be a string matching one of the tool names from the list above.
            6.  `"tool_params"` **MUST** be a dictionary containing parameters with the correct data types as defined in the `inputSchema`.
            ## CORRECT RESPONSE EXAMPLE
            To call the '{example_tool_name}' tool, your response must look **EXACTLY** like this (the values are examples, you should use real values):
            <tool_call>
            {example_json_string}
            </tool_call>
        """
        cleaned_prompt = re.sub(r'^\s+', '', prompt_template, flags=re.MULTILINE)
        
        return cleaned_prompt.strip()

    def _create_example_action_json(self, tool_info: Dict, indent: Optional[int] = None) -> str:
        """
        Creates a well-formatted JSON string example for a given tool.
        
        Args:
            tool_info (Dict): The metadata dictionary for a single tool.
            indent (Optional[int]): If provided, formats the JSON string with
                                    the specified indentation for readability.
        """
        tool_name = tool_info.get("name", "tool_name")
        example_params = {}
        
        input_schema = tool_info.get("inputSchema", {})
        
        def get_example_from_schema(schema: dict):
            """
            Recursively generates an example value based on a JSON schema.
            """
            if "anyOf" in schema:
                for option in schema["anyOf"]:
                    if option.get("type") != "null":
                        return get_example_from_schema(option)
                return None    
            
            param_type = schema.get("type")

            if param_type == "object":
                example_obj = {}
                properties = schema.get("properties", {})
                for prop_name, prop_schema in properties.items():
                    example_obj[prop_name] = get_example_from_schema(prop_schema)
                return example_obj
            
            if param_type == "array":
                item_schema = schema.get("items", {})
                if item_schema:
                    return [get_example_from_schema(item_schema)]
                return []
                
            if param_type == "integer":
                return 1 
            elif param_type == "string":
                return "example_value" 
            elif param_type == "boolean":
                return True 
            elif param_type == "number":
                return 1.23
            else:
                return "value"
        
        example_params = get_example_from_schema(input_schema)
        
        example_payload = {
            "tool_name": tool_name,
            "tool_params": example_params
        }
        
        return json.dumps(example_payload, indent=indent, ensure_ascii=False)        
        
    def _parse_action(self, action: str) -> Tuple[str, str, bool]:
        """
        Parses the action string to extract content within <tool_call> tags.

        Returns:
            A tuple (content, parsed_action, is_parsed):
            - json_content (str): The raw content inside the tag.
            - parsed_action (str): The action segment up to the end of the tag.
            - is_parsed (bool): True if the tag was found, False otherwise.
        """
        # only take the first match
        pattern = r"<tool_call>(.*?)</tool_call>"
        match = re.search(pattern, action, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            parsed_action = action[: match.end()]  # including thinking process
            return json_content, parsed_action, True
        else:
            return "", "", False
        
    def _validate_tool_call(self, tool_name: str, tool_params: Dict):
        """
        Validates tool parameters against the JSON Schema provided by the server.
        
        Raises:
            ValueError: If the tool is unknown.
            ValidationError: If the tool parameters do not match the schema.
                            (We will catch this in execute_action)
        """
        # Step 1: Find the schema for the requested tool.
        schema = self._get_schema_for_tool(tool_name)
        
        if schema is None:
            # This is a critical error: the tool name was not found in our cached list.
            valid_tools = [t.get('name', 'N/A') for t in self._tool_metadata]
            raise ValueError(f"Unknown tool_name: '{tool_name}'. Available tools are: {valid_tools}")
        
        # Step 2: Use the jsonschema library to validate the parameters.
        # The `validate` function will raise a `ValidationError` if `tool_params`
        # does not conform to the `schema`.
        validate(instance=tool_params, schema=schema)
    
    def _process_server_response(self, result_obj: types.CallToolResult) -> Tuple[bool, str]:
        """
        Processes the server response to create a clean observation string.

        This function extracts all text from the 'content' blocks and formats the
        observation based on the 'isError' flag. It assumes that both success and
        error details are provided within the 'content' list.

        Args:
            result_obj: The CallToolResult instance returned by the client.

        Returns:
            A tuple (is_success, observation_string).
        """
        # --- Step 1: Extract all text content, regardless of success or error ---
        # This is the single source of truth for the observation message.
        all_text_parts = []
        # Use getattr for safety in case result_obj or its attributes are missing.
        content_list = getattr(result_obj, 'content', [])
        
        if isinstance(content_list, list):
            for item in content_list:
                # Check if the item is a text block and has non-empty text
                if getattr(item, 'type', None) == 'text' and getattr(item, 'text', None):
                    all_text_parts.append(item.text)
        
        extracted_text = "\n".join(all_text_parts).strip()
            
        # --- Step 2: Format the output based on the isError flag ---
        if result_obj.isError:
            # If an error is flagged, but no text is found, provide a generic message.
            # Otherwise, use the text extracted from the content.
            if not extracted_text:
                observation = "[Execution Error: Server indicated an error but provided no details.]"
            else:
                observation = f"[Execution Error: {extracted_text}]"
            
            return False, observation
            
        else: # Success case
            # If the call was successful, but no text is found, inform the agent.
            # This is a valid state, not an error.
            if not extracted_text:
                observation = "<information>Tool executed successfully with no text output.</information>"
            else:
                observation = f"<information>{extracted_text}</information>"
            
            return True, observation      
    
    def _get_schema_for_tool(self, tool_name: str) -> Optional[Dict]:
        """Finds the inputSchema for a given tool name from the cached metadata."""
        for tool_meta in self._tool_metadata:
            if tool_meta.get("name") == tool_name:
                return tool_meta.get("inputSchema")
        return None
