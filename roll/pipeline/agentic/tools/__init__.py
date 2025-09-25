from roll.pipeline.agentic.tools.registration import register_tools

register_tools(tool_id="python_code", entry_point="roll.pipeline.agentic.tools.python_code_tool:PythonCodeTool")
register_tools(tool_id="search", entry_point="gem.tools.search_tool:SearchTool")
register_tools(tool_id="mcp", entry_point="roll.pipeline.agentic.tools.mcp_tool:MCPTool")
