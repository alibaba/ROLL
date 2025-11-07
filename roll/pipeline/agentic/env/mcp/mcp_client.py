from typing import Optional
from mcp import ClientSession
from mcp.client.sse import sse_client
from roll.utils.logging import get_logger

logger = get_logger()

class MCPClient:
    def __init__(self, server_url: str):
        """
        Initialize MCPClient with the server URL.
        Args:
            server_url (str): The URL of the MCP SSE server to connect to.
        """
        self.server_url = server_url
        self._streams_context = None
        self._session_context = None
        self.session: Optional[ClientSession] = None
        
    async def __aenter__(self):
        """
        Enter the async context manager: connect to the MCP SSE server,
        initialize the session and prepare for tool calls.
        
        Returns:
            MCPClient: The connected client instance itself.
        """
        self._streams_context = sse_client(url=self.server_url)       
        self._streams = await self._streams_context.__aenter__()
        self._session_context = ClientSession(*self._streams)
        self.session = await self._session_context.__aenter__()
        
        initialize = await self.session.initialize()
        logger.debug(f"Session initialize: {initialize}")
        
        response = await self.session.list_tools()
        tools = getattr(response, "tools", [])
        logger.debug(f"Connected to server with tools: {[tool.name for tool in tools]}")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the async context: cleanly close session and streams,
        ensuring resources are properly released.
        Args:
            exc_type, exc_val, exc_tb: Exception information if exiting because of an exception.
        """
        if self._session_context:
            await self._session_context.__aexit__(exc_type, exc_val, exc_tb)
            self._session_context = None
        if self._streams_context:
            await self._streams_context.__aexit__(exc_type, exc_val, exc_tb)
            self._streams_context = None
            
    async def tools(self):
        """
        List available tools on the MCP server.
        Returns:
            List of tools info retrieved from the server.
        """  
        try:
            # Call the server to get tools list
            response = await self.session.list_tools()
            if not hasattr(response, 'tools'):
                logger.error(f"Invalid tools response: 'tools' attribute not found in {response}")
                return []
            tools_list = response.tools 
            logger.debug(f"Retrieved {len(tools_list)} tools")
            
            tool_names = [tool.name for tool in tools_list]
            logger.debug(f"Tools available: {tool_names}") 
            
            return tools_list
            
        except Exception as e:
            logger.exception(f"Error retrieving tools: {e}")
            return []
        
    async def call_tool(self, tool_name: str, tool_params: Optional[dict] = None):
        """
        Call a specific tool on the MCP server with optional parameters.
        Args:
            tool_name (str): The name of the tool to call.
            tool_params (Optional[dict]): Parameters to pass to the tool call.
        Returns:
            The result object returned from the MCP server's tool call.
        """
        if tool_params is None:
            tool_params = {}
        result = await self.session.call_tool(tool_name, tool_params)     
        logger.debug(f"Call tool '{tool_name}' with params {tool_params} received: {result}" )  
        
        return result