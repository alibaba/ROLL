import asyncio
import logging
import socket
import threading
import time
from typing import Callable, Dict, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


class ProxyServer:
    """
    Proxy server that routes requests to environment managers.

    Each EnvironmentWorker creates a ProxyServer instance that:
    1. Starts a FastAPI server on a specified port
    2. Exposes the server URL (ip:port) for external access
    3. Routes requests to registered environment managers by env_id

    Attributes:
        host (str): Server host address
        port (int): Server port
        url (str): Full server URL (http://host:port)
        app (FastAPI): FastAPI application instance
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 0):
        """
        Initialize the proxy server.

        Args:
            host (str): Host address to bind to (default: 0.0.0.0)
            port (int): Port to listen on (0 for auto-assigned port)
        """
        self.host = host
        self.port = self._find_available_port(port) if port == 0 else port
        self.url = f"http://{self._get_public_ip()}:{self.port}"

        self.app: Optional[FastAPI] = None
        self.server_instance: Optional[uvicorn.Server] = None
        self.server_thread: Optional[threading.Thread] = None

        # env_id -> request handler mapping
        self._handlers: Dict[int, Callable] = {}

    def _find_available_port(self, preferred_port: int = 0) -> int:
        """
        Find an available port.

        Args:
            preferred_port (int): Preferred port (0 for any available port)

        Returns:
            int: Available port number
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if preferred_port == 0:
                s.bind(("", 0))
                s.listen(1)
                port = s.getsockname()[1]
            else:
                s.bind(("", preferred_port))
                port = preferred_port
        return port

    def _get_public_ip(self) -> str:
        """
        Get the public IP address of this machine.

        Returns:
            str: Public IP address or hostname
        """
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            return ip_address
        except Exception as e:
            logger.warning(f"Failed to get public IP: {e}, using 0.0.0.0")
            return "0.0.0.0"

    def register_handler(self, env_id: int, handler: Callable):
        """
        Register a request handler for a specific environment.

        Args:
            env_id (int): Environment identifier
            handler (Callable): Async function to handle requests for this env_id
                               Should accept (Request) and return Response
        """
        self._handlers[env_id] = handler
        logger.info(f"Registered handler for env_id={env_id}")

    def unregister_handler(self, env_id: int):
        """
        Unregister a handler for a specific environment.

        Args:
            env_id (int): Environment identifier
        """
        if env_id in self._handlers:
            del self._handlers[env_id]
            logger.info(f"Unregistered handler for env_id={env_id}")

    async def _route_request(self, request: Request):
        """
        Route request to the appropriate environment handler.

        Args:
            env_id (int): Environment identifier from request
            request (Request): FastAPI request object

        Returns:
            Response: Handler response or error response
        """
        env_id = await self._parser_from_request(request)
        if env_id not in self._handlers:
            logger.warning(f"No handler registered for env_id={env_id}")
            return JSONResponse(
                status_code=404,
                content={"error": f"No handler found for env_id={env_id}"}
            )

        handler = self._handlers[env_id]
        try:
            return await handler(request)
        except Exception as e:
            logger.error(f"Handler error for env_id={env_id}: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"Handler error: {str(e)}"}
            )

    async def _health_check(self):
        """Health check endpoint."""
        return JSONResponse(
            content={
                "status": "healthy",
                "url": self.url,
                "registered_env_ids": list(self._handlers.keys())
            }
        )

    async def _state_endpoint(self):
        """State endpoint to query registered handlers."""
        return JSONResponse(
            content={
                "url": self.url,
                "host": self.host,
                "port": self.port,
                "registered_env_ids": list(self._handlers.keys()),
                "num_handlers": len(self._handlers)
            }
        )

    async def _start_async_server(self):
        """Start the FastAPI server asynchronously."""
        self.app = FastAPI(title="Environment Proxy Server")

        # Register routes
        self.app.add_api_route("/v1/chat/completions", self._route_request, methods=["POST"])
        self.app.add_api_route("/health", self._health_check, methods=["GET", "POST"])
        self.app.add_api_route("/state", self._state_endpoint, methods=["GET", "POST"])
        self.app.add_api_route("/v1/state", self._state_endpoint, methods=["GET", "POST"])

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        self.server_instance = uvicorn.Server(config)
        await self.server_instance.serve()

    def start(self, timeout: int = 30):
        """
        Start the proxy server in a background thread.

        Args:
            timeout (int): Maximum time to wait for server startup (seconds)

        Raises:
            RuntimeError: If server fails to start within timeout period
        """
        def run_server_in_new_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._start_async_server())
            loop.close()

        self.server_thread = threading.Thread(target=run_server_in_new_loop)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Wait for server to become ready
        interval = 0.5
        start_time = time.time()
        server_ready = False

        logger.info(f"Waiting for proxy server to start on {self.host}:{self.port}...")

        while time.time() - start_time < timeout:
            try:
                with socket.create_connection(('localhost', self.port), timeout=interval):
                    logger.info(f"Proxy server started successfully on port {self.port}")
                    logger.info(f"Server URL: {self.url}")
                    server_ready = True
                    break
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(interval)

        if not server_ready:
            raise RuntimeError(
                f"Proxy server failed to start within {timeout} seconds on port {self.port}"
            )

    def stop(self, timeout: int = 5):
        """
        Stop the proxy server and wait for thread to exit.

        Args:
            timeout (int): Maximum time to wait for server shutdown (seconds)
        """
        if self.server_instance:
            logger.info(f"Stopping proxy server on port {self.port}")
            self.server_instance.should_exit = True

        # Wait for server thread to finish
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=timeout)
            if self.server_thread.is_alive():
                logger.warning(f"Server thread did not exit within {timeout} seconds")
            else:
                logger.info("Proxy server stopped successfully")
            
            
    async def _parser_from_request(self, request: Request) -> int:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            return -1
            
        try:
            # 假设 Agent 发送的是 "Bearer 1001"
            token = auth_header.split()[-1]
            return int(token) # 得到 1001
        except (ValueError, IndexError):
            return -1    
            
            
async def mock_handler(request: Request):
    """模拟一个环境处理逻辑"""
    return JSONResponse(content={"message": "Hello from Environment Handler!", "env_id": 1})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 1. 初始化服务器（使用固定端口方便调试，如 8000）
    server = ProxyServer(host="127.0.0.1", port=8000)
    
    # 2. 注册一个处理器 (env_id=1)
    server.register_handler(env_id=1001, handler=mock_handler)
    
    # 3. 启动服务器
    try:
        server.start()
        print(f"服务器已启动: {server.url}")
        
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("服务器已停止")            
