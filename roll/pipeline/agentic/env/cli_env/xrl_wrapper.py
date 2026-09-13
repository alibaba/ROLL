import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# xrl.sdk.sandbox在使用的时候import，目的是因为后续想要兼容local模式


class SyncXRLWrapper:
    def __init__(
        self,
        sandbox_image: str,
        sandbox_base_url: str,
        auto_clear_seconds: int = 60 * 20,
        debug_info: bool = False
    ):
        """
        Initialize the sync xrl wrapper.

        Args:
            sandbox_image: Docker image for sandbox
            sandbox_base_url: Base URL for sandbox API
            auto_clear_seconds: Auto clear timeout for sandbox
        """
        self.sandbox_image = sandbox_image
        self.sandbox_base_url = sandbox_base_url
        self.auto_clear_seconds = auto_clear_seconds
        self.debug_info = debug_info

        self.sandbox = None
        self.session_id = None
        self.executor = ThreadPoolExecutor(max_workers=1)

        self._loop = None
        self._thread = None

    def _ensure_async_loop(self):
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(self._loop)
                self._loop.run_forever()

            self._thread = threading.Thread(target=run_loop, daemon=True)
            self._thread.start()

    def _run_async(self, coro):
        self._ensure_async_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def start_sandbox(self, session_id: str = None) -> bool:
        # Start the xrl sandbox synchronously.
        try:
            from xrl.sdk.sandbox.client import Sandbox
            from xrl.sdk.sandbox.config import SandboxConfig
            from xrl.sdk.sandbox.request import CreateBashSessionRequest

            config = SandboxConfig(
                image=self.sandbox_image,
                base_url=self.sandbox_base_url,
                auto_clear_seconds=self.auto_clear_seconds,
            )

            self.sandbox = Sandbox(config)
            self._run_async(self.sandbox.start())

            if session_id:
                self.session_id = session_id
            else:
                self.session_id = f"sync-cli-session-{int(time.time())}"

            self._run_async(self.sandbox.create_session(CreateBashSessionRequest(session=self.session_id)))

            return True

        except ImportError:
            print("xrl SDK not available")
            return False
        except Exception as e:
            print(f"Failed to start xrl sandbox: {e}")
            return False

    def execute_command(self, command: str) -> str:
        if self.debug_info:
            print(f"command: {command}")
        if self.sandbox and self.session_id:
            try:
                from xrl.sdk.sandbox.request import BashAction

                response = self._run_async(
                    self.sandbox.run_in_session(BashAction(session=self.session_id, command=command))
                )
                if self.debug_info:
                    print(f"model response: {response}")
                return response.output if hasattr(response, "output") else str(response)
            except Exception as e:
                return f"Sandbox error: {str(e)}"
        else:
            # Fallback to local execution
            raise NotImplementedError("Local Sandbox not available")

    def stop_sandbox(self):
        if self.sandbox:
            try:
                self._run_async(self.sandbox.stop())
            except Exception:
                pass

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self.executor:
            self.executor.shutdown(wait=False)

    def is_available(self) -> bool:
        try:
            sandbox_live = self.sandbox is not None
            return sandbox_live
        except:

            return False

    def __enter__(self):
        self.start_sandbox()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_sandbox()


class SimpleXRLClient:
    """
    Simple synchronous client for xrl operations.

    This provides the simplest possible interface for xrl operations.
    """

    def __init__(self, **kwargs):
        self.wrapper = SyncXRLWrapper(**kwargs)
        self._started = False
        self.debug_info = kwargs.get("debug_info", False)

    def start(self) -> bool:
        """Start the client."""
        if not self._started:
            self._started = self.wrapper.start_sandbox()
        return self._started

    def run(self, command: str) -> str:
        if self.debug_info:
            print(f"in warp run: {command}")
        if not self._started:
            self.start()
        return self.wrapper.execute_command(command)

    def close(self):
        if self._started:
            self.wrapper.stop_sandbox()
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()


# Convenience function for simple usage
def create_sync_client(
    sandbox_image: str = "hub.docker.alibaba-inc.com/chatos/iflow-cli:1.0",
    sandbox_base_url: str = "https://xrl-sandbox.alibaba-inc.com",
    auto_clear_seconds: int = 1200,
    debug_info: bool = False
) -> SimpleXRLClient:
    """
    Create a simple synchronous xrl client.

    Args:
        sandbox_image: Docker image for sandbox
        sandbox_base_url: Base URL for sandbox API
        auto_clear_seconds: Auto clear timeout

    Returns:
        SimpleXRLClient instance
    """
    return SimpleXRLClient(
        sandbox_image=sandbox_image,
        sandbox_base_url=sandbox_base_url,
        auto_clear_seconds=auto_clear_seconds,
        debug_info=debug_info
    )
