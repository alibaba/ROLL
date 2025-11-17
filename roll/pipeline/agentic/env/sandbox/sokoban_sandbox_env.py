import asyncio
import datetime
import json
import logging
import shlex
import time
from typing import Optional

import httpcore
import httpx
from gem import Env
from rock.actions import CreateBashSessionRequest
from rock.sdk.sandbox.client import Sandbox
from rock.sdk.sandbox.config import SandboxConfig

from roll.utils.logging import get_logger


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("rock").setLevel(logging.WARNING)

logger = get_logger()

class SokobanSandboxEnv(Env):
    """
    An environment for the Sokoban game that runs inside a secure, isolated sandbox.

    This class manages the entire lifecycle of a sandboxed environment, including:
    - Starting a dedicated cloud container (sandbox).
    - Managing separate, isolated shell sessions for the server and client.
    - Starting the game server as a background process.
    - Sending client commands to interact with the server.
    - Parsing and returning game state.
    - Cleaning up all resources on close.
    """
    def __init__(self,
                base_url: str = 'http://localhost:8080',
                server_start_timeout: int = 60):
        """
        Args:
            base_url: URL of the ROCK sandbox service (default: localhost:8080)
            server_start_timeout: Seconds to wait for the game server to become healthy.
        """
        self.sandbox_config = SandboxConfig(
            base_url=base_url,
            image='rock-n-roll-registry.cn-hangzhou.cr.aliyuncs.com/rock/sokoban-sandbox:latest',
            auto_clear_seconds=60 * 120,
            startup_timeout=360,
        )

        self.server_start_timeout = server_start_timeout
        self.server_port = 8001

        self.sandbox = None
        self.session = None
        self.server_session = None
        self.last_known_map_str: str = "Game has not been reset yet."

        self._initialized: bool = False

    def reset(self, seed=None, **kwargs):
        """Resets the environment to a new game."""
        self._lazy_init()

        command = f"python client.py reset --seed {seed} --json" if seed else "python client.py reset --json"

        result = self._execute_command(command)

        if isinstance(result, dict):
            game_state = result
            obs = game_state.get('observation', 'Error: Observation not found in reset response.')
            info = game_state.get('info', {})

            suffix = info.get('suffix', 'Map not available on reset.')
            map_part = suffix.split('\n', 1)[-1] if '\n' in suffix else suffix

            self.last_known_map_str = map_part.strip()
            return obs, info
        else:
            raw_output = result
            obs = "Error: Failed to decode JSON from sandbox on reset."
            info = {"error": "JSONDecodeError", "raw_output": raw_output}
            return obs, info

    def step(self, action: str):
        """Executes one step in the environment."""
        # The quotes around '{action}' handle actions that might contain special characters.
        sanitized_action = action.replace('\n', ' ').replace('\r', '') 
        command = f"python client.py action {shlex.quote(sanitized_action)} --json"

        result = self._execute_command(command)

        if isinstance(result, dict):
            game_state = result
            obs = game_state.get('observation', 'Error: Observation not found in step response.')
            reward = game_state.get('reward', 0)
            terminated = game_state.get('terminated', True)
            truncated = game_state.get('truncated', False)
            info = game_state.get('info', {})

            suffix = info.get('suffix', f"No map update. Last known map:\n{self.last_known_map_str}")
            map_part = suffix.split('\n', 1)[-1] if '\n' in suffix else suffix
            self.last_known_map_str = map_part.strip()

            return obs, reward, terminated, truncated, info

        else:
            raw_output = result
            obs = "Error: Failed to decode JSON from sandbox on step."
            reward = 0 
            terminated = True 
            truncated = False
            info = {"error": "JSONDecodeError", "raw_output": raw_output}
            return obs, reward, terminated, truncated, info

    def render(self) -> str:
        return self.last_known_map_str

    def close(self):
        """
        Stops the underlying sandbox instance and releases all cloud resources.
        This is a critical cleanup step.
        """
        if self._initialized and self.sandbox:
            try:
                # The stop() method of the SandboxClient is likely asynchronous,
                # just like start(), so we must run it with asyncio.run().
                asyncio.run(self.sandbox.stop())
            except Exception as e:
                logger.exception("An error occurred while stopping the sandbox.")

    def _wait_for_server(self, timeout: int):
        """
        Periodically polls the server inside the sandbox until it's ready.
        Waits 5 seconds initially, then checks every 3 seconds.
        """
        logger.debug("Waiting for game server to start...")
        time.sleep(5) 
        start_time = time.time()
        # This curl command attempts to connect, but discards output (-s -o /dev/null).
        # It writes a custom string with the HTTP status code (-w "..."), which we can check.
        health_check_command = f'curl -s -o /dev/null -w "HC_CODE_%{{http_code}}_HC_CODE" http://localhost:{self.server_port}/'

        while time.time() - start_time < timeout:
            try:
                response = asyncio.run(self.sandbox.arun(cmd=health_check_command, session="default", mode="normal"))
                # The response from curl is in resp.output.
                if "HC_CODE_200_HC_CODE" in response.output:
                    return
            except Exception as e:
                if time.time() - start_time >= timeout - 3: 
                    logger.warning(f"Health check failed: {e}")

            time.sleep(3) # Wait 3 seconds before retrying.

        raise RuntimeError(f"Server did not start within the {timeout}s timeout.")

    def _wait_for_sandbox_alive(self, timeout: int = 180):
        """
        Periodically polls the sandbox service until its 'is_alive' flag is True.
        This prevents race conditions between sandbox startup and command execution.

        Args:
            timeout: Maximum seconds to wait for sandbox to become alive

        Raises:
            RuntimeError: If sandbox doesn't become alive within timeout
        """
        logger.debug(f"Waiting up to {timeout}s for sandbox to become 'alive'...") 
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                status_response = asyncio.run(self.sandbox.get_status())

                if status_response.is_alive:
                    return

                logger.debug(f"Sandbox is not 'alive' yet, waiting... Current status details: {repr(status_response)}")

            except Exception as e:
                logger.debug(f"An error occurred while checking sandbox status: {e}. Retrying...")

            time.sleep(3) # Wait 3 seconds before retrying.

        raise RuntimeError(f"Sandbox did not become 'alive' within the {timeout}s timeout.")

    def _lazy_init(self):
        """
        Performs one-time, heavy initialization of the sandbox environment.

        This method is called "lazily" on the first `reset()` to
        avoid the "thundering herd" problem in distributed settings, where
        multiple workers would try to initialize simultaneously. It handles
        sandbox creation, session setup, and game server startup.
        """
        if self._initialized:
            return

        max_attempts = 3
        base_wait_time = 10.0  # Seconds to wait after the first failure.

        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(f"Initialization attempt [{attempt}/{max_attempts}] starting...")

                # 1. Start the Sandbox Container
                # This provisions the underlying container resource via the sandbox service.
                self.sandbox = Sandbox(self.sandbox_config)

                asyncio.run(self.sandbox.start())
                logger.info("Sandbox %s created on host %s (%s)",
                            self.sandbox._sandbox_id, self.sandbox._host_name, self.sandbox._host_ip)

                self._wait_for_sandbox_alive(timeout=180)

                # 2. Create Isolated Execution Sessions
                # Two separate bash sessions are created to prevent the game server's
                # logs from interfering with the clean JSON output of client commands.
                # The 'default' session is used for executing client commands (reset, step).
                self.session = asyncio.run(self.sandbox.create_session(CreateBashSessionRequest(session="default")))
                # The 'server_session' is to contain the noisy server process.
                self.server_session = asyncio.run(self.sandbox.create_session(CreateBashSessionRequest(session="server_session")))

                # 3. Launch the Game Server as a Background Process
                # The '&' runs the command as a background process, allowing the run_in_session call to return immediately.
                start_command = "cd /app && python server.py &"
                asyncio.run(self.sandbox.arun(cmd=start_command, session="server_session", mode="normal"))

                # 4. Wait for the Server to Become Responsive
                self._wait_for_server(timeout=self.server_start_timeout)

                self._initialized = True
                return  # On success, exit the function immediately and do not retry.

            except (httpx.ReadError, httpcore.ReadError, httpx.ConnectError) as e:
                # Catch only the specific network errors that we know are safe to retry.
                sandbox_id = self.sandbox.sandbox_id if self.sandbox else "N/A"
                logger.warning(
                    f"Attempt [{attempt}/{max_attempts}] failed with a retryable network error: {type(e).__name__} "
                    f"for Sandbox ID: {sandbox_id}."
                )

                # Simple cleanup to prepare for the next attempt.
                if self.sandbox:
                    try:
                        asyncio.run(self.sandbox.stop())
                    except Exception as stop_error:
                        logger.error(f"Error while stopping failed sandbox {sandbox_id}: {stop_error}")
                    self.sandbox = None

                if attempt == max_attempts:
                    logger.error(f"All {max_attempts} initialization attempts failed. Giving up.")
                    raise RuntimeError("Failed to initialize sandbox environment after multiple network errors.") from e

                # Wait longer on each subsequent retry.
                wait_time = base_wait_time * attempt
                logger.info(f"Waiting {wait_time:.1f} seconds before retrying...")
                time.sleep(wait_time)

            except Exception as e:
                # Catch all other unexpected errors (e.g., permission issues, code bugs).
                # These are typically non-recoverable, so we fail immediately without retrying.
                logger.exception("A critical, non-retryable error occurred during lazy initialization.")
                if self.sandbox:
                    try:
                        asyncio.run(self.sandbox.stop())
                    except Exception as stop_error:
                        logger.error(f"Error while stopping failed sandbox after critical error: {stop_error}")

                # Raise the exception directly to terminate this worker.
                raise RuntimeError("Failed to initialize sandbox environment due to a critical error.") from e

    def _execute_command(self, command: str) -> Optional[dict]:
        """
        A low-level helper to execute a command and parse its JSON output.
        This is the part that is truly duplicated between reset and step.
        """
        raw_output = ""
        try:
            response = asyncio.run(self.sandbox.arun(cmd=f"cd /app && {command}", session="default", mode="normal"))
            raw_output = response.output.strip()
            return json.loads(raw_output)
        except json.JSONDecodeError:
            sandbox_info = self._get_sandbox_info_str()
            logger.error("%s Failed to decode JSON from sandbox. Raw output: >>>%s<<<", sandbox_info, raw_output)
            return raw_output
        except Exception as e:
            sandbox_info = self._get_sandbox_info_str()
            logger.exception("%s An unexpected error occurred executing command: %s. Error: %s. ", sandbox_info, command, e)
            return f"{sandbox_info} Unexpected sandbox execution error: {e}"   

    def _get_sandbox_info_str(self) -> str:
        """
        Generates a standardized string with sandbox context for logging.

        Returns:
            A formatted string containing the current timestamp, sandbox ID,
            and host information, suitable for prefixing log messages.
            e.g., "Timestamp: ... | Context: [Sandbox: ... on Host: ...]".
        """
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        sandbox_info = "[Sandbox: Not Initialized]"
        if self.sandbox and self.sandbox.sandbox_id:
            sandbox_info = (
                f"[Sandbox: {self.sandbox.sandbox_id} "
                f"on Host: {self.sandbox.host_name} ({self.sandbox.host_ip})]"
            )

        return f"Timestamp: {current_time_str} | Context: {sandbox_info}"
