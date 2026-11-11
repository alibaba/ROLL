import datetime
import json
import time
import uuid
from typing import Optional, Union

from roll.pipeline.agentic.env.sandbox.sokoban_sandbox_env import SokobanSandboxEnv
from roll.utils.logging import get_logger

logger = get_logger()

SOKOBAN_SANDBOX_IMAGE = "rock-n-roll-registry.cn-hangzhou.cr.aliyuncs.com/rock/sokoban-sandbox:latest"


class SokobanRuntaSandboxEnv(SokobanSandboxEnv):
    """
    The sandboxed Sokoban environment, hosted on a Runta runtime (https://runta.com/docs/).

    A Runta runtime is a full cloud machine with Docker preinstalled rather than a
    container, so the game image runs as one container inside the runtime and every
    client command goes through ``docker exec``. The game protocol (``reset``, ``step``,
    ``render``) is inherited unchanged from :class:`SokobanSandboxEnv`.

    Requires the ``runta-sdk`` package (``pip install runta-sdk``) and credentials in
    ``RUNTA_TOKEN`` (or the config file written by ``runta login``).
    """

    CONTAINER_NAME = "roll-sokoban"

    def __init__(
        self,
        image: str = SOKOBAN_SANDBOX_IMAGE,
        vcpus: int = 1,
        memory_mib: int = 2048,
        server_start_timeout: int = 60,
        startup_timeout: int = 600,
        max_lifetime_seconds: int = 60 * 120,
        endpoint: Optional[str] = None,
    ):
        """
        Args:
            image: Docker image that contains the game server and client under ``/app``.
            vcpus: Virtual CPUs of the runtime.
            memory_mib: Memory of the runtime in MiB.
            server_start_timeout: Seconds to wait for the game server to become healthy.
            startup_timeout: Seconds to wait for the runtime to boot and pull the image.
            max_lifetime_seconds: Server-side lifetime of the runtime. Runta shuts it down
                after this long even if the worker died without calling ``close()``.
            endpoint: Runta API endpoint; defaults to ``RUNTA_ENDPOINT`` or the SDK default.
        """
        # The parent constructor only builds ROCK's sandbox config, so it is not called.
        self.image = image
        self.vcpus = vcpus
        self.memory_mib = memory_mib
        self.server_start_timeout = server_start_timeout
        self.startup_timeout = startup_timeout
        self.max_lifetime_seconds = max_lifetime_seconds
        self.endpoint = endpoint
        self.server_port = 8001

        self.client = None
        self.runtime = None
        self.last_known_map_str: str = "Game has not been reset yet."

        self._initialized: bool = False

    def close(self):
        """Deletes the runtime and releases all cloud resources."""
        runtime, client = self.runtime, self.client
        self.runtime = None
        self.client = None
        self._initialized = False
        try:
            if runtime is not None:
                runtime.delete()
        except Exception:
            logger.exception("An error occurred while deleting the Runta runtime.")
        finally:
            if client is not None:
                client.close()

    def _container_exec(self, command: str, timeout: int = 120):
        """Runs a shell command in the game container and returns the SDK result."""
        return self.runtime.exec(["docker", "exec", self.CONTAINER_NAME, "bash", "-c", command], timeout=timeout)

    def _wait_for_server(self, timeout: int):
        """Polls the game server inside the container until it answers."""
        logger.debug("Waiting for game server to start...")
        health_check_command = f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{self.server_port}/"
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._container_exec(health_check_command, timeout=30).stdout_text.strip() == "200":
                return
            time.sleep(3)

        raise RuntimeError(f"Server did not start within the {timeout}s timeout.")

    def _start_runtime(self):
        """Creates the runtime, starts the game container and launches the game server."""
        from runta import Runta

        self.client = Runta(endpoint=self.endpoint) if self.endpoint else Runta()
        self.runtime = self.client.runtimes.create(
            f"roll-sokoban-{uuid.uuid4().hex[:8]}",
            vcpus=self.vcpus,
            memory_mib=self.memory_mib,
            auto_shutdown_after_secs=self.max_lifetime_seconds,
            auto_delete_after_secs=600,
            timeout=self.startup_timeout,
        )
        logger.info("Runta runtime %s created", self.runtime.id)

        # --network host keeps the game server reachable on the runtime's localhost.
        started = self.runtime.exec(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.CONTAINER_NAME,
                "--network",
                "host",
                self.image,
                "sleep",
                "infinity",
            ],
            timeout=self.startup_timeout,
        )
        if started.exit_code != 0:
            raise RuntimeError(f"Failed to start image {self.image}: {started.stderr_text[-400:]}")

        # The server runs detached with its own log file, so its output never mixes
        # with the JSON printed by client commands.
        launched = self.runtime.exec(
            [
                "docker",
                "exec",
                "-d",
                self.CONTAINER_NAME,
                "bash",
                "-c",
                "cd /app && exec python server.py > /tmp/server.log 2>&1",
            ],
            timeout=60,
        )
        if launched.exit_code != 0:
            raise RuntimeError(f"Failed to launch the game server: {launched.stderr_text[-400:]}")

        self._wait_for_server(timeout=self.server_start_timeout)

    def _lazy_init(self):
        """
        Performs one-time, heavy initialization of the sandbox environment.

        Called "lazily" on the first `reset()` so that workers in a distributed run do
        not all provision their runtimes at construction time. Transient Runta API
        errors are retried; anything else fails the worker immediately.
        """
        if self._initialized:
            return

        from runta import ApiError

        max_attempts = 3
        base_wait_time = 10.0  # Seconds to wait after the first failure.

        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(f"Initialization attempt [{attempt}/{max_attempts}] starting...")
                self._start_runtime()
                self._initialized = True
                return

            except (ApiError, TimeoutError) as e:
                logger.warning(f"Attempt [{attempt}/{max_attempts}] failed with a Runta API error: {e}")
                self.close()

                if attempt == max_attempts:
                    logger.error(f"All {max_attempts} initialization attempts failed. Giving up.")
                    raise RuntimeError(
                        "Failed to initialize sandbox environment after multiple Runta API errors."
                    ) from e

                wait_time = base_wait_time * attempt
                logger.info(f"Waiting {wait_time:.1f} seconds before retrying...")
                time.sleep(wait_time)

            except Exception as e:
                logger.exception("A critical, non-retryable error occurred during lazy initialization.")
                self.close()
                raise RuntimeError("Failed to initialize sandbox environment due to a critical error.") from e

    def _execute_command(self, command: str) -> Union[dict, str]:
        """Executes a client command in the game container and parses its JSON output."""
        raw_output = ""
        try:
            raw_output = self._container_exec(f"cd /app && {command}").stdout_text.strip()
            return json.loads(raw_output)
        except json.JSONDecodeError:
            sandbox_info = self._get_sandbox_info_str()
            logger.error("%s Failed to decode JSON from sandbox. Raw output: >>>%s<<<", sandbox_info, raw_output)
            return raw_output
        except Exception as e:
            sandbox_info = self._get_sandbox_info_str()
            logger.exception(
                "%s An unexpected error occurred executing command: %s. Error: %s. ", sandbox_info, command, e
            )
            return f"{sandbox_info} Unexpected sandbox execution error: {e}"

    def _get_sandbox_info_str(self) -> str:
        """Generates a standardized string with sandbox context for logging."""
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        sandbox_info = f"[Runta runtime: {self.runtime.id}]" if self.runtime else "[Sandbox: Not Initialized]"
        return f"Timestamp: {current_time_str} | Context: {sandbox_info}"
