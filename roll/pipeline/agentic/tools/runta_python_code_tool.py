import atexit
import threading
import uuid
import weakref
from typing import List, Optional, Tuple

from gem.utils.constants import BASE_IMPORTS
from gem.utils.sandbox import check_forbidden_imports

from roll.pipeline.agentic.tools.python_code_tool import PythonCodeTool
from roll.utils.logging import get_logger

logger = get_logger()

# Runtimes still alive at interpreter exit are deleted here; the server-side
# lifetime covers workers that are killed without running exit hooks.
_LIVE_TOOLS: "weakref.WeakSet[RuntaPythonCodeTool]" = weakref.WeakSet()


def _close_live_tools() -> None:
    for tool in list(_LIVE_TOOLS):
        tool.close()


atexit.register(_close_live_tools)

# What gem's run_python answers for code rejected by check_forbidden_imports; gem has no
# constant for it.
FORBIDDEN_CODE_MESSAGE = "Execution blocked: Code contains potentially dangerous operations or imports."


class RuntaPythonCodeTool(PythonCodeTool):
    """
    ``python_code`` tool that executes the model's code on a Runta runtime
    (https://runta.com/docs/) instead of in the rollout worker.

    Each tool instance owns one runtime, created on the first execution and reused
    afterwards. Every execution starts a fresh ``python3`` interpreter, so variables do
    not carry over between calls while files written by earlier code do.

    Code is screened and retried like the local tool (gem's ``run_python`` with
    ``sandbox_type="none"``), using gem's own helpers: code that
    ``check_forbidden_imports`` rejects is not executed, and code failing with
    "is not defined" runs once more with gem's ``BASE_IMPORTS`` prepended. The modules
    ``BASE_IMPORTS`` imports (``numpy`` and ``pandas`` among them) must therefore exist on
    the runtime or in ``image``. Two details of gem's local runner are not reproduced: a
    program that is exactly ``...`` runs (and prints nothing) instead of being reported
    as a syntax error, and a timeout is reported in Runta's wording, not gem's.

    Requires the ``runta-sdk`` package (``pip install runta-sdk``) and credentials in
    ``RUNTA_TOKEN`` (or the config file written by ``runta login``).
    """

    CONTAINER_NAME = "roll-python"

    def __init__(
        self,
        timeout: int = 5,
        keep_error_last_line: bool = False,
        tool_instruction: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        image: Optional[str] = None,
        vcpus: int = 1,
        memory_mib: int = 1024,
        startup_timeout: int = 600,
        max_lifetime_seconds: int = 60 * 120,
        endpoint: Optional[str] = None,
    ):
        """
        Args:
            timeout: Seconds one execution may run before it is terminated.
            keep_error_last_line: Keep only the last line of stderr in the observation.
            tool_instruction: Overrides the default tool instruction.
            patterns: Overrides the regular expressions that extract code from an action.
            image: Docker image (with ``python3``) to run the code in. By default the code
                runs on the runtime's own Python.
            vcpus: Virtual CPUs of the runtime.
            memory_mib: Memory of the runtime in MiB.
            startup_timeout: Seconds to wait for the runtime to boot and pull ``image``.
            max_lifetime_seconds: Server-side lifetime of the runtime. Runta shuts it down
                after this long; the next execution then starts a new one.
            endpoint: Runta API endpoint; defaults to ``RUNTA_ENDPOINT`` or the SDK default.
        """
        super().__init__(
            timeout=timeout,
            sandbox_type="none",  # unused: execution is routed to Runta below
            keep_error_last_line=keep_error_last_line,
            tool_instruction=tool_instruction,
            patterns=patterns,
        )
        self.image = image
        self.vcpus = vcpus
        self.memory_mib = memory_mib
        self.startup_timeout = startup_timeout
        self.max_lifetime_seconds = max_lifetime_seconds
        self.endpoint = endpoint

        self.client = None
        self.runtime = None
        self._lock = threading.Lock()

    def _ensure_runtime(self):
        """Returns the tool's runtime, creating it (and its container) when needed."""
        if self.runtime is not None:
            return self.runtime

        from runta import Runta

        client = Runta(endpoint=self.endpoint) if self.endpoint else Runta()
        runtime = None
        try:
            # wait=False hands back the runtime as soon as it exists server-side, so it
            # is deleted below even when it never becomes ready.
            runtime = client.runtimes.create(
                f"roll-python-{uuid.uuid4().hex[:8]}",
                vcpus=self.vcpus,
                memory_mib=self.memory_mib,
                auto_shutdown_after_secs=self.max_lifetime_seconds,
                auto_delete_after_secs=600,
                wait=False,
            )
            runtime.wait_until_ready(timeout=self.startup_timeout)
            if self.image is not None:
                started = runtime.exec(
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
        except BaseException:
            try:
                if runtime is not None:
                    runtime.delete()
            finally:
                client.close()
            raise

        self.client, self.runtime = client, runtime
        _LIVE_TOOLS.add(self)
        logger.info("Runta runtime %s created for python code tool", runtime.id)
        return runtime

    def _python_command(self) -> List[str]:
        if self.image is None:
            return ["python3", "-"]
        return ["docker", "exec", "-i", self.CONTAINER_NAME, "python3", "-"]

    def _run_python(self, code: str) -> Tuple[bool, str, str]:
        """Runs ``code`` like gem's ``run_python`` does, but on the runtime."""
        if check_forbidden_imports(code):
            return False, "", FORBIDDEN_CODE_MESSAGE

        success, stdout, stderr = self._exec_python(code)
        if not success and "is not defined" in stderr:
            retry = self._exec_python(BASE_IMPORTS + "\n" + code)
            # A runtime without the BASE_IMPORTS modules fails on the prepended imports;
            # the model should then see its own error, not a missing-module one.
            if retry[0] or "No module named" not in retry[2]:
                success, stdout, stderr = retry
        return success, stdout, stderr

    def _exec_python(self, code: str) -> Tuple[bool, str, str]:
        """Runs ``code`` on the runtime and returns ``(success, stdout, stderr)``."""
        from runta import ApiError

        with self._lock:
            # A runtime that reached its lifetime (or was deleted) fails with an API
            # error; replace it once before reporting the failure to the model.
            for attempt in range(2):
                try:
                    # The program is sent on stdin, so it is never interpolated into a shell.
                    result = self._ensure_runtime().exec(self._python_command(), timeout=self.timeout, stdin=code)
                    # stderr is stripped as gem does, so keep_error_last_line finds the error.
                    return result.exit_code == 0, result.stdout_text, result.stderr_text.strip()
                except (ApiError, TimeoutError) as e:
                    logger.warning(f"Runta execution failed (attempt {attempt + 1}/2): {e}")
                    self._release()
                    error = str(e)
            return False, "", f"Sandbox error: {error}"

    def _release(self) -> None:
        runtime, client = self.runtime, self.client
        self.runtime = None
        self.client = None
        try:
            if runtime is not None:
                runtime.delete()
        except Exception as e:
            logger.warning(f"Failed to delete Runta runtime: {e}")
        finally:
            if client is not None:
                client.close()

    def close(self) -> None:
        """Deletes the runtime and releases all cloud resources."""
        with self._lock:
            self._release()
        _LIVE_TOOLS.discard(self)
