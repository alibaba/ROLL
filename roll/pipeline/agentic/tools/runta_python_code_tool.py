import atexit
import shlex
import threading
import uuid
import weakref
from typing import List, Optional, Tuple

from gem.utils.constants import BASE_IMPORTS
from gem.utils.sandbox import check_forbidden_imports

from roll.pipeline.agentic.tools.python_code_tool import PythonCodeTool
from roll.utils.logging import get_logger
from roll.utils.runta_lease import RuntaLease

logger = get_logger()

# Runtimes still alive at interpreter exit are deleted here; the renewable lease
# (RuntaLease) covers workers that are killed without running exit hooks.
_LIVE_TOOLS: "weakref.WeakSet[RuntaPythonCodeTool]" = weakref.WeakSet()


def _close_live_tools() -> None:
    for tool in list(_LIVE_TOOLS):
        tool.close()


atexit.register(_close_live_tools)

# What gem's run_python answers for code rejected by check_forbidden_imports; gem has no
# constant for it.
FORBIDDEN_CODE_MESSAGE = "Execution blocked: Code contains potentially dangerous operations or imports."

# Runta's exec timeout kills only the ``docker exec`` client on the runtime; the program
# inside the container keeps running. With ``image`` the program therefore runs under
# ``_TRACK``, which records its process group in a PID file (``$1``, removed again when
# the program ends in time), and ``_KILL`` stops a timed-out program through that file.
# ``setsid -w`` gives the program a group of its own where the exec'd shell does not
# already lead one. The wrapper never reads stdin, which carries the program, and sends
# the shell's own messages ("Killed", for one) to /dev/null while the program, started
# in a subshell, keeps the real stderr through fd 3: a program that ends in time
# produces the output and exit code it would without the wrapper.
_TRACKED = 'exec 3>&2 2>/dev/null; echo $$ > "$1"; (exec python3 - 2>&3 3>&-); rc=$?; rm -f "$1"; exit $rc'
_TRACK = (
    "if setsid -w true </dev/null >/dev/null 2>&1; then "
    f'exec setsid -w sh -c {shlex.quote(_TRACKED)} roll-exec "$@"; fi; {_TRACKED}'
)
_KILL = (
    'pid=$(cat "$1" 2>/dev/null) || exit 0; rm -f "$1"; '
    'kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; exit 0'
)
# Exit code Runta reports for a command that ran into its timeout.
TIMEOUT_EXIT_CODE = 124


class RuntaPythonCodeTool(PythonCodeTool):
    """
    ``python_code`` tool that executes the model's code on a Runta runtime
    (https://runta.com/docs/) instead of in the rollout worker.

    Each tool instance owns one runtime, created on the first execution and reused
    afterwards. Every execution starts a fresh ``python3`` interpreter, so variables do
    not carry over between calls while files written by earlier code do.

    The runtime lives on a renewable lease: Runta shuts it down ``lease_seconds`` after
    the last renewal, and a daemon heartbeat thread renews the lease every
    ``lease_seconds / 3`` for as long as this tool exists, up to ``max_lifetime_seconds``
    in total. A worker that is killed without running ``close()`` or its exit hooks
    (``ray stop --force``) stops renewing, so its runtime is shut down within one lease
    and deleted ten minutes later. A runtime that was shut down is replaced on the next
    execution.

    A program that runs into ``timeout`` is killed: by Runta on the runtime's own
    Python, and with ``image`` by a second ``docker exec`` that kills the program's
    process group inside the container (which needs ``sh`` in the image).

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
        lease_seconds: int = 600,
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
            max_lifetime_seconds: Hard cap on the lifetime of one runtime. The lease is
                never renewed beyond it, so Runta shuts the runtime down after this long
                even while the tool is in use; the next execution then starts a new one.
            endpoint: Runta API endpoint; defaults to ``RUNTA_ENDPOINT`` or the SDK default.
            lease_seconds: Seconds the runtime survives without a renewal, i.e. how long
                a runtime outlives a worker that was killed. The heartbeat renews it
                every ``lease_seconds / 3``; keep it well above the time one Runta API
                request may take.
        """
        if lease_seconds < 1:
            raise ValueError(f"lease_seconds must be at least 1, got {lease_seconds}")
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
        self.lease_seconds = lease_seconds

        self.client = None
        self.runtime = None
        self._lease: Optional[RuntaLease] = None
        self._lock = threading.Lock()

    def _ensure_runtime(self):
        """Returns the tool's runtime, creating it (and its container) when needed."""
        if self.runtime is not None:
            return self.runtime

        from runta import Runta

        client = Runta(endpoint=self.endpoint) if self.endpoint else Runta()
        runtime = None
        # The lifetime cap is counted from here, before the runtime exists.
        lease = RuntaLease(
            lease_seconds=self.lease_seconds,
            max_lifetime_seconds=self.max_lifetime_seconds,
            endpoint=self.endpoint,
            owner=self,
        )
        try:
            # wait=False hands back the runtime as soon as it exists server-side, so it
            # is deleted below even when it never becomes ready.
            runtime = client.runtimes.create(
                f"roll-python-{uuid.uuid4().hex[:8]}",
                vcpus=self.vcpus,
                memory_mib=self.memory_mib,
                auto_shutdown_after_secs=lease.initial_seconds,
                auto_delete_after_secs=600,
                wait=False,
            )
            # Renewed from now on: booting and pulling the image may take longer than a lease.
            lease.start(runtime.id)
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
            lease.stop()
            try:
                if runtime is not None:
                    runtime.delete()
            finally:
                client.close()
            raise

        self.client, self.runtime, self._lease = client, runtime, lease
        _LIVE_TOOLS.add(self)
        logger.info("Runta runtime %s created for python code tool", runtime.id)
        return runtime

    def _python_command(self, pidfile: Optional[str] = None) -> List[str]:
        if self.image is None:
            return ["python3", "-"]
        return ["docker", "exec", "-i", self.CONTAINER_NAME, "sh", "-c", _TRACK, "roll-exec", pidfile]

    def _kill_in_container(self, runtime, pidfile: str) -> None:
        """Kills the process group a timed-out program left running in the container."""
        try:
            runtime.exec(
                ["docker", "exec", "-u", "0", self.CONTAINER_NAME, "sh", "-c", _KILL, "roll-kill", pidfile],
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"Could not stop the timed-out program on Runta runtime {runtime.id}: {e}")

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
            # A runtime whose lease lapsed, that reached its lifetime or that was deleted
            # fails with an API error; replace it once before reporting the failure to
            # the model.
            for attempt in range(2):
                try:
                    runtime = self._ensure_runtime()
                    pidfile = f"/tmp/roll-exec-{uuid.uuid4().hex[:12]}.pid" if self.image is not None else None
                    # The program is sent on stdin, so it is never interpolated into a shell.
                    result = runtime.exec(self._python_command(pidfile), timeout=self.timeout, stdin=code)
                    if pidfile is not None and result.exit_code == TIMEOUT_EXIT_CODE:
                        self._kill_in_container(runtime, pidfile)
                    # stderr is stripped as gem does, so keep_error_last_line finds the error.
                    return result.exit_code == 0, result.stdout_text, result.stderr_text.strip()
                except (ApiError, TimeoutError) as e:
                    logger.warning(f"Runta execution failed (attempt {attempt + 1}/2): {e}")
                    self._release()
                    error = str(e)
            return False, "", f"Sandbox error: {error}"

    def _release(self) -> None:
        runtime, client, lease = self.runtime, self.client, self._lease
        self.runtime = None
        self.client = None
        self._lease = None
        # The heartbeat is stopped first, so it does not renew a runtime being deleted.
        if lease is not None:
            lease.stop()
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
