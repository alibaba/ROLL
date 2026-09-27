"""Validate an optional filesystem for Ray's session and object-store files."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


_DEFAULT_MIN_FREE_BYTES = 128 * 1024**3
_FORCE_TRUE = {"1", "true", "yes", "on"}


def should_force_new_cluster(temp_dir: Path | None) -> bool:
    """Return whether startup must avoid an existing Ray cluster.

    A configured temporary directory is an ownership boundary: Ray session and
    spill files belong to this invocation, so connecting to the host default
    cluster would silently violate the caller's storage contract.
    """
    flag = os.environ.get("ROLL_RAY_FORCE_NEW_CLUSTER", "").strip().lower()
    return temp_dir is not None or flag in _FORCE_TRUE


def resolve_ray_session_dir(temp_dir: Path | None) -> Path | None:
    """Resolve Ray's ``session_latest`` only when it remains inside temp_dir."""
    if temp_dir is None:
        return None
    root = Path(temp_dir).expanduser().resolve()
    link = root / "session_latest"
    try:
        session = link.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    if not session.is_dir() or not session.is_relative_to(root):
        return None
    return session


def owned_ray_process_ids(session_dir: str | os.PathLike[str], process_table: str) -> list[int]:
    """Extract Ray PIDs whose command line names exactly this session directory."""
    marker = str(Path(session_dir).expanduser().resolve())
    markers = tuple(
        f"{flag}={value}"
        for flag, value in (
            *(
                (flag, marker)
                for flag in ("--session-dir", "--session_dir", "--temp-dir", "--temp_dir")
            ),
            *(
                (flag, f"{marker}/logs")
                for flag in (
                    "--log-dir", "--log_dir", "--logs-dir", "--stdout-filepath",
                    "--stdout_filepath", "--stderr-filepath", "--stderr_filepath",
                )
            ),
        )
    )
    pids = []
    for line in process_table.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        command = fields[1]
        command_tokens = {token.strip("\"'") for token in command.split()}
        if command_tokens.intersection(markers):
            pids.append(int(fields[0]))
    return sorted(set(pids))


def stop_owned_ray_processes(session_dir: Path | None) -> int:
    """Terminate only Ray processes carrying the owned session marker.

    ``ray stop --force`` matches process names globally and is unsafe on a
    shared host. This helper scopes both the initial SIGTERM and the fallback
    SIGKILL to the session directory created by this ROLL invocation.
    """
    if session_dir is None:
        return 0
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    pids = [pid for pid in owned_ray_process_ids(session_dir, result.stdout) if pid != os.getpid()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 5.0
    remaining = set(pids)
    while remaining and time.monotonic() < deadline:
        try:
            table = subprocess.run(
                ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.SubprocessError):
            break
        remaining = set(owned_ray_process_ids(session_dir, table))
        remaining.discard(os.getpid())
        if remaining:
            time.sleep(0.1)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return len(pids)


def resolve_ray_temp_dir() -> Path | None:
    """Return the configured Ray temp directory after a free-space check.

    ROLL keeps the historical Ray default when ``ROLL_RAY_TEMP_DIR`` is unset.
    A configured path is created before Ray starts so the head and workers use
    the same filesystem.  The check is intentionally conservative because Ray
    writes logs, the object store, and temporary checkpoint data there.
    """
    raw = os.environ.get("ROLL_RAY_TEMP_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    minimum = int(os.environ.get("ROLL_RAY_TEMP_MIN_FREE_BYTES", str(_DEFAULT_MIN_FREE_BYTES)))
    if minimum < 0:
        raise ValueError("ROLL_RAY_TEMP_MIN_FREE_BYTES must be non-negative")
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise RuntimeError(
            f"Ray temp directory {path} has only {free} bytes free; "
            f"at least {minimum} bytes are required"
        )
    return path


__all__ = [
    "owned_ray_process_ids",
    "resolve_ray_session_dir",
    "resolve_ray_temp_dir",
    "should_force_new_cluster",
    "stop_owned_ray_processes",
]
