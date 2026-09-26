"""Validate an optional filesystem for Ray's session and object-store files."""

from __future__ import annotations

import os
from pathlib import Path
import shutil


_DEFAULT_MIN_FREE_BYTES = 128 * 1024**3


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


__all__ = ["resolve_ray_temp_dir"]
