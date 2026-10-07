from __future__ import annotations

import os
from pathlib import Path


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _download_modelscope_snapshot(model_id: str) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "USE_MODELSCOPE is enabled, but the modelscope package is unavailable"
        ) from exc
    return str(snapshot_download(model_id))


def resolve_model_path(
    model_name_or_path: str,
    *,
    use_modelscope: bool | None = None,
    download_type: str | None = None,
) -> str:
    value = str(model_name_or_path).strip()
    if not value:
        raise ValueError("model_name_or_path must not be empty")

    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    if path.is_absolute():
        raise FileNotFoundError(f"model path does not exist: {path}")

    selected_download_type = (
        download_type if download_type is not None else os.environ.get("MODEL_DOWNLOAD_TYPE", "")
    ).strip().upper()
    if selected_download_type == "OPENLM_HUB":
        raise ValueError(
            "OPENLM_HUB is unsupported in this public runtime; use a public "
            "Hugging Face model ID or an existing local model directory"
        )

    if use_modelscope is None:
        use_modelscope = _env_enabled("USE_MODELSCOPE")
    if not use_modelscope:
        return value

    resolved = Path(_download_modelscope_snapshot(value)).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"ModelScope returned a missing model path: {resolved}")
    return str(resolved.resolve())
