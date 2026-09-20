"""Versioned sidecar lifecycle for Qwen4 frozen n-gram assets.

The sidecar intentionally lives beside MCA checkpoints rather than in public
model configuration. It records where the external table was attached and the
metadata identity produced by :class:`MMapNGramStore`; the large table remains
outside the model state dictionary.

The recorded identity covers the safetensors index, headers, geometry, file
sizes, and hash constants. It does not authenticate every payload byte, so the
referenced files must be treated as immutable checkpoint assets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


EXTERNAL_ASSET_METADATA_NAME = "mca_external_assets.json"
_SCHEMA = "mcore_adapter.qwen4_exp.frozen_ngram_assets"
_SCHEMA_VERSION = 1
_MODEL_RECORD_ATTRIBUTE = "_qwen4_ngram_asset_record"


def _validate_manifests(manifests: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(manifests, dict):
        raise ValueError("external asset sidecar manifests must be an object")
    validated = {}
    for layer, manifest in manifests.items():
        if not isinstance(layer, str) or not layer.isdigit() or not isinstance(manifest, dict):
            raise ValueError("external asset sidecar has an invalid layer manifest")
        required = {
            "format",
            "identity_kind",
            "layer_idx",
            "index_sha256",
            "tensors",
            "files",
            "hash_constants",
        }
        if not required.issubset(manifest):
            raise ValueError(f"external asset sidecar layer {layer} has an incomplete manifest")
        if manifest["format"] != 1 or manifest["identity_kind"] != "index_and_header_sha256":
            raise ValueError(f"external asset sidecar layer {layer} uses an unsupported manifest format")
        if manifest["layer_idx"] != int(layer):
            raise ValueError(f"external asset sidecar layer key {layer} disagrees with its manifest")
        validated[layer] = manifest
    return validated


def _configured_layers(model) -> set[str] | None:
    layers = getattr(getattr(model, "config", None), "ple_layer_indices", None)
    return {str(layer) for layer in layers} if layers is not None else None


def _validate_coverage(manifests, required):
    if required is not None and set(manifests) != required:
        raise ValueError(
            "n-gram external asset coverage mismatch: "
            f"missing={sorted(required - set(manifests))}, unexpected={sorted(set(manifests) - required)}"
        )


def _asset_owner(model):
    # PEFT forwards the bound hook to its base model. Keep the record there so
    # both wrapper and base-model checkpoint hooks observe the same identity.
    hook = model.attach_ngram_assets
    return getattr(hook, "__self__", model)


def _gather_results(result, error):
    """Reach the same validation outcome on every rank before further I/O."""
    import torch.distributed as dist

    payload = {"result": result, "error": None if error is None else f"{type(error).__name__}: {error}"}
    gathered = [payload]
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)
    errors = [f"rank {rank}: {item['error']}" for rank, item in enumerate(gathered) if item["error"]]
    if errors:
        if error is not None:
            raise error
        raise ValueError("n-gram external asset validation failed: " + "; ".join(errors))
    return [item["result"] for item in gathered]


def _merge_records(groups):
    records = [record for group in groups for record in group]
    if not records:
        raise RuntimeError("cannot save Qwen4 checkpoint before frozen n-gram assets are attached")
    merged = dict(records[0], manifests={})
    for record in records:
        record = _validate_record(record)
        for layer, manifest in record["manifests"].items():
            if layer in merged["manifests"] and merged["manifests"][layer] != manifest:
                raise ValueError(f"n-gram external asset manifest mismatch across stages for layer {layer}")
            merged["manifests"][layer] = manifest
    return merged


def _validate_global_coverage(record, requirements):
    for group in requirements:
        for required in group:
            _validate_coverage(record["manifests"], required)


def _validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("external asset sidecar must contain a JSON object")
    if record.get("schema") != _SCHEMA or record.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported external asset sidecar schema")
    source = record.get("source")
    if (
        not isinstance(source, dict)
        or source.get("kind") != "local_checkpoint"
        or not isinstance(source.get("path"), str)
        or not source["path"]
    ):
        raise ValueError("external asset sidecar has an invalid source")
    manifests = _validate_manifests(record.get("manifests"))
    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "source": {"kind": "local_checkpoint", "path": source["path"]},
        "manifests": manifests,
    }


def _read_record(checkpoint: Path) -> dict[str, Any] | None:
    sidecar = checkpoint / EXTERNAL_ASSET_METADATA_NAME
    if not sidecar.is_file():
        return None
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read external asset sidecar {sidecar}: {exc}") from exc
    return _validate_record(record)


def _resolved_source(
    checkpoint: Path,
    record: Mapping[str, Any] | None,
    external_asset_path: str | os.PathLike[str] | None,
) -> Path:
    if external_asset_path is not None:
        source = Path(external_asset_path).expanduser()
    elif record is None:
        source = checkpoint
    else:
        source = Path(record["source"]["path"]).expanduser()
        if not source.is_absolute():
            source = checkpoint / source
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Qwen4 frozen n-gram asset checkpoint is missing: {source}")
    return source


def restore_ngram_assets(
    model,
    model_name_or_path: str | os.PathLike[str],
    *,
    external_asset_path: str | os.PathLike[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Attach and validate assets for a direct HF load or MCA resume."""
    checkpoint = Path(model_name_or_path).expanduser().resolve()
    record = _read_record(checkpoint)
    source = _resolved_source(checkpoint, record, external_asset_path)
    expected = record["manifests"] if record is not None else None
    if expected is not None:
        _validate_coverage(expected, _configured_layers(model))
    loaded = model.attach_ngram_assets(source, expected)
    loaded = _validate_manifests(loaded)
    if expected is not None and any(expected.get(layer) != manifest for layer, manifest in loaded.items()):
        raise ValueError("n-gram external asset manifest mismatch")
    setattr(
        model,
        _MODEL_RECORD_ATTRIBUTE,
        {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "source": {"kind": "local_checkpoint", "path": str(source)},
            "manifests": loaded,
        },
    )
    return loaded


def restore_ngram_asset_models(models, model_name_or_path, *, external_asset_path=None):
    """Restore each local chunk and validate the union across all PP/TP/DP ranks.

    All ranks must enter this boundary, including stages without PLE layers.
    Only small identity metadata is communicated; external tables stay mapped
    by their owning local layers.
    """
    models = [_asset_owner(model) for model in models]
    loaded, records, requirements, error = [], [], [], None
    try:
        for model in models:
            loaded.append(restore_ngram_assets(model, model_name_or_path, external_asset_path=external_asset_path))
            records.append(getattr(model, _MODEL_RECORD_ATTRIBUTE))
            requirements.append(_configured_layers(model))
    except Exception as exc:
        error = exc
    gathered = _gather_results({"records": records, "requirements": requirements}, error)
    record = _merge_records([item["records"] for item in gathered])
    _validate_global_coverage(record, [item["requirements"] for item in gathered])
    return loaded


def persist_ngram_assets(model, save_directory: str | os.PathLike[str]) -> Path:
    """Persist the attached asset identity beside an MCA or adapter checkpoint."""
    return persist_ngram_asset_models([model], save_directory)


def persist_ngram_asset_models(models, save_directory: str | os.PathLike[str]) -> Path:
    """Collect all stage identities and atomically publish one complete sidecar."""
    import torch.distributed as dist

    records, requirements, error = [], [], None
    try:
        for model in models:
            model = _asset_owner(model)
            record = getattr(model, _MODEL_RECORD_ATTRIBUTE, None)
            if record is None:
                raise RuntimeError("cannot save Qwen4 checkpoint before frozen n-gram assets are attached")
            records.append(_validate_record(record))
            requirements.append(_configured_layers(model))
    except Exception as exc:
        error = exc
    gathered = _gather_results({"records": records, "requirements": requirements}, error)
    record = _merge_records([item["records"] for item in gathered])
    _validate_global_coverage(record, [item["requirements"] for item in gathered])
    destination = Path(save_directory) / EXTERNAL_ASSET_METADATA_NAME
    error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            _write_record(record, Path(save_directory))
        except Exception as exc:
            error = exc
    _gather_results(None, error)
    return destination


def _write_record(record, directory):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / EXTERNAL_ASSET_METADATA_NAME
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=f".{EXTERNAL_ASSET_METADATA_NAME}.", delete=False
    ) as temporary:
        json.dump(record, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination)
    return destination


__all__ = [
    "EXTERNAL_ASSET_METADATA_NAME",
    "persist_ngram_assets",
    "persist_ngram_asset_models",
    "restore_ngram_assets",
    "restore_ngram_asset_models",
]
