"""Run actual ROLL SFT and verify that its merged checkpoint is complete.

This is an engineering validation runner, not a declaration of model support.
Run from the ROLL checkout in the pinned validation environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


# These names are the public MCA/ROLL checkpoint contract. Keeping them local
# lets --help run before optional Megatron and mcore_adapter packages exist.
MCA_CONFIG_NAME = "mca_config.json"
EXTERNAL_ASSET_METADATA_NAME = "mca_external_assets.json"
ADAPTER_CONFIG_NAME = "adapter_config.json"
DIST_MODEL_DIR = "dist_model"
DIST_OPTIMIZER_DIR = "dist_optimizer"
TRACKER_FILENAME = "latest_checkpointed_iteration.txt"
CHECKPOINT_ITERATION = 1
ITERATION_DIR = f"iter_{CHECKPOINT_ITERATION:07d}"
RNG_STATE_DIR = "rng_state"
SCHEDULER_NAME = "scheduler.pt"
EXTERNAL_ASSET_SCHEMA = "mcore_adapter.qwen4_exp.frozen_ngram_assets"
EXTERNAL_ASSET_SCHEMA_VERSION = 1
MAX_SMALL_STATE_BYTES = 64 * 1024 * 1024
PIPELINE_DEVICE_KEY = "cuda"
NGRAM_HASH_CONSTANTS = {
    "layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets",
}


def select_records(source, destination, limit):
    payload = source.read_bytes()
    rows = [line for line in payload.split(b"\n") if line.strip()]
    if limit is not None:
        if limit <= 0 or limit > len(rows):
            raise ValueError(f"Invalid record limit {limit} for {source}: {len(rows)} available")
        rows = rows[:limit]
    selected = b"\n".join(rows) + b"\n"
    destination.write_bytes(selected)
    return {"source": str(source.resolve()), "source_sha256": hashlib.sha256(payload).hexdigest(),
            "selected_sha256": hashlib.sha256(selected).hexdigest(), "records": len(rows)}


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _positive_size(topology, name):
    value = topology.get(name, 1)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def validate_heldout_batch(records, worker, dp_size):
    """Reject a validation split that drop_last would discard entirely."""
    training = _field(worker, "training_args")
    batch_size = (dp_size * _field(training, "gradient_accumulation_steps", 1)
                  * _field(worker, "infer_batch_size", 1))
    if records < batch_size:
        raise ValueError(
            f"heldout has {records} records but a complete validation batch requires "
            f"{batch_size} (DP={dp_size}); increase --heldout-records or the dataset"
        )


def _legacy_rank_payloads(adapters, tp_size, pp_size, ep_size):
    payloads = []
    for adapter in adapters:
        for tp_rank in range(tp_size):
            for pp_rank in range(pp_size):
                for ep_rank in range(ep_size):
                    rank_dir = f"mp_rank_{tp_rank:02d}"
                    if pp_size > 1:
                        rank_dir += f"_{pp_rank:03d}"
                    if ep_size > 1:
                        rank_dir += f"_{ep_rank:03d}"
                    payloads.append(str(Path(adapter) / ITERATION_DIR / rank_dir / "model_optim_rng.pt"))
    return payloads


def checkpoint_contract(config):
    """Derive the exact checkpoint layout from the resolved SFT worker config."""
    worker = _field(config, "sft_train")
    if worker is None:
        raise ValueError("Resolved config has no sft_train worker")
    strategy_args = _field(worker, "strategy_args")
    if _field(strategy_args, "strategy_name") != "megatron_train":
        raise ValueError("SFT checkpoint validation requires the megatron_train strategy")
    topology_config = _field(strategy_args, "strategy_config", {})
    tp_size = _positive_size(topology_config, "tensor_model_parallel_size")
    pp_size = _positive_size(topology_config, "pipeline_model_parallel_size")
    cp_size = _positive_size(topology_config, "context_parallel_size")
    ep_size = _positive_size(topology_config, "expert_model_parallel_size")
    etp_size = _positive_size(topology_config, "expert_tensor_parallel_size")
    world_size = _field(worker, "world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise ValueError(f"Resolved sft_train.world_size must be positive, got {world_size!r}")
    model_parallel_size = tp_size * pp_size * cp_size
    if world_size % model_parallel_size:
        raise ValueError(
            f"world size {world_size} is not divisible by TP*PP*CP={model_parallel_size}"
        )
    dp_size = world_size // model_parallel_size
    expert_model_parallel_size = etp_size * ep_size * pp_size
    if world_size % expert_model_parallel_size:
        raise ValueError(
            f"world size {world_size} is not divisible by expert ETP*EP*PP={expert_model_parallel_size}"
        )
    expert_dp_size = world_size // expert_model_parallel_size

    model_args = _field(worker, "model_args")
    training_args = _field(worker, "training_args")
    # Match MegatronInferStrategy: strategy_config overrides training_args.
    def training_option(name, default=None):
        return topology_config.get(name, _field(training_args, name, default))

    ckpt_format = training_option("ckpt_format")
    if not training_option("use_distributed_optimizer", False):
        raise ValueError("Exact SFT resume validation requires use_distributed_optimizer=True")
    if training_option("save_only_model", False):
        raise ValueError("Exact SFT resume validation requires save_only_model=False")

    lora_target = _field(model_args, "lora_target")
    adapters = ["default"] if lora_target is not None else []
    if adapters:
        if ckpt_format != "legacy":
            raise ValueError("The Megatron LoRA strategy only saves legacy checkpoints")
        if etp_size != tp_size:
            raise ValueError(
                "Legacy LoRA requires expert tensor parallel size to equal tensor parallel size"
            )
        trainable_scope = {
            "mode": "lora_adapters",
            "adapters": adapters,
            "lora_rank": _field(model_args, "lora_rank"),
            "target_modules": lora_target,
            "additional_modules": _field(model_args, "additional_target") or [],
        }
        layout = "legacy_lora"
    else:
        if ckpt_format != "torch_dist":
            raise ValueError("Backbone validation currently requires ckpt_format=torch_dist")
        frozen = _field(model_args, "freeze_module_prefix")
        trainable_scope = {
            "mode": "partially_frozen_backbone" if frozen else "text_backbone_frozen_ngram",
            "ngram_table": "frozen_external",
            "preserved_auxiliary": ["vision", "mtp"],
        }
        if frozen:
            trainable_scope["frozen_prefixes"] = frozen
        layout = "torch_dist_backbone"

    topology = {
        "world_size": world_size,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "context_parallel_size": cp_size,
        "data_parallel_size": dp_size,
        "expert_parallel_size": ep_size,
        "expert_tensor_parallel_size": etp_size,
        "expert_data_parallel_size": expert_dp_size,
    }
    return {
        "layout": layout,
        "checkpoint_format": ckpt_format,
        "adapters": adapters,
        "adapter_rank_payloads": _legacy_rank_payloads(adapters, tp_size, pp_size, ep_size),
        "topology": topology,
        "trainable_scope": trainable_scope,
    }


def _required_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty checkpoint state file: {path}")
    return path


def _read_json_object(path):
    _required_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint JSON must contain an object: {path}")
    return value


def _load_small_torch_state(path, label, *, allow_numpy=False):
    import torch

    _required_file(path)
    if path.stat().st_size > MAX_SMALL_STATE_BYTES:
        raise ValueError(f"Invalid {label} state {path}: exceeds bounded metadata size")
    try:
        if allow_numpy:
            import numpy as np
            from numpy.core.multiarray import _reconstruct

            safe_numpy = [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
            with torch.serialization.safe_globals(safe_numpy):
                state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise ValueError(f"Invalid {label} state {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Invalid {label} state {path}: expected an object")
    return state


def _require_state_keys(state, required, label):
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Invalid {label} state: missing keys {missing}")


def _validate_torch_rng(value, label):
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"Invalid {label} state: expected a nonempty uint8 tensor")


def _validate_python_rng(value, label):
    import random

    try:
        random.Random().setstate(value)
    except Exception as exc:
        raise ValueError(f"Invalid {label} state: malformed Python RNG state: {exc}") from exc


def _validate_numpy_rng(value, label):
    import numpy as np

    try:
        np.random.RandomState().set_state(value)
    except Exception as exc:
        raise ValueError(f"Invalid {label} state: malformed NumPy RNG state: {exc}") from exc


def _inspect_scheduler(path):
    state = _load_small_torch_state(path, "scheduler")
    required = {
        "max_lr", "min_lr", "lr_warmup_steps", "lr_decay_steps", "lr_decay_style", "num_steps",
        "start_wd", "end_wd", "wd_incr_steps", "wd_incr_style",
    }
    _require_state_keys(state, required, "scheduler")
    steps = state["num_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Invalid scheduler state: num_steps must be a nonnegative integer")
    if not isinstance(state["lr_decay_style"], str) or not isinstance(state["wd_incr_style"], str):
        raise ValueError("Invalid scheduler state: schedule styles must be strings")
    return {"num_steps": steps}


def _inspect_worker_rng(path):
    state = _load_small_torch_state(path, "worker rng", allow_numpy=True)
    required = {
        "random_rng_state", "np_rng_state", "torch_rng_state", "cuda_rng_state", "rng_tracker_states",
    }
    _require_state_keys(state, required, "worker rng")
    _validate_python_rng(state["random_rng_state"], "worker rng")
    _validate_numpy_rng(state["np_rng_state"], "worker rng")
    _validate_torch_rng(state["torch_rng_state"], "worker rng")
    _validate_torch_rng(state["cuda_rng_state"], "worker rng")
    trackers = state["rng_tracker_states"]
    if not isinstance(trackers, dict) or not trackers:
        raise ValueError("Invalid worker rng state: rng_tracker_states must be nonempty")
    for tracker in trackers.values():
        _validate_torch_rng(tracker, "worker rng tracker")
    return {"trackers": len(trackers)}


def _inspect_pipeline_rng(path):
    state = _load_small_torch_state(path, "pipeline rng", allow_numpy=True)
    required = {"python", "numpy", "cpu", PIPELINE_DEVICE_KEY}
    _require_state_keys(state, required, "pipeline rng")
    _validate_python_rng(state["python"], "pipeline rng")
    _validate_numpy_rng(state["numpy"], "pipeline rng")
    _validate_torch_rng(state["cpu"], "pipeline rng")
    values = state[PIPELINE_DEVICE_KEY]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(
            f"Invalid pipeline rng state: {PIPELINE_DEVICE_KEY} RNG states must be nonempty"
        )
    for value in values:
        _validate_torch_rng(value, f"pipeline {PIPELINE_DEVICE_KEY} rng")
    return {"device_keys": [PIPELINE_DEVICE_KEY]}


def _configured_asset_layers(model_config):
    layers = model_config.get("ple_layer_indices")
    if layers is None:
        layer_ids = model_config.get("ple_layer_ids")
        if not isinstance(layer_ids, list):
            raise ValueError("Invalid external asset model config: missing ple_layer_ids")
        if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 1 for layer in layer_ids):
            raise ValueError("Invalid external asset model config: malformed ple_layer_ids")
        layers = [layer - 1 for layer in layer_ids]
    if not isinstance(layers, list) or any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers
    ):
        raise ValueError("Invalid external asset model config: malformed ple_layer_indices")
    return {str(layer) for layer in layers}


def _read_asset_index(source):
    index_path = _required_file(source / "model.safetensors.index.json")
    if index_path.stat().st_size > MAX_SMALL_STATE_BYTES:
        raise ValueError(f"Invalid external asset index {index_path}: exceeds bounded metadata size")
    try:
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid external asset index {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map or any(
        not isinstance(key, str) or not key or not isinstance(filename, str) or not filename
        for key, filename in weight_map.items()
    ):
        raise ValueError(f"Invalid external asset index {index_path}: malformed weight_map")
    return index_bytes, weight_map


def _read_safetensors_header(source, filename, expected):
    file = (source / filename).resolve()
    if not file.is_relative_to(source) or not file.is_file():
        raise ValueError(f"Invalid external asset manifest file: {filename}")
    if (
        not isinstance(expected, dict)
        or not isinstance(expected.get("size"), int)
        or isinstance(expected.get("size"), bool)
        or expected["size"] < 1
        or file.stat().st_size != expected["size"]
    ):
        raise ValueError(f"Invalid external asset manifest file metadata: {filename}")
    try:
        with file.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("truncated header length")
            length = int.from_bytes(raw_length, "little")
            if not 2 <= length <= MAX_SMALL_STATE_BYTES:
                raise ValueError("invalid header length")
            raw_header = handle.read(length)
            if len(raw_header) != length:
                raise ValueError("truncated header")
        header = json.loads(raw_header)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid external asset manifest file {filename}: {exc}") from exc
    expected_hash = expected.get("header_sha256")
    if (
        not isinstance(header, dict)
        or not isinstance(expected_hash, str)
        or hashlib.sha256(raw_header).hexdigest() != expected_hash
    ):
        raise ValueError(f"Invalid external asset manifest header identity: {filename}")
    return header, length + 8, file.stat().st_size


def _validate_asset_entry(entry, *, key, filename, file_size, payload_begin):
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid external asset manifest tensor: {key}")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(entry.get("dtype"), str)
        or not isinstance(shape, list)
        or not shape
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
        or offsets[0] < 0
        or offsets[1] <= offsets[0]
        or payload_begin + offsets[1] > file_size
    ):
        raise ValueError(f"Invalid external asset manifest tensor extent: {key} in {filename}")


def _validate_asset_manifest(layer, manifest, source, weight_map):
    index_hash = manifest.get("index_sha256")
    tensors = manifest.get("tensors")
    files = manifest.get("files")
    constants = manifest.get("hash_constants")
    if not isinstance(index_hash, str) or re.fullmatch(r"[0-9a-f]{64}", index_hash) is None:
        raise ValueError(f"Invalid external asset manifest index identity for layer {layer}")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError(f"Invalid external asset manifest tensors for layer {layer}")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Invalid external asset manifest files for layer {layer}")
    if (
        not isinstance(constants, dict)
        or set(constants) != NGRAM_HASH_CONSTANTS
        or any(not isinstance(value, list) or not value for value in constants.values())
    ):
        raise ValueError(f"Invalid external asset manifest hash constants for layer {layer}")

    resolved_headers = {}
    for filename, identity in files.items():
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Invalid external asset manifest file for layer {layer}")
        resolved_headers[filename] = _read_safetensors_header(source, filename, identity)

    shard_indices = []
    prefix = f"model.language_model.layers.{layer}.ple.ple_embedding."
    shard_pattern = re.compile(re.escape(prefix) + r"ngram_embedding\.shard_(\d+)\.weight")
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise ValueError(f"Invalid external asset manifest tensor for layer {layer}")
        key, filename = tensor.get("key"), tensor.get("file")
        match = shard_pattern.fullmatch(key) if isinstance(key, str) else None
        if match is None or not isinstance(filename, str) or weight_map.get(key) != filename:
            raise ValueError(f"Invalid external asset manifest tensor for layer {layer}")
        if filename not in resolved_headers:
            raise ValueError(f"Invalid external asset manifest file reference: {filename}")
        header, payload_begin, file_size = resolved_headers[filename]
        entry = header.get(key)
        _validate_asset_entry(
            entry, key=key, filename=filename, file_size=file_size, payload_begin=payload_begin
        )
        if any(tensor.get(name) != entry.get(name) for name in ("shape", "dtype", "data_offsets")):
            raise ValueError(f"Invalid external asset manifest tensor identity: {key}")
        shard_indices.append(int(match.group(1)))
    if sorted(shard_indices) != list(range(len(shard_indices))):
        raise ValueError(f"Invalid external asset manifest shard sequence for layer {layer}")

    for name in NGRAM_HASH_CONSTANTS:
        key = prefix + name
        filename = weight_map.get(key)
        if filename not in resolved_headers:
            raise ValueError(f"Invalid external asset manifest hash constant: {key}")
        header, payload_begin, file_size = resolved_headers[filename]
        entry = header.get(key)
        _validate_asset_entry(
            entry, key=key, filename=filename, file_size=file_size, payload_begin=payload_begin
        )
        if entry.get("dtype") != "I64" or len(entry.get("shape", [])) != 1:
            raise ValueError(f"Invalid external asset manifest hash constant: {key}")


def _inspect_external_assets(path, model_config):
    record = _read_json_object(path)
    if (
        record.get("schema") != EXTERNAL_ASSET_SCHEMA
        or record.get("schema_version") != EXTERNAL_ASSET_SCHEMA_VERSION
    ):
        raise ValueError("Invalid external asset sidecar schema")
    source = record.get("source")
    if (
        not isinstance(source, dict)
        or source.get("kind") != "local_checkpoint"
        or not isinstance(source.get("path"), str)
        or not source["path"]
    ):
        raise ValueError("Invalid external asset sidecar source")
    source_path = Path(source["path"]).expanduser()
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    source_path = source_path.resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Qwen4 external asset source is missing: {source_path}")
    index_bytes, weight_map = _read_asset_index(source_path)
    manifests = record.get("manifests")
    if not isinstance(manifests, dict):
        raise ValueError("Invalid external asset sidecar manifests")
    required_fields = {
        "format", "identity_kind", "layer_idx", "index_sha256", "tensors", "files", "hash_constants",
    }
    for layer, manifest in manifests.items():
        if not isinstance(layer, str) or not layer.isdigit() or not isinstance(manifest, dict):
            raise ValueError("Invalid external asset sidecar manifest")
        if not required_fields.issubset(manifest):
            raise ValueError(f"Invalid external asset sidecar manifest: incomplete layer {layer}")
        if (
            manifest["format"] != 1
            or manifest["identity_kind"] != "index_and_header_sha256"
            or manifest["layer_idx"] != int(layer)
        ):
            raise ValueError(f"Invalid external asset sidecar manifest for layer {layer}")
        if manifest["index_sha256"] != hashlib.sha256(index_bytes).hexdigest():
            raise ValueError(f"Invalid external asset index identity for layer {layer}")
        _validate_asset_manifest(layer, manifest, source_path, weight_map)
    expected_layers = _configured_asset_layers(model_config)
    if set(manifests) != expected_layers:
        raise ValueError(
            "Invalid external asset coverage: "
            f"missing={sorted(expected_layers - set(manifests))}, "
            f"unexpected={sorted(set(manifests) - expected_layers)}"
        )
    return {"layers": sorted(expected_layers, key=int)}


def _inspect_distributed_directory(directory):
    from torch.distributed.checkpoint import FileSystemReader
    from scripts.qwen38.checkpoint_integrity import validate_dcp_storage_inventory

    common = _load_small_torch_state(directory / "common.pt", "distributed common")
    native_metadata = _read_json_object(directory / "metadata.json")
    if not native_metadata:
        raise ValueError(f"Invalid distributed checkpoint metadata.json: {directory}")
    if not (directory / ".metadata").is_file():
        raise FileNotFoundError(f"Incomplete distributed checkpoint: {directory}/.metadata")
    try:
        metadata = FileSystemReader(directory).read_metadata()
    except Exception as exc:
        raise ValueError(f"Invalid distributed checkpoint metadata in {directory}: {exc}") from exc
    if not metadata.state_dict_metadata or not metadata.storage_data:
        raise ValueError(f"Distributed checkpoint has no state: {directory}")
    validate_dcp_storage_inventory(metadata)
    files = set()
    for storage in metadata.storage_data.values():
        path = directory / storage.relative_path
        if not path.is_file() or path.stat().st_size < storage.offset + storage.length:
            raise ValueError(f"Missing or truncated distributed checkpoint shard: {path}")
        files.add(path)
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "state_items": len(metadata.state_dict_metadata),
        "common_items": len(common),
    }


def _inspect_legacy_adapter(path, adapter_config, model_config):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    _required_file(path)
    try:
        with FakeTensorMode():
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise ValueError(f"Invalid legacy adapter checkpoint {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Invalid legacy adapter checkpoint {path}: expected an object")
    model_state = state.get("model", state)
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"Invalid legacy adapter checkpoint {path}: missing model state")
    from scripts.qwen38.lora_checkpoint_integrity import validate_native_lora_metadata

    validate_native_lora_metadata(model_state, adapter_config, model_config)
    tensors = [(name, value) for name, value in model_state.items() if isinstance(value, torch.Tensor)]
    if not tensors or not any("lora_" in name for name, _ in tensors):
        raise ValueError(f"Invalid legacy adapter checkpoint {path}: no LoRA tensor metadata")
    return {
        "tensor_metadata": len(tensors),
        "tensor_bytes": sum(value.numel() * value.element_size() for _, value in tensors),
        "bytes": path.stat().st_size,
    }


def _normalized_scope(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted(value)
    return value


def _read_adapter_iteration(adapter_directory):
    tracker = _required_file(adapter_directory / TRACKER_FILENAME)
    try:
        value = tracker.read_text(encoding="utf-8").strip()
        iteration = int(value)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid adapter checkpoint tracker {tracker}") from exc
    if iteration < 1:
        raise ValueError(f"Invalid adapter checkpoint tracker {tracker}: iteration {iteration}")
    if iteration != CHECKPOINT_ITERATION:
        raise ValueError(
            f"Adapter checkpoint tracker {tracker} selects iteration {iteration}, "
            f"expected {CHECKPOINT_ITERATION}"
        )
    return iteration


def inspect_checkpoint(checkpoint, contract):
    checkpoint = Path(checkpoint)
    topology = contract["topology"]
    world_size = topology["world_size"]
    model_config = _read_json_object(checkpoint / MCA_CONFIG_NAME)
    external_assets = _inspect_external_assets(checkpoint / EXTERNAL_ASSET_METADATA_NAME, model_config)
    scheduler = _inspect_scheduler(checkpoint / SCHEDULER_NAME)
    pipeline_state = _read_json_object(checkpoint / "pipeline/worker_state_pipeline.json")
    step_match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if step_match is None:
        raise ValueError(f"Checkpoint directory has no completed step: {checkpoint}")
    expected_step = int(step_match.group(1))
    pipeline_step = pipeline_state.get("step")
    if isinstance(pipeline_step, bool) or not isinstance(pipeline_step, int) or pipeline_step != expected_step:
        raise ValueError(
            f"Pipeline worker state step {pipeline_step!r} does not match completed checkpoint step {expected_step}"
        )
    pipeline_rng = _inspect_pipeline_rng(checkpoint / "pipeline/rng_state_pipeline.pth")
    worker_rng = []
    for rank in range(world_size):
        worker_rng.append(_inspect_worker_rng(checkpoint / RNG_STATE_DIR / f"rng_state_{rank}.pth"))

    iteration = checkpoint / ITERATION_DIR
    result = {
        "layout": contract["layout"],
        "external_assets": external_assets,
        "scheduler": scheduler,
        "pipeline_rng": pipeline_rng,
        "worker_rng_files": len(worker_rng),
        DIST_OPTIMIZER_DIR: _inspect_distributed_directory(iteration / DIST_OPTIMIZER_DIR),
    }
    if contract["layout"] == "torch_dist_backbone":
        result[DIST_MODEL_DIR] = _inspect_distributed_directory(iteration / DIST_MODEL_DIR)
    elif contract["layout"] == "legacy_lora":
        result["adapters"] = {}
        rank_paths = [Path(relative) for relative in contract["adapter_rank_payloads"]]
        for adapter in contract["adapters"]:
            adapter_directory = checkpoint / adapter
            adapter_config = _read_json_object(adapter_directory / ADAPTER_CONFIG_NAME)
            adapter_iteration = _read_adapter_iteration(adapter_directory)
            if str(adapter_config.get("peft_type", "")).upper() != "LORA":
                raise ValueError(f"Adapter {adapter} is not a LoRA adapter")
            expected_rank = contract["trainable_scope"]["lora_rank"]
            if adapter_config.get("r") != expected_rank:
                raise ValueError(
                    f"Adapter {adapter} rank {adapter_config.get('r')!r} does not match {expected_rank!r}"
                )
            for saved_name, scope_name in (
                ("target_modules", "target_modules"),
                ("modules_to_save", "additional_modules"),
            ):
                saved = _normalized_scope(adapter_config.get(saved_name))
                expected = _normalized_scope(contract["trainable_scope"][scope_name])
                if saved != expected:
                    raise ValueError(
                        f"Adapter {adapter} {saved_name} {saved!r} does not match configured scope {expected!r}"
                    )
            adapter_paths = [path for path in rank_paths if path.parts[0] == adapter]
            inspected = [
                _inspect_legacy_adapter(checkpoint / path, adapter_config, model_config)
                for path in adapter_paths
            ]
            result["adapters"][adapter] = {
                "iteration": adapter_iteration,
                "rank_payloads": len(inspected),
                "files": len(inspected),
                "bytes": sum(item["bytes"] for item in inspected),
                "tensor_metadata": sum(item["tensor_metadata"] for item in inspected),
                "tensor_bytes": sum(item["tensor_bytes"] for item in inspected),
            }
    else:
        raise ValueError(f"Unknown checkpoint layout: {contract['layout']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("scripts/qwen38/configs/sft_backbone.yaml"))
    parser.add_argument("--model-path", type=Path, default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-output-dir", type=Path,
                        help="Merged checkpoint destination; defaults to OUTPUT_DIR/checkpoints")
    parser.add_argument("--checkpoint-hard-links", action="store_true",
                        help="Archive immutable checkpoint files using same-filesystem hard links when available")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--heldout-data", type=Path, required=True)
    parser.add_argument("--train-records", type=int)
    parser.add_argument("--heldout-records", type=int)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.model_path is None or not args.model_path.is_dir():
        parser.error("--model-path must identify the local checkpoint")
    if args.max_steps < 2 or args.save_steps < 1 or args.eval_steps < 1:
        parser.error("Validation requires at least two total steps and positive save/eval intervals")

    from dacite import from_dict
    from omegaconf import OmegaConf
    from roll.distributed.scheduler.initialize import init
    from roll.pipeline.sft.sft_config import SFTConfig
    from roll.pipeline.sft.sft_pipeline import SFTPipeline

    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split, source, limit in (("train", args.train_data, args.train_records),
                                 ("heldout", args.heldout_data, args.heldout_records)):
        manifest[split] = select_records(source, work / f"{split}.jsonl", limit)
    (work / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    cfg = OmegaConf.load(args.config)
    cfg.pop("hydra", None)
    cfg.pretrain = str(args.model_path.resolve())
    cfg.exp_name = work.name
    cfg.output_dir = str(work / "local-checkpoints")
    cfg.sequence_length = args.sequence_length
    cfg.max_steps = args.max_steps
    cfg.save_steps = args.save_steps
    cfg.eval_steps = args.eval_steps
    # Keep the midpoint for a cold continuation against the final checkpoint.
    cfg.max_ckpt_to_keep = 2
    cfg.sft_train.data_args.file_name = str(work / "train.jsonl")
    cfg.validation.data_args.file_name = str(work / "heldout.jsonl")
    checkpoint_output = (args.checkpoint_output_dir.resolve() if args.checkpoint_output_dir
                         else work / "checkpoints")
    cfg.checkpoint_config = {"type": "file_system", "output_dir": str(checkpoint_output),
                             "async_upload": False, "use_hard_links": args.checkpoint_hard_links}
    cfg.resume_from_checkpoint = str(args.resume.resolve()) if args.resume else False
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (work / "resolved-config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    config = from_dict(SFTConfig, resolved)
    contract = checkpoint_contract(config)
    validate_heldout_batch(manifest["heldout"]["records"], config.sft_train,
                          contract["topology"]["data_parallel_size"])
    (work / "checkpoint-location.json").write_text(json.dumps(config.checkpoint_config, indent=2) + "\n")
    init()
    pipeline = SFTPipeline(config)
    if not hasattr(pipeline, "val_dataloader") or len(pipeline.val_dataloader) == 0:
        raise ValueError("heldout preprocessing produced no complete validation batch")
    actual_steps = len(pipeline.dataloader) * config.sft_train.training_args.num_train_epochs
    if actual_steps != args.max_steps:
        raise ValueError(f"Dataset and real DP topology yield {actual_steps} steps, expected {args.max_steps}")
    first_step = pipeline.state.step + 1
    pipeline.run()
    if pipeline.state.step != args.max_steps - 1 or first_step >= args.max_steps:
        raise RuntimeError("The requested training updates were not completed")
    checkpoint = Path(config.checkpoint_config["output_dir"]) / f"checkpoint-{pipeline.state.step}"
    summary = {"first_step": first_step, "last_step": pipeline.state.step,
               "updates": pipeline.state.step + 1 - first_step, "checkpoint": str(checkpoint),
               "topology": contract["topology"], "trainable_scope": contract["trainable_scope"],
               "distributed_state": inspect_checkpoint(checkpoint, contract)}
    (work / "completion.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ACTUAL_SFT_VALIDATION_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
