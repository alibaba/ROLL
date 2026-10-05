import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import save


_RUNNER_PATH = Path(__file__).parents[2] / "scripts/qwen38/run_sft_validation.py"
_SPEC = importlib.util.spec_from_file_location("qwen38_sft_validation", _RUNNER_PATH)
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)

_NGRAM_PATH = Path(__file__).parents[2] / "mcore_adapter/src/mcore_adapter/models/qwen4_exp/ngram_embedding.py"
_NGRAM_SPEC = importlib.util.spec_from_file_location("qwen38_validator_ngram_embedding", _NGRAM_PATH)
_NGRAM = importlib.util.module_from_spec(_NGRAM_SPEC)
_NGRAM_SPEC.loader.exec_module(_NGRAM)


def _config(*, lora=False, tp=1, pp=1, ep=1, etp=1, world_size=1):
    return SimpleNamespace(
        sft_train=SimpleNamespace(
            world_size=world_size,
            model_args=SimpleNamespace(
                lora_target=r".*\.(linear_qkv|linear_fc1)$" if lora else None,
                lora_rank=8,
                additional_target=["output_layer"] if lora else None,
                freeze_module_prefix=None,
            ),
            training_args=SimpleNamespace(
                ckpt_format="legacy" if lora else "torch_dist",
                use_distributed_optimizer=True,
                save_only_model=False,
            ),
            strategy_args=SimpleNamespace(
                strategy_name="megatron_train",
                strategy_config={
                    "tensor_model_parallel_size": tp,
                    "pipeline_model_parallel_size": pp,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": ep,
                    "expert_tensor_parallel_size": etp,
                },
            ),
        )
    )


def _scheduler_state():
    return {
        "max_lr": 1.0e-5,
        "min_lr": 0.0,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 100,
        "lr_decay_style": "constant",
        "num_steps": 7,
        "start_wd": 0.0,
        "end_wd": 0.0,
        "wd_incr_steps": 0,
        "wd_incr_style": "constant",
    }


def _worker_rng_state():
    state = torch.get_rng_state()
    return {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": state,
        "cuda_rng_state": state.clone(),
        "rng_tracker_states": {"model-parallel-rng": state.clone()},
    }


def _pipeline_rng_state():
    state = torch.get_rng_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": state,
        "cuda": [state.clone()],
    }


def _write_asset_source(checkpoint):
    source = checkpoint / "ngram-assets"
    source.mkdir()
    prefix = "model.language_model.layers.1.ple.ple_embedding."
    tensors = {
        prefix + "ngram_embedding.shard_0.weight": torch.arange(24).reshape(12, 2).to(torch.bfloat16),
        prefix + "layer_multipliers": torch.tensor([13, 17, 29]),
        prefix + "ngram_heads_vocab_sizes": torch.tensor([3, 3, 3, 3]),
        prefix + "ngram_heads_offsets": torch.tensor([0, 3, 6, 9]),
    }
    filename = "model.safetensors"
    save_file(tensors, source / filename)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: filename for name in tensors}}), encoding="utf-8"
    )
    manifest = _NGRAM.MMapNGramStore(source, layer_idx=1).manifest
    return source, manifest


def _write_distributed_state(directory, state):
    save(state, checkpoint_id=directory)
    torch.save({}, directory / "common.pt")
    (directory / "metadata.json").write_text(json.dumps({
        "sharded_backend": "torch_dist",
        "sharded_backend_version": 1,
    }))


def _write_common_state(checkpoint, world_size):
    checkpoint.mkdir()
    _, manifest = _write_asset_source(checkpoint)
    (checkpoint / "mca_external_assets.json").write_text(json.dumps({
        "schema": "mcore_adapter.qwen4_exp.frozen_ngram_assets",
        "schema_version": 1,
        "source": {"kind": "local_checkpoint", "path": "ngram-assets"},
        "manifests": {"1": manifest},
    }))
    (checkpoint / "mca_config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "ple_layer_ids": [2],
    }))
    torch.save(_scheduler_state(), checkpoint / "scheduler.pt")
    pipeline = checkpoint / "pipeline"
    pipeline.mkdir()
    (pipeline / "worker_state_pipeline.json").write_text(
        json.dumps({"step": 7, "log_history": [], "kv": {}})
    )
    torch.save(_pipeline_rng_state(), pipeline / "rng_state_pipeline.pth")
    rng = checkpoint / "rng_state"
    rng.mkdir()
    for rank in range(world_size):
        torch.save(_worker_rng_state(), rng / f"rng_state_{rank}.pth")
    _write_distributed_state(
        checkpoint / "iter_0000001/dist_optimizer",
        {"optimizer": {"exp_avg": torch.arange(3)}},
    )


def _write_lora_rank(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {
                "model.layers.0.linear_qkv.lora_A.weight": torch.zeros(8, 16),
                "model.layers.0.linear_qkv.lora_B.weight": torch.zeros(16, 8),
                "model.output_layer.modules_to_save.weight": torch.zeros(4, 16),
            }
        },
        path,
    )


def _write_adapter_config(adapter, *, target=r".*\.(linear_qkv|linear_fc1)$"):
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "r": 8,
        "target_modules": target,
        "modules_to_save": ["output_layer"],
    }))
    (adapter / "latest_checkpointed_iteration.txt").write_text("1")


def _write_single_rank_lora_checkpoint(tmp_path):
    contract = _RUNNER.checkpoint_contract(_config(lora=True))
    checkpoint = tmp_path / "checkpoint-7"
    _write_common_state(checkpoint, world_size=1)
    adapter = checkpoint / "default"
    adapter.mkdir()
    _write_adapter_config(adapter)
    _write_lora_rank(checkpoint / contract["adapter_rank_payloads"][0])
    return checkpoint, contract, adapter


def test_backbone_contract_requires_distributed_model_instead_of_inferring_lora(tmp_path):
    contract = _RUNNER.checkpoint_contract(_config(world_size=2, tp=2, etp=2))
    checkpoint = tmp_path / "checkpoint-7"
    _write_common_state(checkpoint, world_size=2)

    with pytest.raises(FileNotFoundError, match="dist_model"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)

    _write_distributed_state(
        checkpoint / "iter_0000001/dist_model",
        {"model": {"weight": torch.arange(4)}},
    )
    result = _RUNNER.inspect_checkpoint(checkpoint, contract)
    assert result["layout"] == "torch_dist_backbone"
    assert result["dist_model"]["state_items"] == 1


def test_lora_contract_validates_every_tp_pp_ep_rank_and_reports_trainable_scope(tmp_path):
    contract = _RUNNER.checkpoint_contract(_config(lora=True, tp=2, pp=2, ep=2, etp=2, world_size=8))
    assert contract["trainable_scope"] == {
        "mode": "lora_adapters",
        "adapters": ["default"],
        "lora_rank": 8,
        "target_modules": r".*\.(linear_qkv|linear_fc1)$",
        "additional_modules": ["output_layer"],
    }
    checkpoint = tmp_path / "checkpoint-7"
    _write_common_state(checkpoint, world_size=8)
    adapter = checkpoint / "default"
    adapter.mkdir()
    _write_adapter_config(adapter)
    for relative_path in contract["adapter_rank_payloads"]:
        _write_lora_rank(checkpoint / relative_path)

    result = _RUNNER.inspect_checkpoint(checkpoint, contract)

    assert result["layout"] == "legacy_lora"
    assert result["adapters"]["default"]["iteration"] == 1
    assert result["adapters"]["default"]["rank_payloads"] == 8
    assert result["adapters"]["default"]["tensor_metadata"] == 24
    assert result["dist_optimizer"]["state_items"] == 1


def test_lora_contract_rejects_missing_rank_payload(tmp_path):
    contract = _RUNNER.checkpoint_contract(_config(lora=True, tp=1, pp=1, ep=2, world_size=2))
    checkpoint = tmp_path / "checkpoint-7"
    _write_common_state(checkpoint, world_size=2)
    adapter = checkpoint / "default"
    adapter.mkdir()
    _write_adapter_config(adapter)
    _write_lora_rank(checkpoint / contract["adapter_rank_payloads"][0])

    with pytest.raises(FileNotFoundError, match="mp_rank_00_001/model_optim_rng.pt"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_lora_contract_rejects_truncated_rank_payload(tmp_path):
    contract = _RUNNER.checkpoint_contract(_config(lora=True))
    checkpoint = tmp_path / "checkpoint-7"
    _write_common_state(checkpoint, world_size=1)
    adapter = checkpoint / "default"
    adapter.mkdir()
    _write_adapter_config(adapter)
    payload = checkpoint / contract["adapter_rank_payloads"][0]
    _write_lora_rank(payload)
    payload.write_bytes(payload.read_bytes()[:64])

    with pytest.raises(ValueError, match="Invalid legacy adapter checkpoint"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_lora_contract_rejects_layout_the_strategy_cannot_save():
    config = _config(lora=True, tp=2, etp=1, world_size=2)

    with pytest.raises(ValueError, match="expert tensor parallel size"):
        _RUNNER.checkpoint_contract(config)


def test_lora_contract_rejects_adapter_config_for_a_different_trainable_scope(tmp_path):
    contract = _RUNNER.checkpoint_contract(_config(lora=True))
    checkpoint = tmp_path / "checkpoint-7"
    _write_common_state(checkpoint, world_size=1)
    adapter = checkpoint / "default"
    adapter.mkdir()
    _write_adapter_config(adapter, target="linear_proj")
    _write_lora_rank(checkpoint / contract["adapter_rank_payloads"][0])

    with pytest.raises(ValueError, match="target_modules"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_lora_contract_rejects_missing_iteration_tracker(tmp_path):
    checkpoint, contract, adapter = _write_single_rank_lora_checkpoint(tmp_path)
    (adapter / "latest_checkpointed_iteration.txt").unlink()

    with pytest.raises(FileNotFoundError, match="latest_checkpointed_iteration.txt"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_lora_contract_rejects_malformed_iteration_tracker(tmp_path):
    checkpoint, contract, adapter = _write_single_rank_lora_checkpoint(tmp_path)
    (adapter / "latest_checkpointed_iteration.txt").write_text("not-an-iteration")

    with pytest.raises(ValueError, match="Invalid.*latest_checkpointed_iteration.txt"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_lora_contract_rejects_tracker_for_another_iteration(tmp_path):
    checkpoint, contract, adapter = _write_single_rank_lora_checkpoint(tmp_path)
    (adapter / "latest_checkpointed_iteration.txt").write_text("2")

    with pytest.raises(ValueError, match="selects iteration 2.*expected 1"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


@pytest.mark.parametrize("corruption", ["schema", "source", "manifest", "coverage"])
def test_checkpoint_rejects_invalid_external_asset_contract(tmp_path, corruption):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    sidecar_path = checkpoint / "mca_external_assets.json"
    sidecar = json.loads(sidecar_path.read_text())
    if corruption == "schema":
        sidecar["schema_version"] = 2
    elif corruption == "source":
        sidecar["source"]["path"] = ""
    elif corruption == "manifest":
        del sidecar["manifests"]["1"]["files"]
    else:
        sidecar["manifests"] = {}
    sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="external asset"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_checkpoint_rejects_missing_external_asset_source(tmp_path):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    sidecar_path = checkpoint / "mca_external_assets.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["source"]["path"] = "missing-assets"
    sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(FileNotFoundError, match="external asset"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_checkpoint_rejects_missing_external_asset_index(tmp_path):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    (checkpoint / "ngram-assets/model.safetensors.index.json").unlink()

    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_checkpoint_rejects_external_asset_index_identity_mismatch(tmp_path):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    index = checkpoint / "ngram-assets/model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {}}))

    with pytest.raises(ValueError, match="external asset.*index"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("tensors", []),
        ("tensors", {}),
        ("files", {}),
        ("files", []),
        ("hash_constants", {}),
        ("hash_constants", []),
    ],
)
def test_checkpoint_rejects_empty_or_wrongly_typed_asset_manifest_collection(
    tmp_path, field, replacement
):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    sidecar_path = checkpoint / "mca_external_assets.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["manifests"]["1"][field] = replacement
    sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="external asset.*manifest"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


@pytest.mark.parametrize("step", [True, 6])
def test_checkpoint_rejects_invalid_or_mismatched_pipeline_step(tmp_path, step):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    pipeline_state = checkpoint / "pipeline/worker_state_pipeline.json"
    pipeline_state.write_text(json.dumps({"step": step, "log_history": [], "kv": {}}))

    with pytest.raises(ValueError, match="Pipeline worker state"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


@pytest.mark.parametrize("artifact", ["scheduler", "worker_rng", "pipeline_rng"])
@pytest.mark.parametrize("corruption", ["truncated", "missing_key"])
def test_checkpoint_rejects_corrupt_native_training_state(tmp_path, artifact, corruption):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    if artifact == "scheduler":
        path = checkpoint / "scheduler.pt"
        state = _scheduler_state()
        missing_key = "num_steps"
    elif artifact == "worker_rng":
        path = checkpoint / "rng_state/rng_state_0.pth"
        state = _worker_rng_state()
        missing_key = "rng_tracker_states"
    else:
        path = checkpoint / "pipeline/rng_state_pipeline.pth"
        state = _pipeline_rng_state()
        missing_key = "cuda"
    if corruption == "truncated":
        path.write_bytes(b"x")
    else:
        del state[missing_key]
        torch.save(state, path)

    with pytest.raises(ValueError, match=artifact.replace("_", " ")):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


@pytest.mark.parametrize(
    ("artifact", "key", "value"),
    [
        ("worker", "random_rng_state", (3, (), None)),
        ("pipeline", "numpy", ("MT19937", np.array([1], dtype=np.uint32), 0, 0, 0.0)),
    ],
)
def test_checkpoint_rejects_rng_inner_state_rejected_by_native_setter(tmp_path, artifact, key, value):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    if artifact == "worker":
        path = checkpoint / "rng_state/rng_state_0.pth"
        state = _worker_rng_state()
    else:
        path = checkpoint / "pipeline/rng_state_pipeline.pth"
        state = _pipeline_rng_state()
    state[key] = value
    torch.save(state, path)

    with pytest.raises(ValueError, match=f"{artifact} rng"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_checkpoint_rejects_pipeline_rng_without_cuda_device_key(tmp_path):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    path = checkpoint / "pipeline/rng_state_pipeline.pth"
    state = _pipeline_rng_state()
    state["bogus"] = state.pop("cuda")
    torch.save(state, path)

    with pytest.raises(ValueError, match="pipeline rng.*cuda"):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


@pytest.mark.parametrize("sidecar", ["common.pt", "metadata.json"])
@pytest.mark.parametrize("corruption", ["missing", "malformed"])
def test_checkpoint_rejects_invalid_distributed_checkpoint_sidecars(tmp_path, sidecar, corruption):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    path = checkpoint / "iter_0000001/dist_optimizer" / sidecar
    if corruption == "missing":
        path.unlink()
    elif sidecar == "common.pt":
        path.write_bytes(b"x")
    else:
        path.write_text("not-json")

    with pytest.raises((FileNotFoundError, ValueError), match=sidecar):
        _RUNNER.inspect_checkpoint(checkpoint, contract)


def test_actual_backbone_yaml_contract_uses_worker_strategy_overrides():
    import yaml

    config = yaml.safe_load((_RUNNER_PATH.parent / "configs/sft_backbone.yaml").read_text())
    config["sft_train"]["world_size"] = 8
    contract = _RUNNER.checkpoint_contract(config)
    assert contract["layout"] == "torch_dist_backbone"
    assert contract["topology"]["data_parallel_size"] == 2  # Eight GPUs with TP4, PP1, CP1.
    assert contract["topology"]["expert_data_parallel_size"] == 1
    assert contract["trainable_scope"]["mode"] == "text_backbone_frozen_ngram"
    assert contract["trainable_scope"]["ngram_table"] == "frozen_external"


def test_actual_lora_yaml_contract_uses_worker_strategy_overrides():
    import yaml

    config = yaml.safe_load((_RUNNER_PATH.parent / "configs/sft_lora.yaml").read_text())
    config["sft_train"]["world_size"] = 8
    config["sft_train"]["training_args"].update(
        ckpt_format="torch_dist", use_distributed_optimizer=False, save_only_model=True)
    contract = _RUNNER.checkpoint_contract(config)
    assert contract["layout"] == "legacy_lora"
    assert contract["topology"]["data_parallel_size"] == 8
    assert contract["topology"]["expert_data_parallel_size"] == 1
    assert len(contract["adapter_rank_payloads"]) == 8


def test_expert_topology_must_divide_physical_world():
    with pytest.raises(ValueError, match="expert"):
        _RUNNER.checkpoint_contract(_config(world_size=8, tp=2, ep=8, etp=2))


@pytest.mark.parametrize('fault', ['scalar', 'wrong_rank', 'missing_b', 'mixed_dtype'])
def test_lora_metadata_rejects_unusable_matrix_pair(tmp_path, fault):
    checkpoint, contract, _ = _write_single_rank_lora_checkpoint(tmp_path)
    path = checkpoint / contract['adapter_rank_payloads'][0]
    state = torch.load(path, weights_only=True)
    a = 'model.layers.0.linear_qkv.lora_A.weight'
    b = 'model.layers.0.linear_qkv.lora_B.weight'
    if fault == 'scalar':
        state['model'][a] = torch.tensor(1.)
    elif fault == 'wrong_rank':
        state['model'][a] = torch.zeros(4, 16)
        state['model'][b] = torch.zeros(16, 4)
    elif fault == 'missing_b':
        del state['model'][b]
    else:
        state['model'][b] = state['model'][b].to(torch.bfloat16)
    torch.save(state, path)
    with pytest.raises(ValueError, match='LoRA'):
        _RUNNER.inspect_checkpoint(checkpoint, contract)
