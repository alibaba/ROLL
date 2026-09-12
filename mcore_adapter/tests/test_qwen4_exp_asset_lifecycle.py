"""External frozen n-gram asset lifecycle regressions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

import pytest
import torch
from safetensors.torch import save_file


QWEN4_ROOT = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp"
TEST_PACKAGE = types.ModuleType("_qwen4_asset_lifecycle_test")
TEST_PACKAGE.__path__ = [str(QWEN4_ROOT)]
sys.modules[TEST_PACKAGE.__name__] = TEST_PACKAGE


def load_qwen4_module(name):
    spec = importlib.util.spec_from_file_location(
        f"{TEST_PACKAGE.__name__}.{name}", QWEN4_ROOT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as exc:
        pytest.fail(f"missing Qwen4 asset lifecycle module: {exc}")
    return module


ngram_embedding = load_qwen4_module("ngram_embedding")


def checkpoint_fixture(path: Path, *, constants_delta: int = 0):
    path.mkdir()
    prefix = "model.language_model.layers.1.ple.ple_embedding."
    tensors = {
        prefix + "ngram_embedding.shard_0.weight": torch.arange(24).reshape(12, 2).to(torch.bfloat16),
        prefix + "layer_multipliers": torch.tensor([13, 17, 29 + constants_delta]),
        prefix + "ngram_heads_vocab_sizes": torch.tensor([3, 3, 3, 3]),
        prefix + "ngram_heads_offsets": torch.tensor([0, 3, 6, 9]),
    }
    filename = "model.safetensors"
    save_file(tensors, path / filename)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: filename for name in tensors}}), encoding="utf-8"
    )


class TinyAssetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ngram_embedding.FrozenNGramEmbedding(
            embedding_dim=8,
            ngram_size=3,
            heads_per_ngram=2,
            vocab_size=3,
            eos_token_id=0,
        )

    def attach_ngram_assets(self, checkpoint, manifests=None):
        return {
            "1": self.embedding.attach_checkpoint(
                checkpoint,
                layer_idx=1,
                expected_manifest=(manifests or {}).get("1"),
            )
        }

    def load_external_assets(self, model_name_or_path, external_asset_path=None):
        lifecycle = load_qwen4_module("asset_lifecycle")
        return lifecycle.restore_ngram_assets(
            self,
            model_name_or_path,
            external_asset_path=external_asset_path,
        )

    def save_external_assets(self, save_directory):
        lifecycle = load_qwen4_module("asset_lifecycle")
        return lifecycle.persist_ngram_assets(self, save_directory)


def test_sidecar_save_and_resume_bind_real_external_table(tmp_path):
    lifecycle = load_qwen4_module("asset_lifecycle")
    source = tmp_path / "source"
    checkpoint_fixture(source)
    model = TinyAssetModel()

    manifests = lifecycle.restore_ngram_assets(model, source)
    saved = tmp_path / "saved"
    saved.mkdir()
    sidecar = lifecycle.persist_ngram_assets(model, saved)

    assert model.embedding.store is not None
    assert manifests["1"]["identity_kind"] == "index_and_header_sha256"
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["schema"] == "mcore_adapter.qwen4_exp.frozen_ngram_assets"
    assert record["schema_version"] == 1
    assert record["source"]["kind"] == "local_checkpoint"
    assert record["source"]["path"] == str(source.resolve())
    assert record["manifests"] == manifests

    resumed = TinyAssetModel()
    restored = lifecycle.restore_ngram_assets(resumed, saved)
    assert resumed.embedding.store is not None
    assert restored == manifests
    ids = torch.tensor([[0, 3, 11]])
    torch.testing.assert_close(resumed.embedding.store.lookup(ids), model.embedding.store.lookup(ids))


def test_explicit_relocation_accepts_same_identity_and_rejects_changed_assets(tmp_path):
    lifecycle = load_qwen4_module("asset_lifecycle")
    source = tmp_path / "source"
    checkpoint_fixture(source)
    model = TinyAssetModel()
    lifecycle.restore_ngram_assets(model, source)
    saved = tmp_path / "saved"
    saved.mkdir()
    lifecycle.persist_ngram_assets(model, saved)

    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    shutil.rmtree(source)
    with pytest.raises(FileNotFoundError):
        lifecycle.restore_ngram_assets(TinyAssetModel(), saved)

    resumed = TinyAssetModel()
    lifecycle.restore_ngram_assets(resumed, saved, external_asset_path=relocated)
    assert resumed.embedding.store.checkpoint == relocated.resolve()

    changed = tmp_path / "changed"
    checkpoint_fixture(changed, constants_delta=1)
    with pytest.raises(ValueError, match="manifest mismatch"):
        lifecycle.restore_ngram_assets(TinyAssetModel(), saved, external_asset_path=changed)


def test_rejects_unknown_or_incomplete_sidecar_before_attachment(tmp_path):
    lifecycle = load_qwen4_module("asset_lifecycle")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    sidecar = checkpoint / lifecycle.EXTERNAL_ASSET_METADATA_NAME
    sidecar.write_text(json.dumps({"schema": "unknown", "schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        lifecycle.restore_ngram_assets(TinyAssetModel(), checkpoint)


@pytest.mark.skipif(importlib.util.find_spec("megatron") is None, reason="requires mcore_adapter runtime")
def test_factory_from_pretrained_and_save_pretrained_run_asset_hooks(tmp_path, monkeypatch):
    from mcore_adapter.models import model_factory
    from mcore_adapter.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

    lifecycle = load_qwen4_module("asset_lifecycle")
    source = tmp_path / "hf-source"
    checkpoint_fixture(source)

    class TinyConfig:
        virtual_pipeline_model_parallel_size = None
        padded_vocab_size = 32
        tie_embeddings_and_output_weights = False

        @classmethod
        def from_pretrained(cls, path, args=None):
            return cls()

        def distribute_config_match(self, other):
            return True

        def save_pretrained(self, save_directory):
            Path(save_directory, "mca_config.json").write_text("{}", encoding="utf-8")

    class FactoryAssetModel(TinyAssetModel):
        config_class = TinyConfig
        from_pretrained = classmethod(model_factory.PretrainedModel.from_pretrained.__func__)
        save_pretrained = model_factory.PretrainedModel.save_pretrained
        attach_ngram_assets = Qwen4ExpModel.attach_ngram_assets
        load_external_assets = Qwen4ExpModel.load_external_assets
        save_external_assets = Qwen4ExpModel.save_external_assets

        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config
            ple = types.SimpleNamespace(ple_embedding=self.embedding)
            self.decoder = types.SimpleNamespace(
                layers=[types.SimpleNamespace(layer_number=2, ple=ple)]
            )

    class EmptyConverter:
        def __init__(self, config, resized_vocab_size=None):
            pass

        def load_mca_state_dict_from_hf(self, model_name_or_path, vp_stage=0):
            return {
                "embedding.layer_multipliers": torch.ones(3, dtype=torch.long),
                "embedding.ngram_heads_vocab_sizes": torch.full((4,), 3, dtype=torch.long),
                "embedding.ngram_heads_offsets": torch.zeros(4, dtype=torch.long),
            }

    empty_state = {"model": EmptyConverter(None).load_mca_state_dict_from_hf(source)}

    def save_tiny_checkpoint(save_directory, config, state_dict, ckpt_format="legacy"):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        config.save_pretrained(save_directory)

    monkeypatch.setattr(model_factory, "ModelConverter", EmptyConverter)
    monkeypatch.setattr(model_factory, "save_config_and_state_dict", save_tiny_checkpoint)
    monkeypatch.setattr(model_factory, "exists_hf_config", lambda path: True)
    monkeypatch.setattr(model_factory, "exists_mca_config", lambda path: Path(path).name == "mca-checkpoint")
    monkeypatch.setattr(model_factory, "find_dist_ckpt", lambda path: False)
    monkeypatch.setattr(model_factory, "load_state_dict_from_checkpoint", lambda path, models=None: empty_state)
    monkeypatch.setattr(model_factory.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_factory.mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_factory.mpu, "get_expert_model_parallel_rank", lambda: 0)

    loaded = FactoryAssetModel.from_pretrained(str(source), use_cpu_initialization=True)
    assert loaded[0].embedding.store is not None

    saved = tmp_path / "mca-checkpoint"
    loaded[0].save_pretrained(saved, state_dict={"model": {}})
    assert (saved / lifecycle.EXTERNAL_ASSET_METADATA_NAME).is_file()
    assert "source" not in json.loads((saved / "mca_config.json").read_text(encoding="utf-8"))

    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    shutil.rmtree(source)
    resumed = FactoryAssetModel.from_pretrained(
        str(saved),
        use_cpu_initialization=True,
        external_asset_path=str(relocated),
    )
    assert resumed[0].embedding.store is not None
    assert resumed[0].embedding.store.manifest == loaded[0].embedding.store.manifest
    assert resumed[0].embedding.store.checkpoint == relocated.resolve()


@pytest.mark.skipif(importlib.util.find_spec("megatron") is None, reason="requires mcore_adapter runtime")
def test_peft_adapter_save_preserves_external_asset_sidecar(tmp_path, monkeypatch):
    from mcore_adapter.models import model_factory

    lifecycle = load_qwen4_module("asset_lifecycle")
    source = tmp_path / "hf-source"
    checkpoint_fixture(source)

    class TinyConfig:
        def save_pretrained(self, save_directory):
            Path(save_directory).mkdir(parents=True, exist_ok=True)
            Path(save_directory, "mca_config.json").write_text("{}", encoding="utf-8")

    class AdapterConfig:
        def save_pretrained(self, save_directory):
            Path(save_directory).mkdir(parents=True, exist_ok=True)
            Path(save_directory, "adapter_config.json").write_text("{}", encoding="utf-8")

    class BaseAssetModel(TinyAssetModel):
        save_pretrained = model_factory.PretrainedModel.save_pretrained

        def __init__(self):
            super().__init__()
            self.config = TinyConfig()

    def save_tiny_checkpoint(save_directory, config, state_dict, ckpt_format="legacy"):
        config.save_pretrained(save_directory)

    class PeftWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.base_model = types.SimpleNamespace(model=model)
            self.peft_config = {"default": AdapterConfig()}

        def state_dict_for_save_checkpoint(self):
            return {}

    base_model = BaseAssetModel()
    lifecycle.restore_ngram_assets(base_model, source)
    wrapped = PeftWrapper(base_model)
    virtual = object.__new__(model_factory.VirtualModels)
    virtual.models = [wrapped]
    virtual.config = TinyConfig()

    monkeypatch.setattr(model_factory, "PeftModel", PeftWrapper)
    monkeypatch.setattr(model_factory, "is_peft_available", lambda: True)
    monkeypatch.setattr(model_factory, "get_peft_model_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(model_factory, "save_config_and_state_dict", save_tiny_checkpoint)

    saved = tmp_path / "adapter-checkpoint"
    virtual.save_pretrained(saved)
    adapter_sidecar = saved / "default" / lifecycle.EXTERNAL_ASSET_METADATA_NAME
    assert adapter_sidecar.is_file()
    record = json.loads(adapter_sidecar.read_text(encoding="utf-8"))
    assert record["manifests"]["1"] == base_model.embedding.store.manifest


@pytest.mark.skipif(importlib.util.find_spec("megatron") is None, reason="requires mcore_adapter runtime")
def test_hf_export_preserves_external_asset_sidecar(tmp_path, monkeypatch):
    from mcore_adapter.models import model_factory

    lifecycle = load_qwen4_module("asset_lifecycle")
    source = tmp_path / "hf-source"
    checkpoint_fixture(source)

    class TinyConfig:
        moe_parallel_folding = False

    class ExportAssetModel(TinyAssetModel):
        def __init__(self):
            super().__init__()
            lifecycle.restore_ngram_assets(self, source)

    class EmptyConverter:
        def __init__(self, config, to_hf=False):
            pass

        def save_model_as_hf_inflight(self, *args, **kwargs):
            Path(kwargs.get("save_directory", args[1] if len(args) > 1 else tmp_path / "unused")).mkdir(
                parents=True, exist_ok=True
            )

    virtual = object.__new__(model_factory.VirtualModels)
    virtual.models = [ExportAssetModel()]
    virtual.config = TinyConfig()
    monkeypatch.setattr(model_factory, "ModelConverter", EmptyConverter)

    saved = tmp_path / "hf-export"
    virtual.save_pretrained_as_hf(saved)
    sidecar = saved / lifecycle.EXTERNAL_ASSET_METADATA_NAME
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["manifests"]["1"]
