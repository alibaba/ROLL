"""The real worker must avoid split-K atomics for Flash-Next LoRA only."""

import ast
import importlib.util
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).parents[3]
SHRINK_MODULE = "vllm.lora.ops.triton_ops.lora_shrink_op"


def initialize_worker(monkeypatch, model_type, lora_config):
    # Execute the actual initializer without importing the optional GPU engine.
    helper = ROOT / "roll/third_party/vllm/lora_shrink.py"
    if helper.exists():
        name = "roll.third_party.vllm.lora_shrink"
        spec = importlib.util.spec_from_file_location(name, helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
    tree = ast.parse((ROOT / "roll/third_party/vllm/worker.py").read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerBase")
    method = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == "custom_init_worker")
    namespace = {"TensorLoraManager": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_worker_init", "exec"), namespace)
    owner = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
                            vllm_config=SimpleNamespace(lora_config=lora_config))
    namespace["custom_init_worker"](owner)


def native_config_module(monkeypatch):
    module = ModuleType(SHRINK_MODULE)
    configs = {"shrink": {"split_k": 64, "block_k": 256}, "expand": {"split_k": 8, "block_k": 32}}

    def select(op_type, *, batch):
        assert batch == 210
        return configs[op_type]

    module.get_lora_op_configs = select
    monkeypatch.setitem(sys.modules, SHRINK_MODULE, module)
    return module, configs


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text", "qwen3_8_flash_next", "qwen3_8_flash_next_text"])
def test_worker_uses_deterministic_shrink_without_mutating_native_config(monkeypatch, model_type):
    module, configs = native_config_module(monkeypatch)
    initialize_worker(monkeypatch, model_type, SimpleNamespace())
    assert module.get_lora_op_configs("shrink", batch=210) == {"split_k": 1, "block_k": 256}
    assert configs["shrink"] == {"split_k": 64, "block_k": 256}
    assert module.get_lora_op_configs("expand", batch=210) is configs["expand"]
    selector = module.get_lora_op_configs
    initialize_worker(monkeypatch, model_type, SimpleNamespace())
    assert module.get_lora_op_configs is selector


@pytest.mark.parametrize("model_type,lora_config", [("qwen4_exp", None), ("qwen3", SimpleNamespace()), (None, None)])
def test_unrelated_workers_do_not_patch_native_lora(monkeypatch, model_type, lora_config):
    module, _ = native_config_module(monkeypatch)
    selector = module.get_lora_op_configs
    initialize_worker(monkeypatch, model_type, lora_config)
    assert module.get_lora_op_configs is selector


@pytest.mark.parametrize("model_type,lora_config", [("qwen3", SimpleNamespace()), ("qwen4_exp", None)])
def test_worker_reinitialization_restores_native_selector(monkeypatch, model_type, lora_config):
    module, _ = native_config_module(monkeypatch)
    native = module.get_lora_op_configs
    initialize_worker(monkeypatch, "qwen4_exp", SimpleNamespace())
    assert module.get_lora_op_configs("shrink", batch=210)["split_k"] == 1
    initialize_worker(monkeypatch, model_type, lora_config)
    assert module.get_lora_op_configs is native


def test_reinitialization_preserves_a_later_native_wrapper(monkeypatch):
    module, _ = native_config_module(monkeypatch)
    initialize_worker(monkeypatch, "qwen4_exp", SimpleNamespace())
    previous = module.get_lora_op_configs

    @wraps(previous)
    def foreign(*args, **kwargs):
        return {**previous(*args, **kwargs), "foreign_setting": 7}

    module.get_lora_op_configs = foreign
    initialize_worker(monkeypatch, "qwen3", SimpleNamespace())
    assert module.get_lora_op_configs is foreign
    assert module.get_lora_op_configs("shrink", batch=210) == {
        "split_k": 64, "block_k": 256, "foreign_setting": 7}
    initialize_worker(monkeypatch, "qwen4_exp", SimpleNamespace())
    assert module.get_lora_op_configs("shrink", batch=210) == {
        "split_k": 1, "block_k": 256, "foreign_setting": 7}


def test_changed_native_config_contract_is_rejected(monkeypatch):
    module, configs = native_config_module(monkeypatch)
    del configs["shrink"]["split_k"]
    initialize_worker(monkeypatch, "qwen4_exp", SimpleNamespace())
    with pytest.raises(RuntimeError, match="split_k"):
        module.get_lora_op_configs("shrink", batch=210)


def test_missing_native_selector_is_reported_at_worker_initialization(monkeypatch):
    monkeypatch.setitem(sys.modules, SHRINK_MODULE, ModuleType(SHRINK_MODULE))
    with pytest.raises(RuntimeError, match="LoRA shrink"):
        initialize_worker(monkeypatch, "qwen4_exp", SimpleNamespace())
