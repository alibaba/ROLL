import asyncio
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest


COMPAT_PATH = Path(__file__).parents[3] / "roll" / "third_party" / "vllm" / "compat.py"


def load_compat():
    spec = importlib.util.spec_from_file_location("roll_vllm_compat_under_test", COMPAT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_import_attribute_uses_first_available_capability(monkeypatch):
    compat = load_compat()
    expected = object()
    install_module(monkeypatch, "test_vllm_old")
    install_module(monkeypatch, "test_vllm_new", LoRAModel=expected)

    actual = compat.import_attribute(("test_vllm_old", "test_vllm_new"), "LoRAModel")

    assert actual is expected


def test_attention_config_null_delegates_backend_selection():
    compat = load_compat()

    explicit_default = {"attention_config": None}
    compat.apply_default_attention_config(explicit_default, supports_attention_config=True)
    assert "attention_config" not in explicit_default

    omitted = {}
    compat.apply_default_attention_config(omitted, supports_attention_config=True)
    assert omitted == {"attention_config": {"backend": "FLASH_ATTN"}}


def test_import_attribute_reports_all_attempted_locations(monkeypatch):
    compat = load_compat()
    install_module(monkeypatch, "test_vllm_empty")

    with pytest.raises(ImportError, match=r"Missing.*test_vllm_empty.*test_vllm_absent"):
        compat.import_attribute(("test_vllm_empty", "test_vllm_absent"), "Missing")


def test_module_has_attributes_requires_the_native_ray_contract(monkeypatch):
    compat = load_compat()
    install_module(
        monkeypatch,
        "test_vllm_ray",
        RayDistributedExecutor=object(),
        RayWorkerMetaData=object(),
    )

    assert compat.module_has_attributes(
        "test_vllm_ray", ("RayDistributedExecutor", "RayWorkerMetaData")
    )
    assert not compat.module_has_attributes(
        "test_vllm_ray", ("RayDistributedExecutor", "Missing")
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_call_maybe_await_accepts_sync_and_async_results(asynchronous):
    compat = load_compat()
    expected = object()

    async def async_call():
        return expected

    def sync_call():
        return expected

    result = asyncio.run(compat.call_maybe_await(async_call if asynchronous else sync_call))

    assert result is expected


def test_worker_delegates_level_two_buffer_ownership_to_native_sleep_wake():
    compat = load_compat()
    calls = []

    class NativeWorker:
        _sleep_saved_buffers = {}
        _sleep_saved_draft_buffers = {}

        def sleep(self, level):
            calls.append(("sleep", level))

        def wake_up(self, tags):
            calls.append(("wake_up", tags))

    worker = NativeWorker()

    assert compat.native_sleep_owns_buffers(worker)
    compat.native_sleep(worker, 2)
    compat.native_wake_up(worker, ["weights", "kv_cache"])

    assert calls == [("sleep", 2), ("wake_up", ["weights", "kv_cache"])]


def test_worker_detects_when_legacy_buffer_fallback_is_required():
    compat = load_compat()

    assert not compat.native_sleep_owns_buffers(object())


def test_process_weights_after_loading_uses_available_context_manager(monkeypatch):
    compat = load_compat()
    calls = []

    def process(model, model_config, target_device):
        calls.append((model, model_config, target_device))

    class DTypeContext:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, *_):
            calls.append("exit")

    install_module(monkeypatch, "test_vllm_loader", process_weights_after_loading=process)
    install_module(monkeypatch, "test_vllm_torch", set_default_torch_dtype=lambda _: DTypeContext())
    model = object()
    model_config = types.SimpleNamespace(dtype="bf16")

    processed = compat.process_weights_after_loading(
        model=model,
        model_config=model_config,
        target_device="cpu",
        loader_modules=("test_vllm_loader",),
        torch_utils_modules=("test_vllm_torch",),
    )

    assert processed
    assert calls == ["enter", (model, model_config, "cpu"), "exit"]


@pytest.mark.parametrize("legacy_version", ["0.8.4", "0.10.2"])
def test_legacy_process_weights_after_loading_is_noop_without_capability(
    monkeypatch, legacy_version
):
    compat = load_compat()
    install_module(monkeypatch, f"test_vllm_loader_{legacy_version}")
    install_module(monkeypatch, f"test_vllm_torch_{legacy_version}")

    processed = compat.process_weights_after_loading(
        model=object(),
        model_config=types.SimpleNamespace(dtype="bf16"),
        target_device="cpu",
        loader_modules=(f"test_vllm_loader_{legacy_version}",),
        torch_utils_modules=(f"test_vllm_torch_{legacy_version}",),
    )

    assert not processed


@pytest.mark.parametrize(
    "internal_error",
    [
        ImportError("transitive dependency is broken"),
        ModuleNotFoundError("transitive dependency is missing", name="broken_dependency"),
    ],
)
def test_process_weights_after_loading_propagates_internal_import_error(
    monkeypatch, internal_error
):
    compat = load_compat()
    real_import_module = compat.importlib.import_module

    def broken_import(module_name):
        if module_name == "test_vllm_broken_loader":
            raise internal_error
        return real_import_module(module_name)

    monkeypatch.setattr(compat.importlib, "import_module", broken_import)

    with pytest.raises(ImportError, match="transitive dependency"):
        compat.process_weights_after_loading(
            model=object(),
            model_config=types.SimpleNamespace(dtype="bf16"),
            target_device="cpu",
            loader_modules=("test_vllm_broken_loader",),
            torch_utils_modules=("test_vllm_unused_torch",),
        )


def test_moe_loader_patch_skips_dense_layers_and_preserves_native_ownership():
    compat = load_compat()

    class Parameter:
        pass

    dense_parameter = Parameter()
    replacement_parameter = Parameter()
    native_parameter = Parameter()

    class Experts:
        def weight_loader(self):
            pass

    experts = Experts()
    native_parameter.weight_loader = experts.weight_loader

    class MLP:
        def __init__(self, parameters, experts=None):
            self.parameters = parameters
            if experts is not None:
                self.experts = experts

        def named_parameters(self):
            return self.parameters

    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            layers=[
                types.SimpleNamespace(mlp=MLP([("dense.weight", dense_parameter)])),
                types.SimpleNamespace(
                    mlp=MLP(
                        [
                            ("experts.w13_weight", replacement_parameter),
                            ("experts.w2_weight", native_parameter),
                        ],
                        experts,
                    )
                ),
            ]
        )
    )

    compat.patch_moe_model_weight_loaders(model)

    assert not hasattr(dense_parameter, "weight_loader")
    assert replacement_parameter.weight_loader.__self__ is experts
    assert native_parameter.weight_loader.__self__ is experts


def test_installed_vllm_cpu_import_contract():
    vllm = pytest.importorskip("vllm")
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.executor.ray_executor import RayDistributedExecutor
    from vllm.v1.worker.gpu_worker import Worker

    import roll.third_party.vllm as roll_vllm
    from roll.third_party.vllm import vllm_utils
    from roll.third_party.vllm.vllm_utils import LoRAModel
    from roll.third_party.vllm.worker import WorkerBase

    assert vllm.__version__
    assert LoRAModel.__module__ in {"vllm.lora.lora_model", "vllm.lora.models"}
    assert issubclass(roll_vllm.ray_executor_class_v1, RayDistributedExecutor)
    assert callable(AsyncLLM.get_tokenizer)
    if vllm.__version__.startswith("0.1.dev"):
        assert LoRAModel.__module__ == "vllm.lora.lora_model"
        assert not inspect.iscoroutinefunction(AsyncLLM.get_tokenizer)
        assert "_sleep_saved_buffers" in inspect.getsource(Worker.sleep)
        assert "_sleep_saved_draft_buffers" in inspect.getsource(Worker.wake_up)

        class Parameter:
            pass

        class DenseMLP:
            def named_parameters(self):
                return [("dense.weight", Parameter())]

        model = types.SimpleNamespace(
            model=types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=DenseMLP())])
        )
        original_models = vllm_utils.SUPPORTED_MOE_MODELS[:]
        try:
            vllm_utils.SUPPORTED_MOE_MODELS[:] = [type(model)]
            vllm_utils.patch_vllm_moe_model_weight_loader(model)
        finally:
            vllm_utils.SUPPORTED_MOE_MODELS[:] = original_models
    WorkerBase().custom_init_worker()
