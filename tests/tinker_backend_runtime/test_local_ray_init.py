"""Local runtime Ray isolation, resource configuration and failure cleanup."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    calls = []
    state = {"initialized": False, "fail_construct": False}

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(__import__("sys").modules, name, module)
        return module

    for name in ("roll", "roll.pipeline", "roll.pipeline.tinker_backend_runtime",
                 "roll.pipeline.agentic", "roll.distributed", "roll.distributed.scheduler", "roll.utils"):
        stub(name)
    protocol = stub("roll.pipeline.tinker_backend_runtime.types")
    protocol.__getattr__ = lambda name: object
    config = SimpleNamespace(num_gpus_per_node=8, num_nodes=1)
    stub("dacite", from_dict=lambda **kwargs: config)
    stub("hydra", compose=lambda **kwargs: {}, initialize_config_dir=lambda **kwargs: nullcontext())
    stub("omegaconf", OmegaConf=SimpleNamespace(to_container=lambda *args, **kwargs: {}))
    stub("roll.pipeline.agentic.agentic_config", AgenticConfig=object)
    stub("roll.distributed.scheduler.initialize", init=lambda: calls.append(("default_init", {})))
    stub("roll.platforms", current_platform=SimpleNamespace(get_custom_env_vars=lambda: {"PUBLIC_PLATFORM_FLAG": "1"}))
    stub("roll.utils.constants", RAY_NAMESPACE="roll")
    stub("ray", is_initialized=lambda: state["initialized"],
         init=lambda **kwargs: calls.append(("ray_init", kwargs)),
         shutdown=lambda: calls.append(("ray_shutdown", {})))

    def construct(pipeline_config):
        calls.append(("construct", pipeline_config))
        if state["fail_construct"]:
            raise ValueError("backend construction failed")
        return pipeline_config

    stub("roll.pipeline.tinker_backend_runtime.roll_backend", ROLLRuntimeBackend=construct)
    path = Path(__file__).resolve().parents[2] / "roll/pipeline/tinker_backend_runtime/train_adapters.py"
    spec = importlib.util.spec_from_file_location("local_ray_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, config, calls, state


def test_local_ray_starts_isolated_cluster_with_platform_environment(adapter, tmp_path):
    module, config, calls, _ = adapter
    result = module.init_roll_backend({"config_name": "example", "ray_address": "local", "ray_temp_dir": str(tmp_path)})
    assert result is config
    assert calls == [("ray_init", {
        "address": "local", "num_gpus": 8, "num_cpus": 16,
        "include_dashboard": False, "namespace": "roll",
        "runtime_env": {"env_vars": {"PUBLIC_PLATFORM_FLAG": "1"}}, "_temp_dir": str(tmp_path),
    }), ("construct", config)]


def test_default_ray_initialization_is_preserved(adapter):
    module, config, calls, _ = adapter
    assert module.init_roll_backend({"config_name": "example"}) is config
    assert calls == [("default_init", {}), ("construct", config)]


def test_local_ray_refuses_existing_connection_without_disconnect(adapter):
    module, _, calls, state = adapter
    state["initialized"] = True
    with pytest.raises(RuntimeError, match="existing Ray connection"):
        module.init_roll_backend({"config_name": "example", "ray_address": "local"})
    assert calls == []


def test_local_ray_cleans_own_cluster_if_backend_construction_fails(adapter):
    module, _, calls, state = adapter
    state["fail_construct"] = True
    with pytest.raises(ValueError, match="backend construction failed"):
        module.init_roll_backend({"config_name": "example", "ray_address": "local"})
    assert [name for name, _ in calls] == ["ray_init", "construct", "ray_shutdown"]
    assert "_temp_dir" not in calls[0][1]
