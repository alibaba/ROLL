"""A level-2 wake must retain external tables without parking the whole model."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).parents[3]
PATH = ROOT / "roll/third_party/vllm/frozen_ngram_sleep.py"
spec = importlib.util.spec_from_file_location("frozen_ngram_sleep", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FrozenNGramSleepState = module.FrozenNGramSleepState


def test_flash_next_frozen_tables_are_loaded_from_checkpoint_before_weight_sync():
    _, config = make_model()
    loading = SimpleNamespace(load_format="dummy")
    module.configure_frozen_ngram_loading(config, loading)
    assert loading.load_format == "auto"


@pytest.mark.parametrize("model_type, layers, initial", [
    ("qwen3_5", [2], "dummy"), ("qwen4_exp", [], "dummy"),
    ("qwen4_exp", [2], "safetensors"), ("qwen4_exp", [2], "auto"),
])
def test_external_table_load_policy_preserves_other_models_and_real_loaders(model_type, layers, initial):
    config = SimpleNamespace(model_type=model_type, text_config=SimpleNamespace(ple_layer_ids=layers))
    loading = SimpleNamespace(load_format=initial)
    module.configure_frozen_ngram_loading(config, loading)
    assert loading.load_format == initial


def make_model(layer_ids=(2,)):
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
    model.backbone = nn.Linear(3, 3, bias=False, dtype=torch.bfloat16)
    for index in layer_ids:
        layer = model.layers[index - 1]
        layer.ple = nn.Module()
        layer.ple.ple_embedding = nn.Module()
        layer.ple.ple_embedding.ngram_embedding = nn.Embedding(8, 3, dtype=torch.bfloat16)
        with torch.no_grad():
            layer.ple.ple_embedding.ngram_embedding.weight.copy_(torch.arange(24).reshape(8, 3) + index)
    config = SimpleNamespace(model_type="qwen4_exp", text_config=SimpleNamespace(ple_layer_ids=list(layer_ids)))
    return model, config


def table(model, index=2):
    return model.layers[index - 1].ple.ple_embedding.ngram_embedding.weight


def test_two_cycles_restore_only_current_frozen_tables_with_bounded_transfers(monkeypatch):
    model, config = make_model((2, 4))
    original_copy = torch.Tensor.copy_
    transfers = []

    def copy_(destination, source, *args, **kwargs):
        transfers.append(source.numel() * source.element_size())
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    for cycle in range(2):
        with torch.no_grad():
            table(model, 2).add_(cycle)
        expected = [table(model, index).detach().clone() for index in (2, 4)]
        snapshot = FrozenNGramSleepState.capture(model, config, chunk_bytes=8)
        assert snapshot.cpu_bytes == 96
        # Simulate level-2 discard of all parameters; this helper is responsible
        # for the frozen tables alone, not the trainable weight sync protocol.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float("nan"))
        snapshot.restore(model)
        for index, wanted in zip((2, 4), expected):
            torch.testing.assert_close(table(model, index), wanted, atol=0, rtol=0)
        assert torch.isnan(model.backbone.weight).all()
        assert snapshot.cpu_bytes == 0
    assert transfers and max(transfers) <= 8


@pytest.mark.parametrize("fault", ["missing", "wrong_layer", "extra", "dtype", "noncontiguous"])
def test_capture_rejects_incomplete_or_unsupported_tables_before_sleep(fault):
    model, config = make_model()
    if fault == "missing":
        del model.layers[1].ple
    elif fault == "wrong_layer":
        config.text_config.ple_layer_ids = [1]
    elif fault == "extra":
        other, _ = make_model((4,))
        model.layers[3] = other.layers[3]
    elif fault == "dtype":
        table(model).requires_grad_(False)
        table(model).data = torch.ones(8, 3, dtype=torch.int32)
    else:
        table(model).data = torch.zeros(3, 8, dtype=torch.bfloat16).t()
    with pytest.raises(ValueError, match="N-gram"):
        FrozenNGramSleepState.capture(model, config)


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype"])
def test_restore_validates_every_table_before_mutating_any_table(fault):
    model, config = make_model((2, 4))
    snapshot = FrozenNGramSleepState.capture(model, config)
    with torch.no_grad():
        table(model, 2).zero_()
    if fault == "missing":
        del model.layers[3].ple
    elif fault == "shape":
        table(model, 4).data = torch.zeros(7, 3, dtype=torch.bfloat16)
    else:
        table(model, 4).data = torch.zeros(8, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="N-gram"):
        snapshot.restore(model)
    assert torch.count_nonzero(table(model, 2)) == 0
    assert snapshot.cpu_bytes == 96


@pytest.mark.parametrize("model_type", ["qwen4_exp_text", "qwen3_8_flash_next_text"])
def test_text_config_aliases_are_supported(model_type):
    model, config = make_model()
    config.text_config.model_type = model_type
    snapshot = FrozenNGramSleepState.capture(model, config.text_config)
    assert snapshot.cpu_bytes == 48


def test_other_model_has_no_frozen_table_backup():
    model, _ = make_model()
    assert FrozenNGramSleepState.capture(model, SimpleNamespace(model_type="qwen3_5")) is None


def test_lora_only_weight_sync_cannot_repopulate_discarded_backbone():
    model, config = make_model()
    with pytest.raises(ValueError, match="full model"):
        FrozenNGramSleepState.capture(model, config, lora_enabled=True)


def _worker(model, config):
    """Execute real WorkerBase methods with the native sleep transport replaced."""
    path = ROOT / "roll/third_party/vllm/worker.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "WorkerBase")
    names = {"reload_model", "offload_states", "load_states"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    events = []

    def sleep(worker, level):
        events.append(("sleep", level))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

    def wake(worker, tags):
        events.append(("wake", tags))

    namespace = dict(FrozenNGramSleepState=FrozenNGramSleepState, native_sleep=sleep,
                     restore_ngram_context_offsets=module.restore_ngram_context_offsets,
                     native_wake_up=wake, native_sleep_owns_buffers=lambda _: True,
                     clear_memory=lambda: None)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model), model_config=SimpleNamespace(hf_config=config),
                             weight_loaded=True, kv_cache_loaded=True, _frozen_ngram_sleep_state=None,
                             _sleep_wake_failed=False)
    for name in names:
        setattr(worker, name, namespace[name].__get__(worker))
    return worker, events


def test_native_kv_wake_failure_requires_reconstruction_without_repeated_wake():
    model, config = make_model()
    worker, events = _worker(model, config)
    worker.offload_states(2)
    worker.reload_model()
    namespace = worker.load_states.__func__.__globals__
    original = namespace["native_wake_up"]

    def fail_after_wake(worker, tags):
        original(worker, tags)
        raise RuntimeError("KV wake completed only partly")

    namespace["native_wake_up"] = fail_after_wake
    with pytest.raises(RuntimeError, match="only partly"):
        worker.load_states()
    assert not worker.kv_cache_loaded
    assert worker._sleep_wake_failed
    with pytest.raises(RuntimeError, match="reconstruct"):
        worker.load_states()
    assert events.count(("wake", ["kv_cache"])) == 1


def test_roll_worker_preserves_table_before_native_sleep_and_restores_before_ready():
    model, config = make_model()
    expected = table(model).detach().clone()
    worker, events = _worker(model, config)
    worker.offload_states(2)
    assert not worker.weight_loaded
    assert torch.count_nonzero(table(model)) == 0
    worker.reload_model()
    torch.testing.assert_close(table(model), expected, atol=0, rtol=0)
    assert worker.weight_loaded
    assert worker._frozen_ngram_sleep_state is None
    assert events == [("sleep", 2), ("wake", ["weights"])]


def test_invalid_table_prevents_native_sleep():
    model, config = make_model()
    del model.layers[1].ple
    worker, events = _worker(model, config)
    with pytest.raises(ValueError, match="N-gram"):
        worker.offload_states(2)
    assert events == [] and worker.weight_loaded


@pytest.mark.parametrize("discarded", [False, True])
def test_native_sleep_failure_preserves_snapshot_and_requires_worker_reconstruction(discarded):
    model, config = make_model()
    expected = table(model).detach().clone()
    worker, events = _worker(model, config)
    namespace = worker.offload_states.__func__.__globals__
    original_sleep = namespace["native_sleep"]

    def fail(worker, level):
        if discarded:
            original_sleep(worker, level)
        raise RuntimeError("injected native sleep failure")

    namespace["native_sleep"] = fail
    with pytest.raises(RuntimeError, match="injected"):
        worker.offload_states(2)
    snapshot = worker._frozen_ngram_sleep_state
    assert snapshot.cpu_bytes == 48
    assert not worker.weight_loaded and not worker.kv_cache_loaded
    for operation in (lambda: worker.offload_states(2), worker.reload_model):
        with pytest.raises(RuntimeError, match="reconstruct"):
            operation()
        assert worker._frozen_ngram_sleep_state is snapshot
    # The original valid table remains available for diagnosis/reconstruction;
    # no retry may recapture potentially unmapped or discarded GPU storage.
    snapshot.restore(model)
    torch.testing.assert_close(table(model), expected, atol=0, rtol=0)
    assert events == ([("sleep", 2)] if discarded else [])


def test_table_copy_retry_does_not_repeat_native_weight_wake(monkeypatch):
    model, config = make_model()
    expected = table(model).detach().clone()
    worker, events = _worker(model, config)
    worker.offload_states(2)
    copy = module._copy_chunks

    def fail(*args):
        raise RuntimeError("injected table transfer failure")

    monkeypatch.setattr(module, "_copy_chunks", fail)
    with pytest.raises(RuntimeError, match="transfer"):
        worker.reload_model()
    assert not worker.weight_loaded
    assert worker._frozen_ngram_sleep_state.cpu_bytes == 48
    monkeypatch.setattr(module, "_copy_chunks", copy)
    worker.reload_model()
    torch.testing.assert_close(table(model), expected, atol=0, rtol=0)
    assert worker.weight_loaded and worker._frozen_ngram_sleep_state is None
    assert events == [("sleep", 2), ("wake", ["weights"])]


def test_native_weight_wake_failure_requires_reconstruction():
    model, config = make_model()
    worker, events = _worker(model, config)
    worker.offload_states(2)
    namespace = worker.reload_model.__func__.__globals__

    def fail(worker, tags):
        events.append(("wake_failed", tags))
        raise RuntimeError("injected native wake failure")

    namespace["native_wake_up"] = fail
    with pytest.raises(RuntimeError, match="injected"):
        worker.reload_model()
    assert not worker.weight_loaded
    for operation in (worker.reload_model, lambda: worker.offload_states(2)):
        with pytest.raises(RuntimeError, match="reconstruct"):
            operation()
    assert worker._frozen_ngram_sleep_state.cpu_bytes == 48
    assert events == [("sleep", 2), ("wake_failed", ["weights"])]


def test_level_one_does_not_capture_external_tables(monkeypatch):
    model, config = make_model()
    worker, events = _worker(model, config)

    def fail(*args, **kwargs):
        raise AssertionError("level 1 must use native backup")

    monkeypatch.setattr(FrozenNGramSleepState, "capture", fail)
    worker.offload_states(1)
    worker.reload_model()
    assert events == [("sleep", 1), ("wake", ["weights"])]
