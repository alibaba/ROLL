import ctypes
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from roll.utils import nccl_suspend


@pytest.fixture(autouse=True)
def reset_nccl_suspend_state(monkeypatch):
    monkeypatch.setattr(nccl_suspend, "_nccl_library", None)
    monkeypatch.setattr(nccl_suspend, "_nccl_library_path", None)
    monkeypatch.setattr(nccl_suspend, "_nccl_runtime_version", None)
    monkeypatch.setattr(nccl_suspend, "_nccl_load_error", None)
    monkeypatch.setattr(nccl_suspend, "_nccl_load_attempted", False)
    nccl_suspend._suspended_handles_by_pid.clear()
    yield
    nccl_suspend._suspended_handles_by_pid.clear()


def test_add_process_group_handle_deduplicates_initialized_comms():
    class Backend:
        def _comm_ptr(self):
            return 1234

    class ProcessGroup:
        def _get_backend(self, device):
            assert device.type == "cuda"
            return Backend()

    handles = []
    seen_handles = set()
    nccl_suspend._add_process_group_handle(handles, seen_handles, "first", ProcessGroup())
    nccl_suspend._add_process_group_handle(handles, seen_handles, "second", ProcessGroup())

    assert handles == [("first", 1234)]


def test_suspend_and_resume_keep_communicators_and_are_idempotent(monkeypatch):
    handles = [("tp", 101), ("dp", 202)]
    calls = []

    monkeypatch.setattr(nccl_suspend, "collect_nccl_communicator_handles", lambda: handles)
    monkeypatch.setattr(nccl_suspend, "require_nccl_comm_suspend_support", lambda: None)
    monkeypatch.setattr(nccl_suspend, "_synchronize_cuda", lambda: calls.append("sync"))
    monkeypatch.setattr(nccl_suspend, "suspend_nccl_comm", lambda handle: calls.append(("suspend", handle)))
    monkeypatch.setattr(nccl_suspend, "resume_nccl_comm", lambda handle: calls.append(("resume", handle)))

    assert nccl_suspend.suspend_nccl_communicators() == handles
    assert nccl_suspend.suspend_nccl_communicators() == handles
    assert nccl_suspend.resume_nccl_communicators() == handles
    assert nccl_suspend.resume_nccl_communicators() == []

    assert calls == [
        "sync",
        ("suspend", 101),
        ("suspend", 202),
        "sync",
        ("resume", 101),
        ("resume", 202),
    ]
    assert os.getpid() not in nccl_suspend._suspended_handles_by_pid


def test_require_support_has_no_legacy_fallback(monkeypatch):
    monkeypatch.setattr(nccl_suspend, "is_nccl_comm_suspend_supported", lambda: False)
    monkeypatch.setattr(nccl_suspend, "_find_loaded_nccl_library", lambda: "/opt/nccl/libnccl.so.2")

    with pytest.raises(nccl_suspend.NcclCommSuspendUnavailable, match="NCCL >= 2.29.7"):
        nccl_suspend.require_nccl_comm_suspend_support()


def test_collection_excludes_cross_role_weight_update_groups(monkeypatch):
    from types import SimpleNamespace

    from roll.utils.collective import collective

    train_group, update_group = object(), object()
    monkeypatch.setattr(nccl_suspend.dist, "is_available", lambda: True)
    monkeypatch.setattr(nccl_suspend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(nccl_suspend.dist, "distributed_c10d", SimpleNamespace(
        _world=SimpleNamespace(pg_map={train_group: None, update_group: None})
    ))
    monkeypatch.setattr(collective._group_mgr, "_name_group_map", {
        "model_update/actor_train_2_actor_infer_pp0_ep0": update_group
    })
    collected = []
    monkeypatch.setattr(nccl_suspend, "_add_process_group_handle", lambda h, s, label, group: collected.append(group))
    nccl_suspend._collect_torch_process_groups([], set())
    assert collected == [train_group]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("offload_states", [False, True])
def test_state_offload_preserves_model_and_nccl_order(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, offload_states: bool
) -> None:
    """Model state transitions must remain ordered without optional instrumentation."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from roll.utils import context_managers

    calls = []
    metrics = {}
    strategy = SimpleNamespace(
        offload_nccl=enabled,
        load_states=lambda **kwargs: calls.append("load_model"),
        offload_states=lambda **kwargs: calls.append("offload_model"),
    )
    platform = SimpleNamespace(device_count=lambda: 0, clear_cublas_workspaces=lambda: None)
    monkeypatch.setattr(context_managers, "current_platform", platform)
    monkeypatch.setattr(context_managers, "is_roll_debug_mode", lambda: False)
    monkeypatch.setattr(context_managers, "log_gpu_memory_usage", lambda **kwargs: None)
    monkeypatch.setattr(context_managers, "local_profiler", nullcontext)
    monkeypatch.setattr(context_managers, "gpu_memory_offload_profiler", lambda *args: nullcontext())
    monkeypatch.setattr(context_managers, "resume_nccl_communicators", lambda: calls.append("resume_nccl"))
    monkeypatch.setattr(context_managers, "suspend_nccl_communicators", lambda: calls.append("suspend_nccl"))

    with context_managers.state_offload_manger(
        strategy, metrics, "actor_train/model_update", is_offload_states=offload_states
    ):
        calls.append("execute")

    expected = ["load_model"]
    if enabled:
        expected.append("resume_nccl")
    expected.append("execute")
    if offload_states:
        expected.append("offload_model")
        if enabled:
            expected.append("suspend_nccl")
    assert calls == expected
    prefix = "time/actor_train/model_update"
    assert (f"{prefix}/resume_nccl" in metrics) == enabled
    assert (f"{prefix}/suspend_nccl" in metrics) == (enabled and offload_states)


def mock_native_library(monkeypatch, version: int = 22907, status: int = 0):
    """Supply a mapped native library without loading CUDA or NCCL locally."""
    def get_version(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = version
        return status

    library = SimpleNamespace(
        ncclGetVersion=Mock(side_effect=get_version),
        ncclCommSuspend=Mock(return_value=0),
        ncclCommResume=Mock(return_value=0),
    )
    monkeypatch.setattr(nccl_suspend, "_find_loaded_nccl_library", lambda: "/opt/nccl/libnccl.so.2")
    monkeypatch.setattr(nccl_suspend.ctypes, "CDLL", lambda path: library)
    # The check must use ncclGetVersion, not torch's compile-time metadata.
    monkeypatch.setattr(nccl_suspend.torch.cuda.nccl, "version", Mock(side_effect=AssertionError("build metadata")))
    return library


@pytest.mark.parametrize("version,supported", [
    (0, False), (22809, False), (22906, False),
    (22907, True), (22909, True), (23102, True), (30000, True), (30100, True),
])
def test_runtime_version_gate(monkeypatch, version, supported):
    library = mock_native_library(monkeypatch, version)
    assert nccl_suspend.is_nccl_comm_suspend_supported() is supported
    assert nccl_suspend._nccl_runtime_version == version
    if supported:
        nccl_suspend.require_nccl_comm_suspend_support()
        assert nccl_suspend._get_nccl_library() is library
    else:
        with pytest.raises(nccl_suspend.NcclCommSuspendUnavailable, match=f"runtime: {version}"):
            nccl_suspend.require_nccl_comm_suspend_support()
    library.ncclGetVersion.assert_called_once()


@pytest.mark.parametrize("symbol", ["ncclGetVersion", "ncclCommSuspend", "ncclCommResume"])
def test_missing_native_symbol_is_rejected(monkeypatch, symbol):
    library = mock_native_library(monkeypatch)
    delattr(library, symbol)
    with pytest.raises(nccl_suspend.NcclCommSuspendUnavailable, match=f"missing {symbol}"):
        nccl_suspend.require_nccl_comm_suspend_support()


def test_failed_native_version_query_is_rejected(monkeypatch):
    mock_native_library(monkeypatch, status=3)
    with pytest.raises(nccl_suspend.NcclCommSuspendUnavailable, match="status=3"):
        nccl_suspend.require_nccl_comm_suspend_support()


def test_native_calls_use_memory_flag_and_original_handle(monkeypatch):
    library = mock_native_library(monkeypatch)
    nccl_suspend.suspend_nccl_comm(101)
    nccl_suspend.resume_nccl_comm(101)
    suspend_args = library.ncclCommSuspend.call_args.args
    assert suspend_args[0].value == 101
    assert suspend_args[1] == 0x01
    assert library.ncclCommResume.call_args.args[0].value == 101


def test_partial_suspend_rolls_back_before_propagating_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(nccl_suspend, "collect_nccl_communicator_handles", lambda: [("tp", 101), ("dp", 202)])
    monkeypatch.setattr(nccl_suspend, "require_nccl_comm_suspend_support", lambda: None)
    monkeypatch.setattr(nccl_suspend, "_synchronize_cuda", lambda: None)

    def suspend(handle):
        if handle == 202:
            raise RuntimeError("suspend failed")
        calls.append(("suspend", handle))

    monkeypatch.setattr(nccl_suspend, "suspend_nccl_comm", suspend)
    monkeypatch.setattr(nccl_suspend, "resume_nccl_comm", lambda handle: calls.append(("resume", handle)))
    with pytest.raises(RuntimeError, match="suspend failed"):
        nccl_suspend.suspend_nccl_communicators()
    assert calls == [("suspend", 101), ("resume", 101)]
    assert not nccl_suspend._suspended_handles_by_pid


@pytest.mark.parametrize("strategy", ["megatron_train", "megatron_infer"])
def test_worker_config_preserves_offload_opt_in(strategy):
    from roll.configs.worker_config import StrategyArguments, WorkerConfig

    args = StrategyArguments(strategy_name=strategy)
    assert WorkerConfig(strategy_args=args, offload_nccl=True).offload_nccl is True
    assert WorkerConfig(strategy_args=args).offload_nccl is False


@pytest.mark.parametrize("strategy", ["vllm", "vllm_omni"])
def test_vllm_offload_is_explicitly_unsupported(strategy):
    from roll.configs.worker_config import StrategyArguments, WorkerConfig

    args = StrategyArguments(strategy_name=strategy)
    assert WorkerConfig(strategy_args=args).offload_nccl is False
    with pytest.raises(ValueError, match="not supported for vLLM"):
        WorkerConfig(strategy_args=args, offload_nccl=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_only_validates_enabled_offload_before_ray_setup(monkeypatch, enabled):
    from roll.distributed.executor import worker

    validate = Mock(side_effect=nccl_suspend.NcclCommSuspendUnavailable("unsupported runtime"))
    storage = SimpleNamespace(options=Mock(side_effect=RuntimeError("reached Ray setup")))
    monkeypatch.setattr(worker, "require_nccl_comm_suspend_support", validate)
    monkeypatch.setattr(worker, "SharedStorage", storage)
    expected = "unsupported runtime" if enabled else "reached Ray setup"
    with pytest.raises(RuntimeError, match=expected):
        worker.Worker(SimpleNamespace(offload_nccl=enabled))
    assert validate.call_count == int(enabled)
    assert storage.options.call_count == int(not enabled)
