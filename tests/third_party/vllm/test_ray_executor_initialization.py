"""Opt-in Ray transport test; no model weights are loaded."""

import os
from types import SimpleNamespace

import pytest


def _initialize_transport_worker(wrapper):
    # The transport fixture does not load a model. Native block-size
    # finalization is a no-op when model_config is absent.
    from types import SimpleNamespace

    wrapper.vllm_config = SimpleNamespace(cache_config=None, model_config=None)


def _load_csa_cache_fixture(wrapper, model_path):
    """Register native cache owners without allocating model weights."""
    import torch
    from vllm.engine.arg_utils import AsyncEngineArgs
    # Match the native model registry import order (model imports qsa).
    from vllm.models.qwen3_8_flash_next.nvidia import model as _model
    from vllm.models.qwen3_8_flash_next.common.qsa_cache import (
        QSACompressedKeyCache, QSAKeyStateCache,
    )
    from vllm.models.qwen3_8_flash_next.nvidia.qsa import (
        Qwen3_8FlashNextQSAAttention, Qwen3_8FlashNextQSAFlashAttentionBackend,
    )

    config = AsyncEngineArgs(
        model=model_path, tensor_parallel_size=8, max_model_len=512,
        distributed_executor_backend="ray",
        max_num_batched_tokens=512, max_num_seqs=1, dtype="bfloat16",
        enforce_eager=True, enable_prefix_caching=False, trust_remote_code=True,
    ).create_engine_config()
    wrapper.vllm_config = config
    # Match Flash-Next's 24 query / 2 KV heads and TP8 geometry. Only the
    # attributes consumed by this native cache owner's methods are needed.
    main = Qwen3_8FlashNextQSAAttention.__new__(Qwen3_8FlashNextQSAAttention)
    torch.nn.Module.__init__(main)
    main.num_kv_heads = 1
    main.head_dim = 256
    main.kv_cache_torch_dtype = torch.bfloat16
    main.kv_cache_dtype = "auto"
    main.attn_backend = Qwen3_8FlashNextQSAFlashAttentionBackend
    config.compilation_config.static_forward_context["model.layers.3.self_attn"] = main
    for cls, suffix, extra in (
        (QSAKeyStateCache, "raw", {"cache_rope_positions": True}),
        (QSACompressedKeyCache, "compressed", {}),
    ):
        cls(head_size=128, dtype=torch.bfloat16, cache_config=config.cache_config,
            prefix=f"model.layers.3.self_attn.indexer.{suffix}",
            vllm_config=config, compress_ratio=4, **extra)


def _inspect_csa_cache_fixture(wrapper):
    from dataclasses import replace
    from vllm.models.qwen3_8_flash_next.nvidia.model import Qwen3_8FlashNextForConditionalGeneration
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_csa_linear
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    config = wrapper.vllm_config
    specs = get_kv_cache_spec(config)
    gdn = Qwen3_8FlashNextForConditionalGeneration.get_mamba_specs_from_config(config)[0]
    specs["model.layers.0.linear_attn"] = replace(gdn, block_size=config.cache_config.block_size)
    groups = _get_kv_cache_groups_csa_linear(config, specs)
    return {"block_size": config.cache_config.block_size, "groups": len(groups)}


def _worker_device_identity(wrapper):
    import os

    import torch

    return {
        "rank": wrapper.rpc_rank,
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device_count": torch.cuda.device_count(),
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
    }


@pytest.mark.parametrize("cache_fixture", [False, True], ids=["device_order", "csa_geometry"])
def test_native_ray_workers_preserve_explicit_device_order(monkeypatch, cache_fixture):
    # The old node/GPU RPC name fails on current native Ray actors. Replacing
    # ROLL's explicit device mapping with Ray's fractional allocation would
    # also incorrectly send both ranks to the same physical GPU.
    if os.environ.get("RUN_ROLL_VLLM_RAY_TESTS") != "1":
        pytest.skip("requires an isolated Ray cluster and four visible CUDA GPUs")
    model_path = os.environ.get("ROLL_VLLM_TEST_MODEL")
    if cache_fixture and not model_path:
        pytest.skip("requires a Flash-Next config directory for native cache geometry")

    import ray
    import torch

    from roll.distributed.scheduler.resource_manager import ResourceManager
    from roll.third_party.vllm.ray_distributed_executor import CustomRayDistributedExecutor

    assert not ray.is_initialized(), "test must own its Ray cluster"
    assert torch.cuda.device_count() >= 4
    expected_uuids = [str(torch.cuda.get_device_properties(i).uuid) for i in (3, 1)]
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=4,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
    )

    class TransportOnlyExecutor(CustomRayDistributedExecutor):
        # Keep the real Ray wrappers, actor creation, rank/environment RPCs
        # and device discovery. Only stop the subsequent heavyweight model
        # construction; the full RL smoke covers that independent boundary.
        def collective_rpc(self, method, *args, **kwargs):
            if method == "init_worker":
                self.worker_init_kwargs = kwargs["args"][0]
                return super().collective_rpc(_initialize_transport_worker)
            if method == "load_model" and cache_fixture:
                return super().collective_rpc(_load_csa_cache_fixture, args=(model_path,))
            if method in ("init_device", "load_model"):
                return [None] * len(self.workers)
            return super().collective_rpc(method, *args, **kwargs)

    manager = None
    executor = TransportOnlyExecutor.__new__(TransportOnlyExecutor)
    executor.forward_dag = None
    executor.workers = []
    executor.parallel_config = SimpleNamespace(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        ray_workers_use_nsight=False,
    )
    executor.vllm_config = SimpleNamespace(parallel_config=executor.parallel_config)
    try:
        manager = ResourceManager(num_gpus_per_node=4, num_nodes=1)
        groups = manager.allocate_placement_group(world_size=1, device_mapping=[3, 1])[0]
        executor._init_workers_ray(groups)
        if cache_fixture:
            # Omitting post-load block-size finalization reproduces the real
            # CSA geometry failure: the raw state page exceeds the compressed
            # page. This RPC runs the installed native grouping validator.
            caches = executor.collective_rpc(_inspect_csa_cache_fixture)
            assert all(item["groups"] == 3 for item in caches)
            assert all(item["block_size"] > 16 for item in caches)
        identities = executor.collective_rpc(_worker_device_identity)
        assert [item["rank"] for item in identities] == [0, 1]
        assert [item["visible_devices"] for item in identities] == ["3", "1"]
        assert [item["device_count"] for item in identities] == [1, 1]
        assert [item["uuid"] for item in identities] == expected_uuids
        assert [item["local_rank"] for item in executor.worker_init_kwargs] == [0, 0]
        assert [item["rank"] for item in executor.worker_init_kwargs] == [0, 1]
    finally:
        for worker in executor.workers:
            ray.kill(worker)
        executor.workers = []
        del executor
        if manager is not None:
            manager.destroy_placement_group()
        ray.shutdown()
