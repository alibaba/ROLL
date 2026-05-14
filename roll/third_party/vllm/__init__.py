import os
import pathlib
from typing import Dict, List

import torch
import vllm
from packaging.version import Version
from vllm import envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.envs import get_default_cache_root
from vllm.usage.usage_lib import UsageContext

from roll.platforms import current_platform
import roll.third_party.vllm.fp8 as fp8
from roll.third_party.vllm.versioning import supports_vllm_0_11_v0_ray_executor, uses_vllm_0_11_adapter
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger


logger = get_logger()

if Version("0.8.4") == Version(vllm.__version__):
    import roll.third_party.vllm.vllm_0_8_4 # apply patch
    ray_executor_class_v0 = safe_import_class("roll.third_party.vllm.vllm_0_8_4.ray_distributed_executor.CustomRayDistributedExecutor")
    ray_executor_class_v1 = safe_import_class("roll.third_party.vllm.vllm_0_8_4.v1.ray_distributed_executor.CustomRayDistributedExecutor")
elif Version("0.10.2") == Version(vllm.__version__):
    ray_executor_class_v0 = safe_import_class("roll.third_party.vllm.vllm_0_10_2.ray_distributed_executor.CustomRayDistributedExecutor")
    ray_executor_class_v1 = safe_import_class("roll.third_party.vllm.vllm_0_10_2.v1.ray_distributed_executor.CustomRayDistributedExecutor")
elif uses_vllm_0_11_adapter(vllm.__version__):
    ray_executor_class_v0 = None
    if supports_vllm_0_11_v0_ray_executor(vllm.__version__):
        ray_executor_class_v0 = safe_import_class("roll.third_party.vllm.vllm_0_11_0.ray_distributed_executor.CustomRayDistributedExecutor")
        ray_executor_class_v1 = safe_import_class("roll.third_party.vllm.vllm_0_11_0.v1.ray_distributed_executor.CustomRayDistributedExecutor")
    else:
        ray_executor_class_v1 = safe_import_class("roll.third_party.vllm.vllm_0_12_0.ray_distributed_executor.CustomRayDistributedExecutor")
elif Version("0.12.0") == Version(vllm.__version__):
    ray_executor_class_v0 = None  # V0 deprecated
    ray_executor_class_v1 = safe_import_class("roll.third_party.vllm.vllm_0_12_0.ray_distributed_executor.CustomRayDistributedExecutor")
elif Version("0.15") <= Version(vllm.__version__):
    if Version("0.16").release <= Version(vllm.__version__).release:
        import roll.third_party.vllm.patch_transformers # apply patch
    ray_executor_class_v0 = None  # V0 deprecated
    ray_executor_class_v1 = safe_import_class("roll.third_party.vllm.ray_distributed_executor.CustomRayDistributedExecutor")
else:
    ray_executor_class_v0 = None
    ray_executor_class_v1 = None
    logger.warning(f"ROLL is not tested on vllm version {vllm.__version__}, something strange may happen!!!")

logger.info(f"Using vllm version {vllm.__version__}")


def _is_cumem_available() -> bool:
    try:
        from vllm.device_allocator.cumem import cumem_available

        return bool(cumem_available)
    except Exception as exc:
        logger.warning("Failed to probe vLLM cumem allocator availability: %s", exc)
        return False


def _resolve_enable_sleep_mode(requested: bool) -> bool:
    if not requested:
        return False

    if _is_cumem_available():
        return True

    logger.warning(
        "vLLM sleep mode requested but cumem allocator is unavailable on this platform. "
        "Disabling sleep mode and continuing without vLLM offload/reload."
    )
    return False


async def create_async_llm(resource_placement_groups: List[Dict], **kwargs):
    requested_sleep_mode = kwargs.pop("enable_sleep_mode", True)
    kwargs["enable_sleep_mode"] = _resolve_enable_sleep_mode(requested_sleep_mode)

    if "worker_extension_cls" not in kwargs:
        # VLLM_USE_V1 is deprecated in vllm>=0.11.1
        if not hasattr(envs, "VLLM_USE_V1") or envs.VLLM_USE_V1:
            kwargs["worker_extension_cls"] = "roll.third_party.vllm.worker.WorkerV1"
        else:
            kwargs["worker_extension_cls"] = "roll.third_party.vllm.worker.WorkerBase"

    # https://github.com/vllm-project/vllm/pull/14189/files
    # TODO do not override other options in PYTORCH_CUDA_ALLOC_CONF
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
    # torch.cuda may already init, explicitly disable expandable_segments
    # here (only matters when VLLM_USE_RAY_SPMD_WORKER=0)
    current_platform.memory._set_allocator_settings("expandable_segments:False")

    os.environ["VLLM_CACHE_ROOT"] = os.path.join(get_default_cache_root(), "vllm", os.environ.get("WORKER_NAME", ""))

    os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
        pathlib.Path.home().as_posix(), ".cache", os.environ.get("WORKER_NAME", "")
    )

    # Default fork method is not compatible with Roll.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if Version(torch.__version__) >= Version("2.8.0"):
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        # os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN" # for 280 rollout pipeline 乱码

    engine_args = AsyncEngineArgs(**kwargs)
    # VLLM_USE_V1 may be modified inside create_engine_config
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)

    fp8.update_quant_config(vllm_config)

    # change parallel_config.placement_group for CustomRayDistributedExecutor
    parallel_config = vllm_config.parallel_config
    assert len(resource_placement_groups) == parallel_config.world_size
    parallel_config.placement_group = resource_placement_groups

    if not hasattr(envs, "VLLM_USE_V1") or envs.VLLM_USE_V1:
        from vllm.v1.executor.abstract import Executor

        from roll.third_party.vllm.async_llm import CustomAsyncLLM

        executor_class = Executor.get_class(vllm_config)
        if parallel_config.distributed_executor_backend == "ray":
            assert ray_executor_class_v1 is not None, (
                f"ROLL does not support using ray distributed executor with vllm version {vllm.__version__}"
            )
            executor_class = ray_executor_class_v1

        logger.info(f"Using executor_class: {executor_class}")
        logger.info(f"Using {parallel_config.worker_cls=} {parallel_config.worker_extension_cls=}")
        async_llm = CustomAsyncLLM(
            vllm_config=vllm_config,
            executor_class=executor_class,
            start_engine_loop=True,
            log_requests=engine_args.enable_log_requests
            if hasattr(engine_args, "enable_log_requests")
            else not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            usage_context=UsageContext.ENGINE_CONTEXT,
        )
    else:
        from vllm.v1.engine.async_llm import AsyncLLM

        from roll.third_party.vllm.async_llm_engine import CustomAsyncLLMEngine

        assert not issubclass(CustomAsyncLLMEngine, AsyncLLM)

        executor_class = CustomAsyncLLMEngine._get_executor_cls(vllm_config)
        if parallel_config.distributed_executor_backend == "ray":
            assert ray_executor_class_v0 is not None, (
                f"ROLL does not support using ray distributed executor with vllm version {vllm.__version__}"
            )
            executor_class = ray_executor_class_v0

        logger.info(f"Using executor_class: {executor_class}")
        logger.info(f"Using worker cls: {parallel_config.worker_cls}")
        async_llm = CustomAsyncLLMEngine(
            vllm_config=vllm_config,
            executor_class=executor_class,
            start_engine_loop=True,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            usage_context=UsageContext.ENGINE_CONTEXT,
            stat_loggers=None,
        )

    setattr(async_llm, "roll_sleep_mode_enabled", kwargs["enable_sleep_mode"])
    await async_llm.custom_init_worker()

    return async_llm


__all__ = ["create_async_llm"]
