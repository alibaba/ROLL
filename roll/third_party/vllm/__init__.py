import os
import pathlib
from typing import Dict, List

import dataclasses
import torch
import vllm
from packaging.version import Version
from vllm import envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.envs import get_default_cache_root
from vllm.usage.usage_lib import UsageContext

from roll.platforms import current_platform
from roll.third_party.vllm.compat import apply_default_attention_config, module_has_attributes
from roll.third_party.vllm.frozen_ngram_sleep import configure_frozen_ngram_loading
import roll.third_party.vllm.fp8 as fp8
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger
from roll.utils.qwen38_gdn import configure_qwen38_gdn_backend


logger = get_logger()
vllm_version = Version(vllm.__version__)
legacy_ray_executor_package = None
dev_build_ray_executor = False

if vllm_version.release[:2] == (0, 17):
    # vLLM 0.17.x initializes multi-node queues before distributed groups.
    # Fixed upstream by https://github.com/vllm-project/vllm/pull/35892.
    # The fix is included in the official vLLM 0.18.0 release.
    from roll.third_party.vllm.vllm_0_17_0 import patch_multiproc_executor_init_order

    patch_multiproc_executor_init_order()

if Version("0.11.0") <= vllm_version < Version("0.11.1"):
    legacy_ray_executor_package = "roll.third_party.vllm.vllm_0_11_0"
elif Version("0.15") <= vllm_version:
    if Version("0.16").release <= vllm_version.release:
        import roll.third_party.vllm.patch_transformers # apply patch
elif module_has_attributes(
    "vllm.v1.executor.ray_executor",
    ("RayDistributedExecutor", "RayWorkerMetaData"),
):
    # Development builds report a pre-release version but expose the modern
    # ray executor API; detect the capability rather than the version string.
    dev_build_ray_executor = True
elif vllm_version < Version("0.11.0"):
    logger.warning(f"ROLL does not support vLLM version {vllm.__version__}.")
else:
    logger.warning(f"ROLL is not tested on vllm version {vllm.__version__}, something strange may happen!!!")

ray_executor_class_v0 = None
ray_executor_class_v1 = None
if legacy_ray_executor_package is not None:
    ray_executor_class_v0 = safe_import_class(
        f"{legacy_ray_executor_package}.ray_distributed_executor.CustomRayDistributedExecutor"
    )
    ray_executor_class_v1 = safe_import_class(
        f"{legacy_ray_executor_package}.v1.ray_distributed_executor.CustomRayDistributedExecutor"
    )

if dev_build_ray_executor:
    ray_executor_class_v1 = safe_import_class(
        "roll.third_party.vllm.ray_distributed_executor.CustomRayDistributedExecutor"
    )

logger.info(f"Using vllm version {vllm.__version__}")

# SP-MoE shared expert fix for Qwen3Next/Qwen3.5-MoE (vllm 0.21.0 upstream fix; self-guards on class existence)
# TODO: can be removed when vllm >= 0.21.0
try:
    from roll.third_party.vllm import sp_moe_patcher

    sp_moe_patcher.apply_sp_moe_shared_expert_fix()
except Exception:
    logger.exception("SP-MoE shared expert fix failed to install")


async def create_async_llm(resource_placement_groups: List[Dict], headless: bool = False, **kwargs):
    engine_arg_names = {field.name for field in dataclasses.fields(AsyncEngineArgs)}
    configure_qwen38_gdn_backend(kwargs, kwargs.get("model"))
    executor_backend = kwargs.get("distributed_executor_backend")
    if executor_backend == "ray" and vllm_version >= Version("0.11.1"):
        raise ValueError(
            "ROLL only keeps the Ray executor for vLLM versions earlier than 0.11.1."
        )
    if executor_backend not in (None, "mp", "ray"):
        raise ValueError(f"Unsupported distributed_executor_backend={executor_backend!r}.")
    data_parallel_backend = kwargs.get("data_parallel_backend")
    if data_parallel_backend not in (None, "mp"):
        raise ValueError(
            "ROLL vLLM only supports data_parallel_backend='mp'; "
            f"got {data_parallel_backend!r}."
        )
    kwargs["distributed_executor_backend"] = executor_backend or "mp"
    if "data_parallel_backend" in engine_arg_names:
        kwargs["data_parallel_backend"] = "mp"
    else:
        kwargs.pop("data_parallel_backend", None)
    kwargs["enable_sleep_mode"] = True
    apply_default_attention_config(
        kwargs, supports_attention_config="attention_config" in engine_arg_names
    )  # vllm<=0.12.0 does not have attention_config; explicit None opts out
    
    if "moe_backend" not in kwargs and "moe_backend" in engine_arg_names:
        # vLLM0.20.0, remove this while >= 0.29: https://github.com/vllm-project/vllm/issues/45447
        kwargs["moe_backend"] = "triton"
    
    # FlashInfer GDN prefill can hang on SM90/L20Z under concurrent
    # multimodal rollout and surface as an MP sample_tokens RPC timeout.
    # vLLM issue #38916 confirms the Triton backend avoids this failure
    if "gdn_prefill_backend" not in kwargs and "gdn_prefill_backend" in engine_arg_names:
        kwargs["gdn_prefill_backend"] = "triton"

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

    # Multi-node MultiprocExecutor support starts in vLLM 0.11.1.
    # https://github.com/vllm-project/vllm/pull/23691
    if headless and Version(vllm.__version__) < Version("0.11.1"):
        raise RuntimeError("Multi-node vLLM mp execution requires vLLM >= 0.11.1.")

    engine_args = AsyncEngineArgs(**kwargs)
    # VLLM_USE_V1 may be modified inside create_engine_config
    if headless:
        vllm_config = engine_args.create_engine_config(
            UsageContext.ENGINE_CONTEXT, headless=True
        )
    else:
        vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    configure_frozen_ngram_loading(vllm_config.model_config.hf_config, vllm_config.load_config)

    fp8.update_quant_config(config=kwargs, vllm_config=vllm_config)

    parallel_config = vllm_config.parallel_config
    if parallel_config.distributed_executor_backend == "ray":
        if len(resource_placement_groups) != parallel_config.world_size:
            raise ValueError(
                "The legacy vLLM Ray executor requires one placement per model-parallel rank."
            )
        parallel_config.placement_group = resource_placement_groups

    if headless:
        if parallel_config.node_rank_within_dp == 0:
            raise ValueError("A headless vLLM mp worker must run outside DP-local node rank 0.")
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        executor.start_worker_monitor(inline=True)
        return None

    if not hasattr(envs, "VLLM_USE_V1") or envs.VLLM_USE_V1:
        from vllm.v1.executor.abstract import Executor

        from roll.third_party.vllm.async_llm import CustomAsyncLLM

        executor_class = Executor.get_class(vllm_config)
        if parallel_config.distributed_executor_backend == "ray":
            if ray_executor_class_v1 is None:
                raise RuntimeError(
                    f"ROLL has no legacy Ray executor for vLLM {vllm.__version__}."
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
            if ray_executor_class_v0 is None:
                raise RuntimeError(
                    f"ROLL has no legacy Ray executor for vLLM {vllm.__version__}."
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

    await async_llm.custom_init_worker()

    return async_llm


__all__ = ["create_async_llm"]
