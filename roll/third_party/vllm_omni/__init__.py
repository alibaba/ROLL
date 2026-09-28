import os
from typing import Any, Dict, List

from roll.platforms import current_platform
from roll.utils.logging import get_logger

logger = get_logger()

EXPECTED_VLLM_OMNI_VERSION = "0.18.0"
_CUSTOM_OUTPUT_PATCH_SENTINEL = "_roll_custom_output_patch_installed"


def _get_vllm_omni_module():
    import vllm_omni  # type: ignore
    return vllm_omni


def _install_vllm_omni_custom_output_patch() -> None:
    module = _get_vllm_omni_module()
    if getattr(module, _CUSTOM_OUTPUT_PATCH_SENTINEL, False):
        return

    from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
    from vllm_omni.outputs import OmniRequestOutput

    async def patched_generate_batch(
        self,
        prompts,
        sampling_params,
        request_id,
        lora_request=None,
    ):
        if not prompts:
            return OmniRequestOutput(request_id=request_id, images=[], final_output_type="image")

        if sampling_params.guidance_scale:
            sampling_params.guidance_scale_provided = True

        if lora_request is not None:
            sampling_params.lora_request = lora_request

        from vllm_omni.diffusion.request import OmniDiffusionRequest

        request = OmniDiffusionRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            request_ids=[request_id] * len(prompts),
        )

        import asyncio

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                self._executor,
                self.engine.step,
                request,
            )
        except Exception as e:
            logger.error("Batch generation failed for request %s: %s", request_id, e)
            raise RuntimeError(f"Diffusion batch generation failed: {e}") from e

        all_images = []
        merged_custom_output: dict[str, Any] = {}
        merged_stage_durations: dict[str, float] = {}
        peak_memory_mb = 0.0
        first_result = results[0] if results else None

        for result in results:
            all_images.extend(getattr(result, "images", []) or [])
            merged_custom_output.update(getattr(result, "_custom_output", {}) or {})
            merged_stage_durations.update(getattr(result, "stage_durations", {}) or {})
            peak_memory_mb = max(peak_memory_mb, float(getattr(result, "peak_memory_mb", 0.0) or 0.0))

        if len(results) == 1 and first_result is not None:
            return OmniRequestOutput(
                request_id=request_id,
                images=all_images,
                final_output_type="image",
                finished=True,
                request_output=first_result,
                stage_durations=merged_stage_durations,
                peak_memory_mb=peak_memory_mb,
            )

        return OmniRequestOutput(
            request_id=request_id,
            images=all_images,
            final_output_type="image",
            finished=True,
            _custom_output=merged_custom_output,
            stage_durations=merged_stage_durations,
            peak_memory_mb=peak_memory_mb,
        )

    AsyncOmniDiffusion._generate_batch = patched_generate_batch
    setattr(module, _CUSTOM_OUTPUT_PATCH_SENTINEL, True)
    logger.info("ROLL vllm_omni custom_output patch installed")


def assert_vllm_omni_version(expected_version: str = EXPECTED_VLLM_OMNI_VERSION) -> str:
    module = _get_vllm_omni_module()
    version = getattr(module, "__version__", "unknown")
    logger.info(f"vllm_omni version check disabled, current version: {version}")
    return version


def _configure_cuda_allocator_for_vllm_omni() -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
    current_platform.memory._set_allocator_settings("expandable_segments:False")


async def create_async_llm_omni(resource_placement_groups: List[Dict], **kwargs: Any):
    """Factory wrapper with a stable contract for ROLL strategies."""
    from roll.third_party.vllm_omni.async_omni import CustomAsyncOmni

    engine_factory = kwargs.pop("engine_factory", None)
    if engine_factory is not None:
        engine = engine_factory(resource_placement_groups=resource_placement_groups, **kwargs)
        if hasattr(engine, "__await__"):
            engine = await engine
        return engine

    module = _get_vllm_omni_module()
    async_omni_cls = getattr(module, "AsyncOmni", None)
    if async_omni_cls is None:
        raise RuntimeError(
            "ROLL vllm_omni adapter requires `vllm_omni.AsyncOmni`. "
            f"Available exports: {sorted(name for name in dir(module) if not name.startswith('_'))}"
        )

    _configure_cuda_allocator_for_vllm_omni()
    _install_vllm_omni_custom_output_patch()
    kwargs["enable_sleep_mode"] = True
    kwargs.setdefault(
        "worker_extension_cls",
        "roll.third_party.vllm_omni.worker.VllmOmniColocateWorkerExtension",
    )
    logger.info(
        "create_async_llm_omni: module_version=%s constructor=%s worker_extension_cls=%s",
        getattr(module, "__version__", "unknown"),
        async_omni_cls.__name__,
        kwargs.get("worker_extension_cls"),
    )
    async_omni = async_omni_cls(**kwargs)
    if hasattr(async_omni, "__await__"):
        async_omni = await async_omni
    return CustomAsyncOmni(async_omni)


__all__ = [
    "EXPECTED_VLLM_OMNI_VERSION",
    "assert_vllm_omni_version",
    "create_async_llm_omni",
]
