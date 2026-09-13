import asyncio
import gc
import importlib.util
import inspect
import os
import sys

# The single-device smoke uses TP=1; force this before vLLM-Ascend imports
# can cache FlashComm settings from the environment.
os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM"] = "0"

import pytest

from roll.platforms import current_platform


def _require_module(module_name: str) -> bool:
    """Check that *module_name* is importable.

    On CPU environments the test is skipped when the module is missing.
    On NPU environments the module is expected to be present.
    """
    try:
        module_spec = importlib.util.find_spec(module_name)
    except ValueError:
        # Python 3.11+ raises ValueError when a module that is already imported
        # has ``__spec__`` set to ``None`` (an edge case in certain packaging).
        # Check ``sys.modules`` as a fallback.
        module_spec = None

    available = module_spec is not None or module_name in sys.modules
    if not available and not current_platform.is_npu():
        pytest.skip(f"{module_name} is not installed in this environment.")
    assert available, f"{module_name} must be installed for NPU vLLM tests."
    return available


def test_vllm_imports_available():
    _require_module("vllm")
    if current_platform.is_npu():
        _require_module("vllm_ascend")


def test_vllm_npu_worker_class_resolves():
    if not current_platform.is_npu():
        pytest.skip("NPU worker resolution only applies on Ascend NPU.")

    worker_cls = current_platform.get_vllm_worker_class()
    assert worker_cls is not None
    assert worker_cls.__name__.endswith("Worker")


def test_roll_vllm_ray_executor_resolves():
    if not current_platform.is_npu():
        pytest.skip("ROLL vLLM Ray executor resolution only applies on Ascend NPU.")

    import roll.third_party.vllm as roll_vllm
    import vllm

    assert roll_vllm.ray_executor_class_v1 is not None, (
        f"ROLL must resolve a vLLM V1 Ray executor for NPU CI; vllm={vllm.__version__}"
    )


async def _shutdown_async_llm(model):
    for method_name in ("shutdown", "close"):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return


async def _run_with_npu_vllm_smoke_model(callback, **model_kwargs):
    import ray

    from roll.distributed.scheduler.initialize import init
    from roll.distributed.scheduler.resource_manager import ResourceManager
    from roll.third_party.vllm import create_async_llm
    from roll.utils import checkpoint_manager

    init()

    model = None
    resource_manager = None
    try:
        model_name_or_path = os.environ.get("ROLL_NPU_VLLM_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        model_path = checkpoint_manager.download_model(model_name_or_path)

        resource_manager = ResourceManager(num_gpus_per_node=1, num_nodes=1)
        placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])

        kwargs = dict(
            dtype="bfloat16",
            gpu_memory_utilization=0.35,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=1,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            disable_custom_all_reduce=True,
            enforce_eager=True,
            trust_remote_code=True,
        )
        kwargs.update(model_kwargs)

        model = await create_async_llm(
            resource_placement_groups=placement_groups[0],
            model=model_path,
            **kwargs,
        )

        return await callback(model)
    finally:
        if model is not None:
            try:
                await _shutdown_async_llm(model)
            except Exception as e:
                print(f"Failed to shut down vLLM smoke model cleanly: {e}")
        if resource_manager is not None:
            resource_manager.destroy_placement_group()
        if ray.is_initialized():
            ray.shutdown()
        checkpoint_manager.shared_storage = None
        gc.collect()
        empty_cache = getattr(current_platform, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


async def _run_npu_vllm_generate_smoke():
    from vllm import SamplingParams
    from vllm.utils import random_uuid

    async def generate(model):
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4, min_tokens=1)
        result_generator = model.generate(
            prompt="Write one short greeting.",
            sampling_params=sampling_params,
            request_id=random_uuid(),
        )

        output = None
        async for request_output in result_generator:
            output = request_output

        assert output is not None
        assert output.finished
        assert len(output.outputs) == 1
        assert output.outputs[0].token_ids

    await _run_with_npu_vllm_smoke_model(generate)


async def _run_npu_vllm_abort_smoke():
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.utils import random_uuid

    async def abort(model):
        request_id = random_uuid()
        sampling_params = SamplingParams(
            temperature=0.0,
            min_tokens=512,
            max_tokens=512,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        async def collect_output():
            output = None
            async for request_output in model.generate(
                prompt="Count upward and keep going.",
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                output = request_output
            return output

        task = asyncio.create_task(collect_output())
        await asyncio.sleep(float(os.environ.get("ROLL_NPU_VLLM_ABORT_DELAY", "0.2")))
        result = model.abort(request_id)
        if inspect.isawaitable(result):
            await result

        output = await asyncio.wait_for(task, timeout=120)
        assert output is not None
        assert output.finished
        assert output.outputs
        assert all(completion.finish_reason == "abort" for completion in output.outputs)

    await _run_with_npu_vllm_smoke_model(
        abort,
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )


def test_npu_vllm_generate_smoke():
    if not current_platform.is_npu():
        pytest.skip("NPU vLLM generate smoke only applies on Ascend NPU.")
    if os.environ.get("ROLL_NPU_VLLM_GENERATE_SMOKE", "1") == "0":
        pytest.skip("ROLL_NPU_VLLM_GENERATE_SMOKE=0")

    asyncio.run(_run_npu_vllm_generate_smoke())


def test_npu_vllm_abort_smoke():
    if not current_platform.is_npu():
        pytest.skip("NPU vLLM abort smoke only applies on Ascend NPU.")
    if os.environ.get("ROLL_NPU_VLLM_ABORT_SMOKE", "1") == "0":
        pytest.skip("ROLL_NPU_VLLM_ABORT_SMOKE=0")

    asyncio.run(_run_npu_vllm_abort_smoke())
