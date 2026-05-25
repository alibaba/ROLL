import asyncio
import gc
import inspect
import os
from contextlib import contextmanager

import ray
from vllm import SamplingParams
from vllm.utils import random_uuid
from vllm.utils.mem_constants import GiB_bytes

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.platforms import current_platform
from roll.third_party.vllm import create_async_llm
from roll.utils import checkpoint_manager
from roll.utils.checkpoint_manager import download_model
from utils import (
    chat_prompts,
    generate_batch,
    print_request_output,
)


def _platform_memory_history_supported():
    memory = getattr(current_platform, "memory", None)
    return (
        memory is not None
        and hasattr(memory, "_record_memory_history")
        and hasattr(memory, "_dump_snapshot")
    )


@contextmanager
def _platform_mem_usage(mem_profile=False):
    current_platform.empty_cache()
    gc.collect()
    free_bytes, total = current_platform.mem_get_info()
    used_bytes_before = total - free_bytes
    enable_memory_history = mem_profile and _platform_memory_history_supported()
    if mem_profile and not enable_memory_history:
        print(
            f"[mem_usage] memory history is not supported on "
            f"{current_platform.device_type}, skip snapshot"
        )
    if enable_memory_history:
        current_platform.memory._record_memory_history(
            max_entries=100000,
            stacks="python",
        )
    try:
        yield
    finally:
        current_platform.empty_cache()
        gc.collect()
        dump_file = ""
        if enable_memory_history:
            dump_file = f"/tmp/{random_uuid()}.pickle"
            os.makedirs(os.path.dirname(dump_file), exist_ok=True)
            current_platform.memory._dump_snapshot(dump_file)
            current_platform.memory._record_memory_history(enabled=None)
        free_bytes, total = current_platform.mem_get_info()
        used_bytes_after = total - free_bytes
        print(
            f"[mem_usage] before {used_bytes_before / GiB_bytes} "
            f"after {used_bytes_after / GiB_bytes}, dump to file {dump_file}"
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


async def _run_vllm_with_load_offload():
    os.environ["VLLM_USE_V1"] = "1"
    old_task_queue_enable = os.environ.get("TASK_QUEUE_ENABLE")
    old_vllm_ascend_enable_nz = os.environ.get("VLLM_ASCEND_ENABLE_NZ")
    if current_platform.is_npu():
        os.environ["TASK_QUEUE_ENABLE"] = "1"
        os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"

    tensor_parallel_size = int(
        os.environ.get(
            "ROLL_VLLM_LOCAL_TP_SIZE",
            "2" if current_platform.is_npu() else "4",
        )
    )
    model_name = os.environ.get(
        "ROLL_VLLM_LOCAL_MODEL",
        "Qwen/Qwen2.5-7B-Instruct",
    )

    model = None
    resource_manager = None
    try:
        ray.init(ignore_reinit_error=True)
        resource_manager = ResourceManager(tensor_parallel_size, 1)
        placement_groups = resource_manager.allocate_placement_group(
            world_size=1,
            device_mapping=list(range(tensor_parallel_size)),
        )

        mem_profile = os.environ.get("ROLL_VLLM_LOCAL_MEM_PROFILE") == "1"
        with _platform_mem_usage(mem_profile=mem_profile):
            model_path = download_model(model_name)
            model = await create_async_llm(
                resource_placement_groups=placement_groups[0],
                model=model_path,
                load_format="auto",
                block_size=16,
                dtype="bfloat16",
                gpu_memory_utilization=0.8,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                distributed_executor_backend="ray",
                disable_custom_all_reduce=True,
                enable_sleep_mode=True,
                enforce_eager=current_platform.is_npu(),
            )

            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=0.99,
                top_k=100,
                max_tokens=64,
            )

            print(">>>>>>>>>>>>>>> test_vllm_local: base")
            vllm_outputs = await generate_batch(
                model=model,
                prompts=chat_prompts,
                sampling_params=sampling_params,
            )
            assert len(vllm_outputs) == len(chat_prompts)
            print_request_output(vllm_outputs)

            print(">>>>>>>>>>>>>>> test_vllm_local: offload states sleep_level_1")
            await model.offload_states(1)
            await model.load_states()
            vllm_outputs = await generate_batch(
                model=model,
                prompts=chat_prompts,
                sampling_params=sampling_params,
            )
            assert len(vllm_outputs) == len(chat_prompts)
            print_request_output(vllm_outputs)

            print(">>>>>>>>>>>>>>> test_vllm_local: offload states sleep_level_2")
            await model.offload_states(2)
            await model.load_states()
            vllm_outputs = await generate_batch(
                model=model,
                prompts=chat_prompts,
                sampling_params=sampling_params,
            )
            assert len(vllm_outputs) == len(chat_prompts)
            print_request_output(vllm_outputs)
    finally:
        if model is not None:
            try:
                await _shutdown_async_llm(model)
            except Exception as e:
                print(f"Failed to shut down vLLM model cleanly: {e}")
        if resource_manager is not None:
            resource_manager.destroy_placement_group()
        if ray.is_initialized():
            ray.shutdown()
        checkpoint_manager.shared_storage = None
        if old_task_queue_enable is None:
            os.environ.pop("TASK_QUEUE_ENABLE", None)
        else:
            os.environ["TASK_QUEUE_ENABLE"] = old_task_queue_enable
        if old_vllm_ascend_enable_nz is None:
            os.environ.pop("VLLM_ASCEND_ENABLE_NZ", None)
        else:
            os.environ["VLLM_ASCEND_ENABLE_NZ"] = old_vllm_ascend_enable_nz


def test_vllm_with_load_offload():
    asyncio.run(_run_vllm_with_load_offload())


if __name__ == "__main__":
    test_vllm_with_load_offload()
