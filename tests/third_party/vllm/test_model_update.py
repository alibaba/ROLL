import os
import ray
import inspect
import asyncio
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.platforms import current_platform
from roll.third_party.vllm import create_async_llm
from roll.third_party.vllm.worker import WorkerV1
from roll.utils import checkpoint_manager
from roll.utils.checkpoint_manager import download_model
try:
    from .utils import generate_batch, chat_prompts, print_request_output
except ImportError:
    from utils import generate_batch, chat_prompts, print_request_output

class ModelUpdateWorker(WorkerV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_full_model(self, model_path, zero=False):
        train_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        for param_name, param in tqdm(iterable=train_model.named_parameters(), total=len(list(train_model.named_parameters()))):
            if zero:
                param = param.data.clone().to(self.device).zero_()
            else:
                param = param.data.clone().to(self.device)
            self.load_weights([(param_name, param)])


async def _shutdown_async_llm(model):
    for method_name in ("shutdown", "close"):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return


async def _run_vllm_offload():
    os.environ["VLLM_USE_V1"] = "1"
    old_task_queue_enable = os.environ.get("TASK_QUEUE_ENABLE")
    old_vllm_ascend_enable_nz = os.environ.get("VLLM_ASCEND_ENABLE_NZ")
    if current_platform.is_npu():
        os.environ["TASK_QUEUE_ENABLE"] = "1"
        os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"
    model = None
    resource_manager = None
    try:
        ray.init(ignore_reinit_error=True)
        resource_manager = ResourceManager(2, 1)
        placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0, 1])

        model_path = "Qwen/Qwen2.5-7B-Instruct"
        model_path = download_model(model_path)
        model = await create_async_llm(
            resource_placement_groups=placement_groups[0],
            model=model_path,
            load_format="auto",
            block_size=16,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            tensor_parallel_size=2,
            distributed_executor_backend="ray",
            disable_custom_all_reduce=True,
            enable_sleep_mode=True,
            enforce_eager=current_platform.is_npu(),
            worker_extension_cls="tests.third_party.vllm.test_model_update.ModelUpdateWorker",
        )

        # test offload/onload and sleep_level
        sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)

        print(">>>>>>>>>>>>>>> test_vllm_load_offload: base")
        vllm_outputs = await generate_batch(model=model, prompts=chat_prompts, sampling_params=sampling_params)
        assert len(vllm_outputs) == len(chat_prompts)
        print_request_output(vllm_outputs)

        print(">>>>>>>>>>>>>>> test_vllm_load_offload: offload states sleep_level_1")
        await model.offload_states(1)
        await model.load_states()
        vllm_outputs = await generate_batch(model=model, prompts=chat_prompts, sampling_params=sampling_params)
        print_request_output(vllm_outputs)

        print(">>>>>>>>>>>>>>> test_vllm_load_offload: offload states sleep_level_2")
        await model.offload_states(2)
        await model.load_states()
        vllm_outputs = await generate_batch(model=model, prompts=chat_prompts, sampling_params=sampling_params)
        print_request_output(vllm_outputs)

        print(">>>>>>>>>>>>>>> test_vllm_load_offload: offload states sleep_level_2 + reload")
        await model.offload_states(2)
        await model.engine_core.collective_rpc_async("load_full_model", args=(model_path,))
        await model.load_states()
        vllm_outputs = await generate_batch(model=model, prompts=chat_prompts, sampling_params=sampling_params)
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


def test_vllm_offload():
    asyncio.run(_run_vllm_offload())


if __name__ == "__main__":
    test_vllm_offload()
