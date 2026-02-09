import os
import asyncio
import ray
from tqdm import tqdm

from roll.platforms import current_platform

from transformers import AutoModelForCausalLM
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import create_async_llm
from roll.third_party.vllm.worker import WorkerV1
from roll.utils.checkpoint_manager import download_model
from utils import generate_batch, chat_format, print_current_mem_usage, mem_usage, print_request_output


class Fp8Worker(WorkerV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def custom_wakeup(self):
        print_current_mem_usage("before_wakeup")
        self.wake_up(["weights"])
        print_current_mem_usage("after_wakeup")

    def custom_load_model(self, model_path, zero=False):
        train_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        for param_name, param in tqdm(iterable=train_model.named_parameters(), total=len(list(train_model.named_parameters()))):
            if zero:
                param = param.data.clone().cuda().zero_()
            else:
                param = param.data.clone().cuda()
            self.load_weights([(param_name, param)])

async def test_fp8_mem_usage():
    os.environ["VLLM_USE_V1"] = "1"

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = download_model(model_path)
    model = await create_async_llm(
        resource_placement_groups=[[0]],
        model=model_path,
        load_format="auto",
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        enable_sleep_mode=True,
        enforce_eager=False,
        quantization="fp8",
        worker_extension_cls="tests.third_party.vllm.test_fp8.Fp8Worker"
    )
    await model.offload_states(level=1)
    await model.engine_core.collective_rpc_async("custom_wakeup")

async def test_fp8():
    os.environ["VLLM_USE_DEEP_GEMM"] = "1"

    ray.init()
    resource_manager = ResourceManager(2, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0,1])

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    model_path = "Qwen/Qwen3-32B"
    model_path = download_model(model_path)
    model = await create_async_llm(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        load_format="auto",
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=2,
        disable_custom_all_reduce=True,
        enable_sleep_mode=True,
        enforce_eager=False,
        quantization="fp8",
        worker_extension_cls="tests.third_party.vllm.worker.Fp8Worker"
    )

    prompts = ["类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案"]
    chat_prompts = [chat_format(prompt) for prompt in prompts]

    sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)

    vllm_outputs = await generate_batch(model, chat_prompts, sampling_params)
    print_request_output(vllm_outputs)

    await model.offload_states()
    await model.engine_core.collective_rpc_async("custom_load_model", args=(model_path, True))
    with mem_usage():
        await model.load_states()

    vllm_outputs = await generate_batch(model, chat_prompts, sampling_params)
    print_request_output(vllm_outputs)

    await model.offload_states()
    await model.engine_core.collective_rpc_async("custom_load_model", args=(model_path, False))
    with mem_usage():
        await model.load_states()

    vllm_outputs = await generate_batch(model, chat_prompts, sampling_params)
    print_request_output(vllm_outputs)

async def main():
    await test_fp8_mem_usage()
    await test_fp8()

if __name__ == "__main__":
    asyncio.run(main())
