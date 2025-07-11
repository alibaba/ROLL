import ray
import asyncio
from typing import Optional
from transformers import AutoTokenizer
from vllm import SamplingParams, RequestOutput
from vllm.utils import random_uuid
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import AsyncLLM


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def main():
    model_path = "Qwen/Qwen2.5-7B-Instruct"

    prompts = [
        "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
        "根据关键词描述生成女装/女士精品行业连衣裙品类的发在淘宝的小红书风格的推送配文，包括标题和内容。关键词：pe。要求:1. 推送标题要体现关键词和品类特点，语言通顺，有吸引力，约10个字；2. 推送内容要语言通顺，突出关键词和品类特点，对目标受众有吸引力，长度约30字。标题:",
        "100.25和90.75谁更大？",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    chat_prompts = []
    for prompt in prompts:
        chat_prompts.append(chat_format(prompt))

    ray.init()
    resource_manager = ResourceManager()
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0, 1, 2, 3])
    sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)

    model = AsyncLLM(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=4,
        trust_remote_code=True,
        distributed_executor_backend="ray",
        disable_custom_all_reduce=True,
        enable_sleep_mode=True,
    )

    print(f"start offload states")
    model.offload_states()
    print(f"start reload model")
    model.collective_rpc("reload_model")
    print(f"start load states")
    model.load_states()

    print(f"start generate")

    async def generate(prompt, sampling_params):
        request_id = random_uuid()
        result_generator = model.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)
        vllm_output: Optional[RequestOutput] = None
        async for request_output in result_generator:
            vllm_output = request_output
        assert vllm_output is not None
        return vllm_output

    loop = asyncio.get_event_loop()
    vllm_outputs = loop.run_until_complete(
        asyncio.gather(*[generate(prompt=req, sampling_params=sampling_params) for req in prompts])
    )

    print(vllm_outputs)


if __name__ == "__main__":
    main()
