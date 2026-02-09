import asyncio
import os
import gc
import torch
from contextlib import contextmanager

from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid, GiB_bytes
from vllm.outputs import RequestOutput

from roll.platforms import current_platform


# helper function to generate batch of requests with the same sampling_params
async def generate_batch(model, prompts, sampling_params):
    assert isinstance(sampling_params, SamplingParams)
    async def generate(prompt):
        request_id = random_uuid()
        result_generator = model.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)
        output = None
        async for request_output in result_generator:
            output = request_output
        assert output is not None
        return output
    return await asyncio.gather(*[generate(prompt) for prompt in prompts])

def print_request_output(vllm_output: RequestOutput):
    def _print(output):
        print(f"[request] {output.request_id}")
        print(f"[prompt] {repr(output.prompt)}")
        for text in output.outputs:
            print(f"[text] {repr(text.text)}")

    if vllm_output is None:
        print(f"[output is None]")
    elif isinstance(vllm_output, list):
        for output in vllm_output:
            _print(output)
    else:
        assert isinstance(vllm_output, RequestOutput)
        _print(vllm_output)


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

prompts = [
    "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
    "根据关键词描述生成女装/女士精品行业连衣裙品类的发在淘宝的小红书风格的推送配文，包括标题和内容。关键词：pe。要求:1. 推送标题要体现关键词和品类特点，语言通顺，有吸引力，约10个字；2. 推送内容要语言通顺，突出关键词和品类特点，对目标受众有吸引力，长度约30字。标题:",
    "100.25和90.75谁更大？",
]

chat_prompts = [chat_format(prompt) for prompt in prompts]


def print_current_mem_usage(tag):
    current_platform.empty_cache()
    gc.collect()
    free_bytes, total = current_platform.mem_get_info()
    print(f"[mem_usage] {tag} | current used: {(total - free_bytes) / GiB_bytes}")

@contextmanager
def mem_usage(mem_profile=False):
    free_bytes, total = torch.cuda.mem_get_info()
    used_bytes_before = total - free_bytes
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
    if mem_profile:
        torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT, stacks="python")
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        dump_file = ""
        if mem_profile:
            dump_file = f"/tmp/{random_uuid()}.pickle"
            os.makedirs(os.path.dirname(dump_file), exist_ok=True)
            torch.cuda.memory._dump_snapshot(dump_file)
            # print(f"{torch.cuda.memory._snapshot()}")
            torch.cuda.memory._record_memory_history(enabled=None)
        free_bytes, total = torch.cuda.mem_get_info()
        used_bytes_after = total - free_bytes
        print(
            f"[mem_usage] before {used_bytes_before / GiB_bytes} after {used_bytes_after / GiB_bytes}, dump to file {dump_file}"
        )
