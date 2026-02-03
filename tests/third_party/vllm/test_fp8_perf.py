import os
import json
import time
import itertools

import asyncio
import ray
from vllm import SamplingParams
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import create_async_llm
from roll.utils.checkpoint_manager import download_model
from utils import generate_batch, chat_format


def print_speed_metrics(outputs, start_time):
    now = time.time()
    print(f"total time cost: {now - start_time}s")
    print(f"generate batch-size: {len(outputs)}")
    print(f"max decode token len: {max([len(o.outputs[0].token_ids) for o in outputs])}")
    print(f"mean decode token len: {sum([len(o.outputs[0].token_ids) for o in outputs]) / len(outputs)}")
    print(f"min decode token len: {min([len(o.outputs[0].token_ids) for o in outputs])}")
    print(
        f"max decode token len / cost_time {max([len(o.outputs[0].token_ids) for o in outputs]) / (now - start_time)}"
    )
    print(f"max prompt len: {max([len(o.prompt_token_ids) for o in outputs])}")
    print(f"mean prompt len: {sum([len(o.prompt_token_ids) for o in outputs]) / len(outputs)}")
    print(f"min prompt len: {min([len(o.prompt_token_ids) for o in outputs])}")

async def generate(model, prompts, sampling_params):
    print(f"Begin generate for {len(prompts)} prompts")
    start_time = time.time()
    outputs = await generate_batch(model, prompts, sampling_params)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print_speed_metrics(outputs, start_time)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # need patch vllm084 StatLogger
    # model.llm_engine.do_log_stats()

def get_sampling_param_uniform(limit, num):
    num_tokens = []
    sampling_params = []
    for num_token in range(limit, 16, -(limit // num)):
        num_tokens.append(num_token)
        sampling_param = SamplingParams(
            temperature=0.95,
            top_p=0.7,
            top_k=50,
            max_tokens=num_token,
            min_tokens=num_token,
        )
        sampling_params.append(sampling_param)
    return sampling_params, num_tokens

def get_sampling_param_max(limit, num):
    num_tokens = []
    sampling_params = []
    for i in range(16, limit, limit // num):
        num_token = limit
        num_tokens.append(num_token)
        sampling_param = SamplingParams(
            temperature=0.95,
            top_p=0.7,
            top_k=50,
            max_tokens=num_token,
            min_tokens=num_token,
        )
        sampling_params.append(sampling_param)
    return sampling_params, num_tokens

async def test_uniform(model, chat_prompts, limit, num):
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TEST UNIFORM {limit} {num}")
    sampling_params, num_tokens = get_sampling_param_uniform(limit, num)
    prompts = list(itertools.islice(itertools.cycle(chat_prompts), len(sampling_params)))
    await generate(model, prompts, sampling_params)
    await model.do_log_stats()

async def test_max(model, chat_prompts, limit, num):
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TEST MAX {limit} {num}")
    sampling_params, num_tokens = get_sampling_param_max(limit, num)
    prompts = list(itertools.islice(itertools.cycle(chat_prompts), len(sampling_params)))
    await generate(model, prompts, sampling_params)
    await model.do_log_stats()

async def main():
    os.environ["VLLM_USE_DEEP_GEMM"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    ray.init()
    resource_manager = ResourceManager(2, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0,1])

    model_path = "Qwen/Qwen3-8B"
    model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    model_path = "Qwen/Qwen3-32B"
    model_path = "/data/cpfs_0/common/models/Qwen3-8B"
    model_path = "/data/cpfs_0/common/models/Qwen3-235B-A22B"
    model_path = "/data/cpfs_0/common/models/Qwen3-32B"
    model_path = "/data/cpfs_0/common/models/Qwen3-30B-A3B"
    model_path = download_model(model_path)
    model = await create_async_llm(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        tensor_parallel_size=2,
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        load_format="auto",
        quantization="fp8",
        # hf_overrides={"quantization_config":
        #                 {
        #                     "activation_scheme": "dynamic",
        #                     "fmt": "e4m3",
        #                     "quant_method": "fp8",
        #                     "weight_block_size": [128, 128],
        #                 }
        #              },
    )

    file_path = "data/math_benchmarks.jsonl"
    data = []
    with open(file_path, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            data.append(obj)
    prompts = [item["prompt"] for item in data[:1000] if len(item["prompt"]) >= 100 and len(item["prompt"]) <=300]
    chat_prompts = []
    for prompt in prompts:
        chat_prompts.append(chat_format(prompt))

    # nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node
    #with nvtx.annotate("generate"):
    #    await test_max(model, chat_prompts, 4096, 32)

    await test_max(model, chat_prompts, 4096, 32)
    await test_max(model, chat_prompts, 4096, 16)
    await test_max(model, chat_prompts, 4096, 8)
    await test_max(model, chat_prompts, 4096, 4)
    await test_max(model, chat_prompts, 4096, 1)

    await test_uniform(model, chat_prompts, 4096, 32)
    await test_uniform(model, chat_prompts, 4096, 16)
    await test_uniform(model, chat_prompts, 4096, 8)
    await test_uniform(model, chat_prompts, 4096, 4)
    await test_uniform(model, chat_prompts, 4096, 1)

if __name__ == "__main__":
    asyncio.run(main())