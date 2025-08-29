import gc
import os
import uuid
from contextlib import contextmanager

import ray
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM
from vllm import SamplingParams
from vllm.utils import GiB_bytes

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM
from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.checkpoint_manager import download_model

USE_CUSTOME_MODEL_UPDATE = True

def print_current_mem_usage(tag):
    torch.cuda.empty_cache()
    gc.collect()
    free_bytes, total = torch.cuda.mem_get_info()
    print(f"[mem_usage] {tag} | current used: {(total - free_bytes) / GiB_bytes}")

def custom_wakeup(self):
    print_current_mem_usage("before_wakeup")

    self.wake_up(["weights"])
    print_current_mem_usage("after_wakeup")

WorkerHelper.custom_wakeup = custom_wakeup

def test_fp8_mem():
    ray.init()
    resource_manager = ResourceManager(1, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = download_model(model_path)
    model = LLM(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        load_format="auto",
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        enable_sleep_mode=True,
        enforce_eager=False,
        quantization="fp8",
    )
    model.collective_rpc(method="offload_states", args=(1,))
    model.collective_rpc(method="custom_wakeup")


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
            dump_file = f"/tmp/{uuid.uuid4()}.pickle"
            os.makedirs(os.path.dirname(dump_file), exist_ok=True)
            torch.cuda.memory._dump_snapshot(dump_file)
            # print(f"{torch.cuda.memory._snapshot()}")
            torch.cuda.memory._record_memory_history(enabled=None)
        free_bytes, total = torch.cuda.mem_get_info()
        used_bytes_after = total - free_bytes
        print(
            f"[mem_usage] before {used_bytes_before / GiB_bytes} after {used_bytes_after / GiB_bytes}, dump to file {dump_file}"
        )

def custom_load_model(self, model_path, zero=False):
    train_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    for param_name, param in tqdm(iterable=train_model.named_parameters(), total=len(list(train_model.named_parameters()))):
        if zero:
            param = param.data.clone().cuda().zero_()
        else:
            param = param.data.clone().cuda()
        self.load_weights([(param_name, param)])

WorkerHelper.custom_load_model = custom_load_model

def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def test_fp8():
    os.environ["VLLM_USE_DEEP_GEMM"] = "1"

    ray.init()
    resource_manager = ResourceManager(2, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0,1])

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    model_path = "Qwen/Qwen3-32B"
    model_path = download_model(model_path)
    model = LLM(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        load_format="auto",
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=2,
        enable_sleep_mode=True,
        enforce_eager=False,
        quantization="fp8",
    )

    prompts = [
        "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
    ]
    chat_prompts = []
    for prompt in prompts:
        chat_prompts.append(chat_format(prompt))
    sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)

    vllm_outputs = model.generate(prompts=chat_prompts, sampling_params=sampling_params)
    print(vllm_outputs)

    model.offload_states()
    model.collective_rpc("custom_load_model", args=(model_path, True))
    with mem_usage():
        model.load_states()

    vllm_outputs = model.generate(prompts=chat_prompts, sampling_params=sampling_params)
    print(vllm_outputs)

    model.offload_states()
    model.collective_rpc("custom_load_model", args=(model_path, False))
    with mem_usage():
        model.load_states()

    vllm_outputs = model.generate(prompts=chat_prompts, sampling_params=sampling_params)
    print(vllm_outputs)

if __name__ == "__main__":
    test_fp8()
