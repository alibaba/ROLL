import ray
import torch
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def test_sampling_n(model):
    prompts = [[1, 2, 3]]
    TOTAL = 3
    sampling_params = SamplingParams(temperature=0.1, top_p=0.99, top_k=100, max_tokens=512, n=TOTAL)
    model.add_requests(request_ids=[12345], sampling_params=sampling_params, prompt_token_ids=prompts, multi_modal_data=None)

    vllm_outputs = []
    count = 0
    while count < TOTAL:
        assert model.llm_engine.has_unfinished_requests()
        vllm_outputs = model.fetch_output()
        if len(vllm_outputs) > 0:
            assert len(vllm_outputs) == 1
            count += len(vllm_outputs[0].outputs)
    assert not model.llm_engine.has_unfinished_requests()


def test_abort_request(model):
    prompts = [[1, 2, 3]]
    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=8192,
        max_tokens=8192,
    )
    request_id = "12345"
    model.add_requests(request_ids=[request_id], sampling_params=sampling_params, prompt_token_ids=prompts, multi_modal_data=None)

    assert model.llm_engine.has_unfinished_requests()
    model.abort_request(request_id)
    vllm_outputs = model.fetch_output()
    assert len(vllm_outputs) == 0
    assert not model.llm_engine.has_unfinished_requests()


if __name__ == "__main__":
    ray.init()
    resource_manager = ResourceManager(1, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model = LLM(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
        distributed_executor_backend="ray",
        disable_custom_all_reduce=True,
        enable_sleep_mode=True,
    )
    test_sampling_n(model)
    test_abort_request(model)
