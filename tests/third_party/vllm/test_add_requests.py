import ray
import torch

from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM
from roll.utils.checkpoint_manager import download_model


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def test_sampling_n(model):
    prompts = ["类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案"]
    prompts = [chat_format(prompt) for prompt in prompts]
    tokenizer = model.get_tokenizer()
    prompts = tokenizer(prompts)["input_ids"]

    sampling_params = SamplingParams(temperature=0.1, top_p=0.99, top_k=100, max_tokens=512, n=3, output_kind=RequestOutputKind.FINAL_ONLY)
    model.add_requests(request_ids=[12345], sampling_params=sampling_params, prompt_token_ids=prompts, multi_modal_data=None, lora_requests=None)

    vllm_outputs = []
    while model.llm_engine.has_unfinished_requests():
        output = model.fetch_output()
        for request_output in output:
            if not request_output.finished:
                continue
            vllm_outputs.extend(request_output.outputs)
    assert len(vllm_outputs) == 3 * len(prompts)


def test_abort_request(model):
    prompts = ["类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案"]
    prompts = [chat_format(prompt) for prompt in prompts]
    tokenizer = model.get_tokenizer()
    prompts = tokenizer(prompts)["input_ids"]

    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=8192,
        max_tokens=8192,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    request_id = "12345"
    model.add_requests(request_ids=[request_id], sampling_params=sampling_params, prompt_token_ids=prompts, multi_modal_data=None, lora_requests=None)

    vllm_outputs = []
    assert model.llm_engine.has_unfinished_requests()
    model.abort_request(request_id)
    while model.llm_engine.has_unfinished_requests():
        output = model.fetch_output()
        for request_output in output:
            if not request_output.finished:
                continue
            vllm_outputs.extend(request_output.outputs)
    assert len(vllm_outputs) == 0


if __name__ == "__main__":
    ray.init()
    resource_manager = ResourceManager(4, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0,1,2,3])

    model_path = "Qwen/Qwen3-Next-80B-A3B-Thinking"
    model_path = download_model(model_path)
    model = LLM(
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
    test_sampling_n(model)
    test_abort_request(model)
