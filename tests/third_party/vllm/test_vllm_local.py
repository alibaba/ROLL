import ray
import torch
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM
from roll.utils.checkpoint_manager import download_model


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def test_vllm_with_load_offload():
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = download_model(model_path)

    prompts = [
        "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
        "根据关键词描述生成女装/女士精品行业连衣裙品类的发在淘宝的小红书风格的推送配文，包括标题和内容。关键词：pe。要求:1. 推送标题要体现关键词和品类特点，语言通顺，有吸引力，约10个字；2. 推送内容要语言通顺，突出关键词和品类特点，对目标受众有吸引力，长度约30字。标题:",
        "100.25和90.75谁更大？",
    ]

    chat_prompts = []
    for prompt in prompts:
        chat_prompts.append(chat_format(prompt))

    ray.init()
    resource_manager = ResourceManager(8, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0, 1, 2, 3])
    sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)

    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT, stacks="python")

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

    vllm_outputs = model.generate(
        sampling_params=sampling_params,
        prompts=chat_prompts,
    )
    print(vllm_outputs)

    model.offload_states(1)
    model.load_states()
    vllm_outputs = model.generate(
        sampling_params=sampling_params,
        prompts=chat_prompts,
    )
    print(vllm_outputs)

    model.offload_states(2)
    model.load_states()
    vllm_outputs = model.generate(
        sampling_params=sampling_params,
        prompts=chat_prompts,
    )
    print(vllm_outputs)


if __name__ == "__main__":
    test_vllm_with_load_offload()
