import ray
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM
from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.checkpoint_manager import download_model


def load_weight(self, name, dtype, shape, buffer):
    weight = torch.tensor(buffer, dtype=dtype).cuda()
    self.load_weights([(name, weight)])

WorkerHelper.load_weight = load_weight

def load_weights(self, params):
    for name, p in tqdm(list(params), desc="Updating parameter", unit="param"):
        self.collective_rpc(method="load_weight", args=(name, p.dtype, p.shape, p.tolist()))

LLM.load_weights = load_weights


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def test_sleep_level():
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    model_path = download_model(model_path)

    ray.init()
    resource_manager = ResourceManager(1, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])

    model = LLM(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        enable_sleep_mode=True,
    )

    prompts = [
    "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
    ]
    chat_prompts = []
    for prompt in prompts:
        chat_prompts.append(chat_format(prompt))
    sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)

    request_outputs = model.generate(prompts=chat_prompts, sampling_params=sampling_params)
    for request_output in request_outputs:
        for output in request_output.outputs:
            print(output.text)

    model.offload_states(2)

    model_path = download_model(model_path)
    train_model = AutoModelForCausalLM.from_pretrained(model_path)
    if False:
        for name, p in train_model.named_parameters():
            p.data.zero_()
    model.load_weights(train_model.named_parameters())

    model.load_states()

    request_outputs = model.generate(prompts=chat_prompts, sampling_params=sampling_params)
    for request_output in request_outputs:
        for output in request_output.outputs:
            print(output.text)


if __name__ == "__main__":
    test_sleep_level()
