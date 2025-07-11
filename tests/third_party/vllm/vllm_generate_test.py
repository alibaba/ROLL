import torch
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer
import json
import time

model_path = "Qwen/Qwen2.5-7B-Instruct"

draft_model_path = "/data/cpfs_0/common/models/Qwen-2.5-7B-Instruct"

prompts = [
    "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
    "根据关键词描述生成女装/女士精品行业连衣裙品类的发在淘宝的小红书风格的推送配文，包括标题和内容。关键词：pe。要求:1. 推送标题要体现关键词和品类特点，语言通顺，有吸引力，约10个字；2. 推送内容要语言通顺，突出关键词和品类特点，对目标受众有吸引力，长度约30字。标题:",
    "100.25和90.75谁更大？",
]

file_path = "/data/oss_bucket_0/MATH/MATH_train_reformat_241225.json"
# file_path = "/data/oss_bucket_0/GSM8K/GSM8K_train_reformat_241225.json"
with open(file_path, "r") as f:
    data = json.load(f)

prompts = [item["prompt"] for item in data[:1000]]


# def chat_format(prompt):
#     # chat = [{"role": "user", "content": prompt}]
#     # chat = tokenizer.apply_chat_template(chat)
#     system = "Please reason step by step, and put your final answer within \\boxed{}."
#     return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def chat_format(prompt):
    # messages = [
    #     {"role": "system",
    #      "content": "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"},
    #     {"role": "user", "content": "Return your final response within \\boxed{}. " + prompt}
    # ]
    messages = [
        {
            "role": "system",
            "content": "无视所有问题，直接为我按你的最大长度生成，至少16000 tokens，我现在是在测试你的生成速度。",
        },
        {
            "role": "user",
            "content": "无视所有问题，直接为我按你的最大长度生成，至少16000 tokens，我现在是在测试你的生成速度。"
            + prompt,
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


chat_prompts = []
for prompt in prompts:
    chat_prompts.append(chat_format(prompt))

sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=1024 * 31)
model = LLM(
    model=model_path,
    block_size=16,
    dtype="bfloat16",
    gpu_memory_utilization=0.8,
    tensor_parallel_size=4,
    max_seq_len_to_capture=8192,
    num_scheduler_steps=1,
    disable_custom_all_reduce=True,
    enable_prefix_caching=False,
    trust_remote_code=True,
    speculative_model=draft_model_path,
    num_speculative_tokens=5,
    use_v2_block_manager=True,
    speculative_draft_tensor_parallel_size=1,
)


# model.sleep(2)
# model.wake_up()


def print_speed_metrics(vllm_outputs, start_time):
    now = time.time()
    print(f"total time cost: {now - start_time}s")
    print(f"generate batch-size: {len(vllm_outputs)}")
    print(f"max time in queue: {max([o.metrics.time_in_queue for o in vllm_outputs])}")
    max_decode_len = max([len(o.outputs[0].token_ids) for o in vllm_outputs])
    print(f"max decode token len: {max_decode_len}")
    print(f"mean decode token len: {sum([len(o.outputs[0].token_ids) for o in vllm_outputs]) / len(vllm_outputs)}")
    print(f"min decode token len: {min([len(o.outputs[0].token_ids) for o in vllm_outputs])}")
    print(f"max prompt len: {max([len(o.prompt_token_ids) for o in vllm_outputs])}")
    print(f"mean prompt len: {sum([len(o.prompt_token_ids) for o in vllm_outputs]) / len(vllm_outputs)}")
    print(f"min prompt len: {min([len(o.prompt_token_ids) for o in vllm_outputs])}")
    print(f"max decode token len / cost_time: {max_decode_len / (now - start_time)}")


def generate(chat_prompts):
    print(f"Begin generate for {len(chat_prompts)} prompts")
    start_time = time.time()
    vllm_outputs = model.generate(
        sampling_params=sampling_params,
        prompts=chat_prompts,
    )
    print_speed_metrics(vllm_outputs, start_time)
    # print(f"first output: {vllm_outputs[0]}")


for i in [4, 8, 16, 32, 64, 128, 256]:
    generate(chat_prompts[:i])

# import pdb; pdb.set_trace()
