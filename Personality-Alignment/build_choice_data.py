"""
change_dataset_5_version.py
生成符合格式要求的混淆选项，并将原始数据集转换成 ABCD 四选格式。
7.31version: 错误选项似乎都是一样的，需要进行特殊调整。
修改版本：支持多种干扰项生成模式
"""

import json
import random
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm

# ========================= 1. Prompt 模板 ========================= #

# 模式1：只给ground_truth，违反句式、话题、内容丰富度
DISTRACTOR_PROMPTS = {
    "style_violation": """[System]
You are a distractor generator for multiple-choice questions.
Your task: Given a TARGET sentence, generate exactly one sentence that violates the style/sentence structure of the target while keeping similar topic.
The output should be different in sentence type/structure from the TARGET (e.g., if TARGET is a question, make it a statement; if TARGET is formal, make it informal).
Do NOT explain, produce only the sentence.

[User]
TARGET: {correct_output}

[Assistant] 
""",
    "topic_violation": """[System]
You are a distractor generator for multiple-choice questions.
Your task: Given a TARGET sentence, generate exactly one sentence that changes the main topic/subject while keeping similar sentence structure.
The output should discuss a different topic but maintain similar grammatical structure as the TARGET.
Do NOT explain, produce only the sentence.

[User]
TARGET: {correct_output}

[Assistant] 
""",
    "richness_violation": """[System]
You are a distractor generator for multiple-choice questions.
Your task: Given a TARGET sentence, generate exactly one sentence that significantly differs in content richness/detail level.
If TARGET is detailed, make it very simple; if TARGET is simple, make it overly complex.
Do NOT explain, produce only the sentence.

[User]
TARGET: {correct_output}

[Assistant] 
""",
    "profile_violation": """[System]
You are a distractor generator for multiple-choice questions.
Your task: Given a TARGET sentence and a PROFILE, generate exactly one sentence that obviously violates the personality profile.
The output should contradict the personality traits, preferences, or characteristics described in the profile.
Do NOT explain, produce only the sentence.

[User]
TARGET: {correct_output}
PROFILE: {profile}

[Assistant] 
""",
    "conversation_violation": """[System]
You are a distractor generator for multiple-choice questions.
Your task: Given a TARGET sentence and CONVERSATION HISTORY, generate exactly one sentence that is inappropriate or inconsistent with the conversation context.
The output should ignore the conversation flow, context, or previous topics discussed.
Do NOT explain, produce only the sentence.

[User]
TARGET: {correct_output}
CONVERSATION: {conversation}

[Assistant] 
""",
    "both_violation": """[System]
You are a distractor generator for multiple-choice questions.
Your task: Given a TARGET sentence, PROFILE, and CONVERSATION HISTORY, generate exactly one sentence that violates both the personality profile and conversation context.
The output should contradict the personality traits AND be inappropriate for the conversation context.
Do NOT explain, produce only the sentence.

[User]
TARGET: {correct_output}
PROFILE: {profile}
CONVERSATION: {conversation}

[Assistant] 
""",
}

# ========================= 2. 读取原始数据 ========================= #
DATA_PATH = "/home/zyangdm/ROLL/Personality-Alignment/dialogue_dataset_all_v5_summarized.jsonl"
data: list[dict] = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

data = data[:100]


# ========================= 3. 加载模型 ========================= #
def load_qwen3_8b(
    model_path: str = "Qwen/Qwen3-8B",
    device_map: str = "auto",
    quantize: bool = False,
):
    """
    加载 Qwen3‑8B 模型和分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if quantize
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    return model, tokenizer


model_path = "Qwen/Qwen3-8B"
model, tokenizer = load_qwen3_8b(model_path=model_path, device_map="auto")


def extract_profile_and_history(prompt: str) -> tuple:
    """
    Extract profile and conversation history from the prompt

    Args:
        prompt: The full prompt text

    Returns:
        Tuple of (profile, conversation_history)
    """
    profile = ""
    conversation_history = ""

    # Extract profile
    if "[Profile Begin]" in prompt and "[Profile End]" in prompt:
        start = prompt.find("[Profile Begin]") + len("[Profile Begin]")
        end = prompt.find("[Profile End]")
        profile = prompt[start:end].strip()

    # Extract conversation history
    if "[Conversation History Begin]" in prompt and "[Conversation History End]" in prompt:
        start = prompt.find("[Conversation History Begin]") + len("[Conversation History Begin]")
        end = prompt.find("[Conversation History End]")
        conversation_history = prompt[start:end].strip()

    return profile, conversation_history


def get_max_length(mode: str) -> int:
    """
    根据模式返回最大长度
    """
    if mode in ["style_violation", "topic_violation", "richness_violation"]:
        return 512  # 只需要生成一个句子，较短的长度
    elif mode == "profile_violation":
        return 1024
    elif mode in ["conversation_violation", "both_violation"]:
        return 8192


def generate_distractors_by_mode(data_items, mode, batch_size=8):
    """
    根据指定模式生成干扰项

    Args:
        data_items: 数据项列表
        mode: 生成模式 ('style_violation', 'topic_violation', 'richness_violation',
              'profile_violation', 'conversation_violation', 'both_violation')
        batch_size: 批处理大小

    Returns:
        生成的干扰项列表
    """
    all_distractors = []

    # 构造所有输入
    all_inputs = []
    for item in data_items:
        correct_output = item["output"]
        profile, conversation = extract_profile_and_history(item["prompt"])

        # 根据模式构造prompt
        if mode in ["style_violation", "topic_violation", "richness_violation"]:
            prompt_text = DISTRACTOR_PROMPTS[mode].format(correct_output=correct_output)
        elif mode == "profile_violation":
            prompt_text = DISTRACTOR_PROMPTS[mode].format(correct_output=correct_output, profile=profile)
        elif mode == "conversation_violation":
            prompt_text = DISTRACTOR_PROMPTS[mode].format(correct_output=correct_output, conversation=conversation)
        elif mode == "both_violation":
            prompt_text = DISTRACTOR_PROMPTS[mode].format(
                correct_output=correct_output, profile=profile, conversation=conversation
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_inputs.append(prompt_text)

    # 分批生成
    generated_distractors = []
    for i in tqdm(range(0, len(all_inputs), batch_size), desc=f"Generating {mode} distractors"):
        batch_inputs = all_inputs[i : i + batch_size]

        # Tokenize
        input_tokens = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=get_max_length(mode),
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens,
                max_new_tokens=100,
                # do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
            )

        # Decode
        for j in range(len(batch_inputs)):
            input_length = len(input_tokens["input_ids"][j])
            generated_tokens = outputs[j][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 清理生成的文本
            generated_text = clean_generated_text(generated_text)

            # 确保生成的干扰项不是正确答案
            correct_output = data_items[i + j]["output"]
            if generated_text != correct_output and generated_text.strip():
                generated_distractors.append(generated_text)
            else:
                print(f"Warning: Generated distractor failed for mode {mode}: {generated_text}")
                generated_distractors.append(f"Failed_{mode}_distractor_{len(generated_distractors)}")

    return generated_distractors


def clean_generated_text(text: str) -> str:
    """清理生成的文本"""
    text = text.strip()
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>") :].strip()
    if text.startswith("[Assistant]"):
        text = text[text.index("[Assistant]") + len("[Assistant]") :].strip()
    if text.endswith("</s>"):
        text = text[: -len("</s>")].strip()
    if "TARGET:" in text:
        text = text.split("TARGET:")[-1].strip()
    if "OUTPUT:" in text:
        text = text.split("OUTPUT:")[-1].strip()
    return text.strip()


def generate_all_distractors_batch(data_items, batch_size=8):
    """
    为每个数据项生成所有6种模式的干扰项，然后随机选择3个作为最终干扰项
    """
    modes = [
        "style_violation",
        "topic_violation",
        "richness_violation",
        "profile_violation",
        "conversation_violation",
        "both_violation",
    ]

    # 为每种模式生成干扰项
    all_mode_distractors = {}
    for mode in modes:
        print(f"\n正在生成 {mode} 模式的干扰项...")
        mode_distractors = generate_distractors_by_mode(data_items, mode, batch_size)
        all_mode_distractors[mode] = mode_distractors

    # 为每个数据项随机选择3个不同模式的干扰项
    # final_distractors = []
    # for i in range(len(data_items)):
    #     # 随机选择3种不同的模式
    #     selected_modes = random.sample(modes, 3)
    #     item_distractors = []

    #     for mode in selected_modes:
    #         if i < len(all_mode_distractors[mode]):
    #             item_distractors.append(all_mode_distractors[mode][i])
    #         else:
    #             item_distractors.append(f"Missing_{mode}_distractor")

    #     final_distractors.append(item_distractors)

    return all_mode_distractors


def process_original_prompt(prompt: str) -> str:
    """
    处理原始提示，提取需要的部分，并重构
    """
    profile, conversation_history = extract_profile_and_history(prompt)
    # new_prompt = "You are a helpful assistant.\nNow your task is to choose the most possible output A or B based on the given profile and conversation history.\n"
    # # 重构提示
    new_prompt += f"[Profile Begin]{profile}[Profile End]\n"
    new_prompt += f"[Conversation History Begin]{conversation_history}[Conversation History End]\n"
    # new_prompt += "Now please choose the most possible output A, B, C or D\n"

    return new_prompt


def process_data_batch(data, batch_size=8):
    """
    批量处理数据:生成3个干扰项并混排成 ABCD 四选一
    """
    correct_outputs = [item["output"] for item in data]
    qids = [item["qid"] for item in data]

    print("开始批量生成多模式干扰项…")
    all_distractors = generate_all_distractors_batch(data, batch_size)

    print("构建新数据集…")
    new_data: list[dict] = []
    for i, (qid, original_prompt, correct_output, distractors) in enumerate(
        zip(qids, [item["prompt"] for item in data], correct_outputs, all_distractors)
    ):

        #     # 创建选项列表：正确答案 + 3个干扰项
        #     options = [correct_output] + distractors

        #     # 随机打乱选项顺序
        #     option_labels = ["A", "B", "C", "D"]
        #     shuffled_indices = list(range(4))
        #     random.shuffle(shuffled_indices)

        #     # 找出正确答案的新位置
        #     correct_option = option_labels[shuffled_indices.index(0)]

        #     # 构建选项文本
        #     option_text = ""
        #     for j, label in enumerate(option_labels):
        #         option_index = shuffled_indices[j]
        #         option_text += f"{label}. {options[option_index]}\n"

        prompt_new = f"{process_original_prompt(original_prompt)}\n" "Your choice: /no_think"
        data = {"qid": qid, "prompt": prompt_new, "output": correct_output}
        data.update(distractors)
        new_data.append(data)
    #     data
    #     new_data.append()

    #     # 进度打印
    #     if (i + 1) % 50 == 0:
    #         print(f"已处理 {i + 1}/{len(data)} 条数据")

    return new_data


def test_single_mode(data, mode, batch_size=8):
    """
    测试单一模式的干扰项生成效果
    """
    print(f"\n{'='*50}")
    print(f"测试模式: {mode}")
    print(f"{'='*50}")

    # 取前10条数据进行测试
    test_data = data[:10]
    distractors = generate_distractors_by_mode(test_data, mode, batch_size)

    for i, (item, distractor) in enumerate(zip(test_data, distractors)):
        print(f"\n--- 样本 {i+1} ---")
        print(f"正确答案: {item['output']}")
        print(f"{mode} 干扰项: {distractor}")

        if i >= 4:  # 只显示前5个样本
            break

    return distractors


# ========================= 6. 执行 ========================= #
BATCH_SIZE = 16
print(f"开始处理 {len(data)} 条数据，批量大小: {BATCH_SIZE}")

# 可选：测试各种模式效果（取消注释来测试）
# test_modes = ["style_violation", "topic_violation", "richness_violation",
#               "profile_violation", "conversation_violation", "both_violation"]
# for mode in test_modes:
#     test_single_mode(data, mode, BATCH_SIZE)

# 正式处理数据
new_data = process_data_batch(data, batch_size=BATCH_SIZE)

# ========================= 7. 保存结果 ========================= #
SAVE_PATH = "/home/zyangdm/ROLL/Personality-Alignment/new_changed_dialogue_dataset_multi_mode_v1.jsonl"
print("保存新数据集…")
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"处理完成！共生成 {len(new_data)} 条新数据")
print("每条数据包含1个正确答案和3个来自不同违反模式的干扰项")

# 统计信息
print(f"\n生成模式说明：")
print(f"1. style_violation: 违反句式结构")
print(f"2. topic_violation: 违反话题内容")
print(f"3. richness_violation: 违反内容丰富度")
print(f"4. profile_violation: 违反个性档案")
print(f"5. conversation_violation: 违反对话上下文")
print(f"6. both_violation: 同时违反档案和对话")
