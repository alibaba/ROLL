import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_profiles(profile_path: str) -> dict:
    """profile.json → {user_id: profile_text}"""
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qwen3_8b(model_path="Qwen/Qwen3-8B", device_map="auto", quantize=False):
    """
    加载 Qwen3-8B 模型和分词器

    参数:
    model_path: 模型路径 (Hugging Face ID 或本地路径)
    device_map: 设备映射 ("auto", "cuda", "cpu")
    quantize: 是否使用 4-bit 量化 (减少显存需求)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # 确保分词器左侧填充
    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    return model, tokenizer


def summarize_history(model, tokenizer, history_text: str, max_summary_length: int = 500) -> str:
    """
    使用 Qwen3-8B 模型对历史对话进行总结

    参数:
    model: 加载的 Qwen3-8B 模型
    tokenizer: 对应的分词器
    history_text: 需要总结的历史对话文本
    max_summary_length: 总结的最大长度

    返回:
    总结后的文本
    """
    prompt = f"""Please summarize the following conversation history in no more than {max_summary_length} characters. Keep the key information especially about the person's personality, preferences, and important context:

{history_text}

Summary:"""

    # Tokenize
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=6000,  # 控制总结长度
            do_sample=True,
            temperature=0.3,  # 较低的温度保证总结质量
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = generated_text.replace(prompt, "").strip()

    # 确保总结不超过指定长度
    if len(summary) > max_summary_length:
        summary = summary[:max_summary_length].rsplit(" ", 1)[0] + "..."

    return summary


def iterate_messages(record: dict):
    """
    根据常见字段名把一条对话里的 message 列表取出来。
    你可以按需要再补别名。
    """
    return record["conversations"]


def format_history(messages: list, model=None, tokenizer=None, max_length: int = 5000) -> str:
    """
    格式化对话历史为字符串，当长度超过阈值时使用模型进行总结。
    假设 messages 是一个列表，包含多个消息字典，每个字典有 'role' 和 'content' 字段。

    参数:
    messages: 消息列表
    model: Qwen3-8B 模型（可选，用于总结）
    tokenizer: 对应的分词器（可选，用于总结）
    max_length: 触发总结的最大长度阈值
    """
    history = []
    for msg in messages:
        if msg["content"].strip() == "EMPTY STRING":
            continue
        if msg.get("role") == "user":
            history.append(f"Your target simulate person says: {msg['content']}")
        elif msg.get("role") == "model":
            history.append(f"LLM assistant says: {msg['content']}")

    # 组合历史记录
    full_history = "\n".join(history)

    # 检查是否需要总结
    if len(full_history) > max_length and model is not None and tokenizer is not None:
        # 计算需要保留的最近消息数量（保留最后约1/3的对话）
        total_msgs = len(history)
        keep_recent = max(1, total_msgs // 3)

        # 分离早期历史和最近历史
        early_history = history[:-keep_recent]
        recent_history = history[-keep_recent:]

        if early_history:  # 只有当有早期历史时才进行总结
            early_text = "\n".join(early_history)
            summary = summarize_history(model, tokenizer, early_text)

            # 组合总结和最近历史
            result_parts = (
                ["[Earlier conversation summary]", summary, "[End of summary]", "", "[Recent conversation]"]
                + recent_history
                + ["[End of recent conversation]"]
            )

            return "\n".join(result_parts)

    return full_history


def format_conversation_history(history_str: str, record: dict) -> str:
    """
    格式化对话历史为完整的对话字符串。
    假设 history_str 是格式化后的对话历史字符串，record 包含其他信息。
    """
    opening_prompt = record.get("opening_prompt", "")
    conversation_type = record.get("conversation_type", "")
    description = None
    if conversation_type == "unguided":
        description = " This is an unguided conversation without any specific topic. The person could ask, request or talk to the model about anything."
    elif conversation_type == "values guided":
        description = " This is a value guided conversation. The person is required to ask, request or talk to the model about something important to it or that represents its values. This could be related to work, religion, family and relationship, politics or culture."
    elif conversation_type == "controversy guided":
        description = " This is a controversy guided conversation. The person is required to ask, request or talk to the model about something controversial or where people would disagree in its community, culture or country"
    else:
        raise ValueError(f"Unknown conversation type: {conversation_type}")
    return (
        f"Below is the conversation history between the person and an LLM assistant.\n{description}\n"
        "[Conversation History Begin]\n"
        f"{history_str}\n"
        "[Conversation History End]\n"
    )


def build_dataset(roleplay_path: str, profile_path: str, output_path: str, use_summarization: bool = True):
    """
    构建数据集

    参数:
    roleplay_path: 角色扮演数据路径
    profile_path: 配置文件路径
    output_path: 输出路径
    use_summarization: 是否使用总结功能
    """
    profiles = load_profiles(profile_path)
    new_records = []

    # 如果启用总结功能，加载模型
    model, tokenizer = None, None
    if use_summarization:
        print("🔄 加载 Qwen3-8B 模型用于历史总结...")
        model, tokenizer = load_qwen3_8b(model_path="Qwen/Qwen3-8B", device_map="auto", quantize=False)
        print("✅ 模型加载完成")

    with open(roleplay_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = iterate_messages(record)

            # 取 user_id；按需补充或修改字段名
            user_id = record.get("user_id") or record.get("uid") or record.get("profile_id") or record.get("user")

            profile_text = profiles.get(str(user_id), "")  # 若找不到可留空

            # 遍历 message，找到第一条 role=user 作为输出
            for msg_idx, msg in enumerate(messages):
                if msg_idx == 0:
                    continue
                if msg.get("role") == "user":
                    history_msgs = messages[:msg_idx]
                    # 传递模型参数给 format_history
                    history_str = format_history(history_msgs, model=model, tokenizer=tokenizer)
                    # if msg.get("content").endswith("?"):
                    #     continue  # 跳过问题
                    conversation_history = format_conversation_history(history_str, record)
                    new_records.append(
                        {
                            "qid": f"{user_id}_{line_idx}_{msg_idx}",
                            "prompt": (
                                "Now, you are required to simulate the person with profile below:\n"
                                "[Profile Begin]\n"
                                f"{profile_text}\n\n"
                                "[Profile End]\n"
                                f"{conversation_history}\n"
                                "Now you should generate a response as if you are the person.\n"
                                "Your output should align with the profile of the person and the conversation history.\n"
                                "Now, your output:"
                            ),
                            "output": msg["content"],
                        }
                    )

            # 显示进度
            if (line_idx + 1) % 100 == 0:
                print(f"已处理 {line_idx + 1} 行数据")

    # 写出新数据集
    with open(output_path, "w", encoding="utf-8") as out_f:
        for rec in new_records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ 生成完成：{output_path}，共生成 {len(new_records)} 条记录")


if __name__ == "__main__":
    # 可以通过 use_summarization 参数控制是否启用总结功能
    build_dataset(
        "roleplay_dataset_en_new.jsonl",
        "profile.json",
        "dialogue_dataset_all_v5_summarized.jsonl",
        use_summarization=True,
    )
    print("✅ 生成完成：dialogue_dataset_all_v5_summarized.jsonl")
