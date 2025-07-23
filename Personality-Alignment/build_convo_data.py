import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


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
        attn_implementation="flash_attention_2",
    )

    return model, tokenizer


def summarize_history_batch(
    model, tokenizer, history_texts: list, max_summary_length: int = 1000, batch_size: int = 8
) -> list:
    """
    批量使用 Qwen3-8B 模型对历史对话进行总结

    参数:
    model: 加载的 Qwen3-8B 模型
    tokenizer: 对应的分词器
    history_texts: 需要总结的历史对话文本列表
    max_summary_length: 总结的最大长度
    batch_size: 批量大小

    返回:
    总结后的文本列表
    """
    all_summaries = []
    total_batches = (len(history_texts) + batch_size - 1) // batch_size

    progress_bar = tqdm(range(0, len(history_texts), batch_size), desc="批量总结历史对话", total=total_batches)

    for i in progress_bar:
        batch_texts = history_texts[i : i + batch_size]
        batch_prompts = [
            f"""Please summarize the following conversation history in no more than {max_summary_length} characters. Keep the key information especially about the target simulate person's personality, preferences, and important context:

{text}

Summary: /no_think"""
            for text in batch_texts
        ]

        # 批量tokenize
        input_tokens = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=10000
        ).to(model.device)

        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 批量解码
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 处理这一批的结果
        batch_summaries = []
        for j, text in enumerate(generated_texts):
            summary = text.replace(batch_prompts[j], "").strip()
            # 确保总结不超过指定长度
            if len(summary) > max_summary_length:
                summary = summary[:max_summary_length].rsplit(" ", 1)[0] + "..."
            batch_summaries.append(summary)

        all_summaries.extend(batch_summaries)

        # 更新进度条信息
        progress_bar.set_postfix(
            {"已完成": len(all_summaries), "总数": len(history_texts), "批次大小": len(batch_texts)}
        )

    return all_summaries


def summarize_history(model, tokenizer, history_text: str, max_summary_length: int = 1000) -> str:
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
    prompt = f"""Please summarize the following conversation history in no more than {max_summary_length} characters. Keep the key information especially about the target simulate person's personality, preferences, and important context:

{history_text}

Summary: /no_think"""

    # Tokenize
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=10000).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=1000,  # 控制总结长度
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


def format_history_prepare(messages: list, max_length: int = 6000) -> tuple:
    """
    准备格式化对话历史，返回是否需要总结和相关信息

    参数:
    messages: 消息列表
    max_length: 触发总结的最大长度阈值

    返回:
    (formatted_history, needs_summary, early_text, recent_history_formatted)
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
    if len(full_history) > max_length:
        # 计算需要保留的最近消息数量（保留最后约1/3的对话）
        total_msgs = len(history)
        keep_recent = max(3, total_msgs // 3)

        # 分离早期历史和最近历史
        early_history = history[:-keep_recent]
        recent_history = history[-keep_recent:]

        if early_history:  # 只有当有早期历史时才进行总结
            early_text = "\n".join(early_history)
            recent_history_formatted = recent_history
            return full_history, True, early_text, recent_history_formatted

    return full_history, False, None, None


def format_history_with_summary(summary: str, recent_history_formatted: list) -> str:
    """
    使用提供的总结来格式化历史记录
    """
    result_parts = (
        ["[Earlier conversation summary]", summary, "[End of summary]", "", "[Recent conversation]"]
        + recent_history_formatted
        + ["[End of recent conversation]"]
    )
    return "\n".join(result_parts)


def format_history(messages: list, model=None, tokenizer=None, max_length: int = 6000) -> str:
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
    summarized = False
    # 检查是否需要总结
    if len(full_history) > max_length and model is not None and tokenizer is not None:
        # 计算需要保留的最近消息数量（保留最后约1/3的对话）
        total_msgs = len(history)
        keep_recent = max(3, total_msgs // 3)

        # 分离早期历史和最近历史
        early_history = history[:-keep_recent]
        recent_history = history[-keep_recent:]

        if early_history:  # 只有当有早期历史时才进行总结
            early_text = "\n".join(early_history)
            summary = summarize_history(model, tokenizer, early_text)
            summarized = True
            # 组合总结和最近历史
            result_parts = (
                ["[Earlier conversation summary]", summary, "[End of summary]", "", "[Recent conversation]"]
                + recent_history
                + ["[End of recent conversation]"]
            )

            return "\n".join(result_parts), summarized

    return full_history, summarized


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


def build_dataset(
    roleplay_path: str, profile_path: str, output_path: str, use_summarization: bool = True, batch_size: int = 8
):
    """
    构建数据集

    参数:
    roleplay_path: 角色扮演数据路径
    profile_path: 配置文件路径
    output_path: 输出路径
    use_summarization: 是否使用总结功能
    batch_size: 批量处理大小
    """
    profiles = load_profiles(profile_path)

    # 第一步：统计总行数（用于进度显示）
    print("🔄 第一步：统计数据总量...")
    total_lines = 0
    with open(roleplay_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
    print(f"📊 数据文件总行数：{total_lines}")

    # 第二步：收集所有需要处理的数据
    print("🔄 第二步：收集所有需要处理的消息...")
    all_processing_items = []

    with open(roleplay_path, "r", encoding="utf-8") as f:
        progress_bar = tqdm(enumerate(f), total=total_lines, desc="收集消息")
        for line_idx, line in progress_bar:
            if not line.strip():
                continue
            try:
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

                        # 准备格式化历史
                        full_history, needs_summary, early_text, recent_history_formatted = format_history_prepare(
                            history_msgs
                        )

                        item = {
                            "qid": f"{user_id}_{line_idx}_{msg_idx}",
                            "user_id": user_id,
                            "profile_text": profile_text,
                            "record": record,
                            "output": msg["content"],
                            "full_history": full_history,
                            "needs_summary": needs_summary,
                            "early_text": early_text,
                            "recent_history_formatted": recent_history_formatted,
                        }
                        all_processing_items.append(item)

                # 更新进度条信息
                progress_bar.set_postfix(
                    {
                        "已收集": len(all_processing_items),
                        "当前用户": str(user_id)[:10] + "..." if len(str(user_id)) > 10 else str(user_id),
                    }
                )
            except json.JSONDecodeError as e:
                print(f"⚠️ 第{line_idx+1}行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"⚠️ 第{line_idx+1}行处理错误: {e}")
                continue

    print(f"✅ 收集完成，共收集到 {len(all_processing_items)} 条消息需要处理")

    # 第三步：批量处理需要总结的历史
    summaries = {}
    if use_summarization:
        print("🔄 第三步：加载模型...")
        model, tokenizer = load_qwen3_8b(model_path="Qwen/Qwen3-8B", device_map="auto", quantize=False)
        print("✅ 模型加载完成")

        # 找出所有需要总结的项目
        items_need_summary = [item for item in all_processing_items if item["needs_summary"]]

        if items_need_summary:
            print(f"🔄 第四步：批量总结历史对话（共 {len(items_need_summary)} 条需要总结）...")

            # 提取所有需要总结的文本
            early_texts = [item["early_text"] for item in items_need_summary]

            # 批量总结
            batch_summaries = summarize_history_batch(model, tokenizer, early_texts, batch_size=batch_size)

            # 将总结结果映射回对应的项目
            for item, summary in zip(items_need_summary, batch_summaries):
                summaries[item["qid"]] = summary

            print("✅ 批量总结完成")
        else:
            print("ℹ️ 没有需要总结的历史对话")

        # 清理模型以释放显存
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🗑️ 模型已清理，显存已释放")
    else:
        print("ℹ️ 跳过总结功能")

    # 第五步：构建最终数据集
    print("🔄 第五步：构建最终数据集...")
    new_records = []

    progress_bar = tqdm(all_processing_items, desc="构建数据集")
    for item in progress_bar:
        # 根据是否需要总结来格式化历史
        if item["needs_summary"] and use_summarization and item["qid"] in summaries:
            history_str = format_history_with_summary(summaries[item["qid"]], item["recent_history_formatted"])
            summarized = True
        else:
            history_str = item["full_history"]
            summarized = False

        conversation_history = format_conversation_history(history_str, item["record"])

        new_records.append(
            {
                "qid": item["qid"],
                "prompt": (
                    "Now, you are required to simulate the person with profile below:\n"
                    "[Profile Begin]\n"
                    f"{item['profile_text']}\n\n"
                    "[Profile End]\n"
                    f"{conversation_history}\n"
                    "Now you should generate a response as if you are the person.\n"
                    "Your output should align with the profile of the person and the conversation history.\n"
                    "Now, your output:"
                ),
                "output": item["output"],
                "summarized": summarized,
            }
        )

        # 更新进度条信息
        progress_bar.set_postfix(
            {"已处理": len(new_records), "已总结": sum(1 for rec in new_records if rec.get("summarized", False))}
        )

    # 写出新数据集
    print("🔄 第六步：保存数据集...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        progress_bar = tqdm(new_records, desc="保存数据")
        for rec in progress_bar:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计信息
    summarized_count = sum(1 for rec in new_records if rec.get("summarized", False))
    print(f"✅ 生成完成：{output_path}")
    print(f"📊 总记录数：{len(new_records)}")
    print(f"📊 使用总结的记录数：{summarized_count}")
    print(f"📊 总结比例：{summarized_count/len(new_records)*100:.1f}%")


if __name__ == "__main__":
    # 可以通过 use_summarization 参数控制是否启用总结功能
    build_dataset(
        "roleplay_dataset_en_new.jsonl",
        "profile.json",
        "dialogue_dataset_all_v5_summarized.jsonl",
        use_summarization=True,
    )
    print("✅ 生成完成：dialogue_dataset_all_v5_summarized.jsonl")
