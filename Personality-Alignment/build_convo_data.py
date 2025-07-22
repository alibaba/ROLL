import json
from pathlib import Path


def load_profiles(profile_path: str) -> dict:
    """profile.json → {user_id: profile_text}"""
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iterate_messages(record: dict):
    """
    根据常见字段名把一条对话里的 message 列表取出来。
    你可以按需要再补别名。
    """
    return record["conversations"]


def format_history(messages: list) -> str:
    """
    格式化对话历史为字符串。
    假设 messages 是一个列表，包含多个消息字典，每个字典有 'role' 和 'content' 字段。
    """
    history = []
    for msg in messages:
        if msg["content"].strip() == "EMPTY STRING":
            continue
        if msg.get("role") == "user":
            history.append(f"Your target simulate person says: {msg['content']}")
        elif msg.get("role") == "model":
            history.append(f"LLM assistant says: {msg['content']}")
    return "\n".join(history)


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


def build_dataset(roleplay_path: str, profile_path: str, output_path: str):
    profiles = load_profiles(profile_path)
    new_records = []

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
                    history_str = format_history(history_msgs)
                    # if msg.get("content").endswith("?"):
                    #     continue  # 跳过问题
                    conversation_history = format_conversation_history(history_str, record)
                    new_records.append(
                        {
                            "qid": record.get("qid") or record.get("id") or f"r{line_idx}_{msg_idx}",
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

    # 写出新数据集
    with open(output_path, "w", encoding="utf-8") as out_f:
        for rec in new_records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build_dataset("roleplay_dataset_en_new.jsonl", "profile.json", "dialogue_dataset_all_v4.jsonl")
    print("✅ 生成完成：dialogue_dataset.jsonl")
