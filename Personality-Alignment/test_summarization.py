#!/usr/bin/env python3
"""
测试总结功能的脚本
"""
import json
from build_convo_data import format_history


def test_format_history():
    """测试 format_history 函数的基本功能"""

    # 创建测试消息列表
    test_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "model", "content": "I'm doing well, thank you for asking. How about you?"},
        {"role": "user", "content": "I'm good too. I wanted to talk about my work."},
        {"role": "model", "content": "That sounds interesting. What kind of work do you do?"},
        {"role": "user", "content": "I work as a software engineer at a tech company."},
        {"role": "model", "content": "That's great! Software engineering is such an exciting field."},
    ]

    # 测试没有模型的情况（正常格式化）
    result_no_model = format_history(test_messages)
    print("=== 测试无模型情况 ===")
    print(f"结果长度: {len(result_no_model)}")
    print(f"结果:\n{result_no_model}")
    print()

    # 创建一个超长的消息列表用于测试总结触发
    long_messages = []
    for i in range(50):  # 创建足够长的对话
        long_messages.append(
            {
                "role": "user",
                "content": f"This is a very long message number {i} that contains lots of text to make the total length exceed 5000 characters. "
                * 10,
            }
        )
        long_messages.append(
            {
                "role": "model",
                "content": f"This is the model's response number {i} which is also quite long and contains detailed information about various topics. "
                * 8,
            }
        )

    # 测试长消息（应该触发总结逻辑，但由于没有模型会直接返回完整历史）
    result_long = format_history(long_messages)
    print("=== 测试长消息情况（无模型）===")
    print(f"原始消息数量: {len(long_messages)}")
    print(f"结果长度: {len(result_long)}")
    print(f"是否超过5000字符: {len(result_long) > 5000}")
    print()

    # 测试带有模拟总结的情况
    print("=== 模拟总结逻辑 ===")
    if len(result_long) > 5000:
        print("✅ 检测到长文本，在实际运行中会触发模型总结")
        # 模拟总结过程
        total_msgs = len(long_messages)
        keep_recent = max(1, total_msgs // 3)
        print(f"总消息数: {total_msgs}")
        print(f"保留最近消息数: {keep_recent}")
        print(f"需要总结的早期消息数: {total_msgs - keep_recent}")
    else:
        print("文本长度未超过阈值，不需要总结")


if __name__ == "__main__":
    test_format_history()
