#!/usr/bin/env python3
"""
测试无iflow_call的任务检查功能
"""

import os
import sys
import json
from cli_env import CLIEnv


def test_no_iflow_checking():
    """测试不使用iflow_call的任务检查"""
    print("🧪 测试无iflow_call的任务检查...")

    # 创建环境
    env = CLIEnv(max_steps=5)

    # 重置环境获取任务
    obs, info = env.reset()
    print(f"📋 任务: {info['task']}")
    print(f"🎯 目标文件: {info['target_files']}")

    # 模拟完成任务
    task = info["task"]
    target_files = info["target_files"]

    # 先检查
    # 测试新的检查逻辑
    print("\n🔍 测试无iflow_call检查...")
    check_result = env._check_task_completion("")
    print(f"检查结果: {check_result}")

    if check_result:
        print("任务检查通过！")
    else:
        print("任务检查失败！")

    # 验证文件
    print("\n📁 验证文件状态：")
    for target_file in target_files:
        try:
            # content = json.loads(env._execute_cli_action_sync(f"read_file {target_file}"))
            content = env._execute_cli_action_sync(f"read_file {target_file}")
            print(f"return content {content}")
            # print(f"tool_results content: {content['tool_results']['content']}")
            print(
                f"{target_file}: {'存在' if 'Error' not in content and 'File not found' not in content else '不存在'}"
            )
        except Exception as e:
            print(f"{target_file}: 错误 - {str(e)}")

        print("\n🔍 执行任务...")

    # 根据任务类型执行相应操作
    if "Hello World" in task:
        # 创建test.txt
        env.step("write_file test.txt 'Hello World'")

    elif "Hello CLI" in task:
        # 创建hello.py
        env.step('write_file hello.py print("Hello CLI")')

    elif "test_dir" in task and "Test data" in task:
        # 创建目录和文件
        env.step("run_shell_command mkdir -p test_dir")
        env.step("write_file test_dir/info.txt 'Test data'")

    elif "todo" in task:
        # 创建todo文件
        env.step("write_file todo.txt '1. Setup environment\n2. Develop features\n3. Test functionality'")

    print("\n🔍 测试无iflow_call检查...")
    check_result = env._check_task_completion("")
    print(f"检查结果: {check_result}")

    if check_result:
        print("任务检查通过！")
    else:
        print("任务检查失败！")

    # 验证文件
    print("\n📁 验证文件状态：")
    for target_file in target_files:
        try:
            content = env._execute_cli_action_sync(f"read_file {target_file}")
            print(f"{target_file}: {'存在' if 'Error' not in content else '不存在'}")
        except Exception as e:
            print(f"{target_file}: 错误 - {str(e)}")

    env.close()
    print("🎉 测试完成！")


if __name__ == "__main__":
    test_no_iflow_checking()
