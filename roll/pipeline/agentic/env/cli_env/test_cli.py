#!/usr/bin/env python3
"""
Simple test for the CLI environment with sync wrapper.
"""

from cli_env import CLIEnv


def test_basic_functionality():
    """Test basic CLI environment functionality."""
    print("🧪 Testing CLI Environment...")

    # Test local mode (default)
    env = CLIEnv(max_steps=5)

    print("Environment created")

    # Test reset
    obs, info = env.reset()
    print("Environment reset successful")
    print(f"Task: {info['task']}")
    print(f"Target files: {info['target_files']}")
    print(f"observation: {obs}")

    # Test basic actions
    # actions = [
    #     # "乱七八糟的模型输出与分析balabalabala123<answer>list_directory</answer>",
    #     # "乱七八糟的模型输出与分析balabalabala234<answer>list_directory ~</answer>",
    #     # "乱七八糟的模型输出与分析balabalabala123<answer>list_directory /root/.npm-global/bin/iflow</answer>",
    #     # "乱七八糟的模型输出与分析balabalabala1<answer>write_file test.txt 'Hello from weixun/CLI env'</answer>",
    #     # "乱七八糟的模型输出与分析balabalabala<answer>read_file test.txt</answer>",
    #     # "乱七八糟的模型输出与分析balabalabala<answer>web_search 中国的首都是哪里？</answer>",
    #     # # "乱七八糟的模型输出与分析balabalabala<answer>todo_write 1 2 3 4</answer>", # not supported yet，主要是参数咋配的，还不太清楚
    #     # "乱七八糟的模型输出与分析balabalabala<answer>run_shell_command pwd</answer>",
    #     # "乱七八糟的模型输出与分析balabalabala<answer>list_directory</answer>",
    #     "乱七八糟的模型输出与分析balabalabala<answer>iflow_call '帮我在当前目录生成一个hello.txt文件，文件里面的内容是hello world'</answer>",
    #     "乱七八糟的模型输出与分析balabalabala<answer>read_file hello.txt</answer>",
    #     # "乱七八糟的模型输出与分析balabalabala<answer>iflow_call '搜一下中国的首都是哪里'</answer>",
    #     # "乱七八糟的模型输出与分析balabalabala<answer>run_shell_command ls -al ~</answer>",
    # ]

    actions = (
        ["乱七八糟的模型输出与分析balabalabala<answer>run_shell_command ls -al ~</answer>"]
        + env.annotated_actions
        + ["乱七八糟的模型输出与分析balabalabala<answer>run_shell_command ls -al ~</answer>"] * 3
    )

    for i, action in enumerate(actions):
        print(f"\n📋 Step {i+1}: {action}")
        obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"   Result: {obs}")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")
        print(f"   Step_info: {step_info}")

        # Test termination
        if terminated or truncated:
            break

    env.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_basic_functionality()
