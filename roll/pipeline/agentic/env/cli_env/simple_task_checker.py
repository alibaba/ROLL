"""
简化任务检查器 - 完全不使用iflow_call
"""

import os
import re
from typing import List, Dict, Any


class SimpleTaskChecker:
    """简化任务检查器，直接通过命令执行验证"""

    def __init__(self, env_instance):
        self.env = env_instance

    def check_task_completion(self, task_description: str, target_files: List[str]) -> bool:
        """
        检查任务是否完成

        Args:
            task_description: 任务描述
            target_files: 目标文件列表

        Returns:
            任务是否完成
        """
        try:
            task_lower = task_description.lower()

            # 1. 检查文件存在性
            if target_files:
                for target_file in target_files:
                    if not self._check_file_exists(target_file):
                        print(f"文件不存在: {target_file}")
                        return False

            # 2. 根据任务类型进行特定检查
            if "hello world" in task_lower:
                return self._check_file_content("test.txt", "Hello World")

            elif "hello cli" in task_lower:
                return self._check_python_script("hello.py", "Hello CLI")

            elif "test data" in task_lower and "test_dir" in task_lower:
                return self._check_file_content("test_dir/info.txt", "Test data")

            elif "todo" in task_lower and "setup" in task_lower:
                return self._check_todo_file()

            elif "search" in task_lower and "python" in task_lower:
                return self._check_python_files()

            elif "directory" in task_lower:
                return self._check_directory_structure(target_files)

            # 3. 默认检查：所有目标文件存在
            return True

        except Exception as e:
            print(f"检查错误: {str(e)}")
            return False

    def _check_file_exists(self, filename: str) -> bool:
        """检查文件是否存在"""
        try:
            result = self.env._execute_cli_action_sync(f"read_file {filename}")
            return "Error" not in result and "No such file" not in result
        except:
            return False

    def _check_file_content(self, filename: str, expected_content: str) -> bool:
        """检查文件内容"""
        try:
            result = self.env._execute_cli_action_sync(f"read_file {filename}")
            return expected_content in result
        except:
            return False

    def _check_python_script(self, filename: str, expected_output: str) -> bool:
        """检查Python脚本"""
        try:
            # 检查文件内容
            content_result = self.env._execute_cli_action_sync(f"read_file {filename}")
            if expected_output not in content_result:
                return False

            # 尝试执行脚本
            exec_result = self.env._execute_cli_action_sync(f"run_shell_command python3 {filename}")
            return expected_output in exec_result
        except:
            return False

    def _check_todo_file(self) -> bool:
        """检查todo文件"""
        try:
            content = self.env._execute_cli_action_sync("read_file todo.txt")
            keywords = ["Setup", "Develop", "Test"]
            return all(keyword in content for keyword in keywords)
        except:
            return False

    def _check_python_files(self) -> bool:
        """检查Python文件"""
        try:
            result = self.env._execute_cli_action_sync("run_shell_command ls *.py")
            return ".py" in result and "No such file" not in result
        except:
            return False

    def _check_directory_structure(self, target_files: List[str]) -> bool:
        """检查目录结构"""
        try:
            for target_file in target_files:
                dir_path = os.path.dirname(target_file)
                if dir_path and dir_path != ".":
                    dir_result = self.env._execute_cli_action_sync(f"list_directory {dir_path}")
                    if "No such file" in dir_result:
                        return False
            return True
        except:
            return False

    def get_check_commands(self, task_description: str, target_files: List[str]) -> List[str]:
        """获取检查命令列表"""
        commands = ["list_directory"]

        # 添加文件检查命令
        for target_file in target_files:
            commands.append(f"read_file {target_file}")

        # 根据任务类型添加特定命令
        task_lower = task_description.lower()

        if "python" in task_lower:
            for target_file in target_files:
                if target_file.endswith(".py"):
                    commands.append(f"run_shell_command python3 {target_file}")

        if "directory" in task_lower:
            commands.append("run_shell_command find . -type d | head -10")

        if "search" in task_lower:
            commands.append("run_shell_command ls -la")

        return commands


def create_simple_verification(task_description: str, target_files: List[str]) -> Dict[str, Any]:
    """
    创建简单验证配置

    Args:
        task_description: 任务描述
        target_files: 目标文件列表

    Returns:
        验证配置字典
    """
    task_lower = task_description.lower()

    verification_map = {
        "hello world": {"type": "file_content", "file": "test.txt", "expected": "Hello World"},
        "hello cli": {"type": "python_script", "file": "hello.py", "expected_output": "Hello CLI"},
        "test data": {"type": "file_content", "file": "test_dir/info.txt", "expected": "Test data"},
        "todo": {"type": "multi_content", "file": "todo.txt", "expected_keywords": ["Setup", "Develop", "Test"]},
        "search python": {"type": "file_exists", "pattern": "*.py"},
    }

    # 匹配任务类型
    for keyword, config in verification_map.items():
        if keyword in task_lower:
            return config

    # 默认配置
    return {"type": "file_exists", "files": target_files}


# 使用示例
if __name__ == "__main__":
    """
    使用示例：

    # 在CLIEnv中使用
    from simple_task_checker import SimpleTaskChecker

    # 创建检查器
    checker = SimpleTaskChecker(cli_env_instance)

    # 检查任务完成状态
    is_complete = checker.check_task_completion(
        cli_env_instance.task_description,
        cli_env_instance.target_files
    )

    # 获取验证命令
    commands = checker.get_check_commands(
        cli_env_instance.task_description,
        cli_env_instance.target_files
    )

    # 创建验证配置
    config = create_simple_verification(
        cli_env_instance.task_description,
        cli_env_instance.target_files
    )
    """
