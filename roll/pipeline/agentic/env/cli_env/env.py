import asyncio
import json
import re
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gem
from gem import Env

from .utils import (
    generate_random_cli_task,
    validate_cli_command,
    create_iflow_call,
    create_iflow_search,
    create_iflow_shell,
    create_iflow_read_file,
)


class CLIEnv(Env):

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 20,
        workspace_dir: str = "/tmp/cli_workspace",
        sandbox_image: str = "hub.docker.alibaba-inc.com/chatos/iflow-cli:4.0",  # version
        sandbox_base_url: str = "https://xrl-sandbox.alibaba-inc.com",
        auto_clear_seconds: int = 60 * 20,  # 20 minutes
        format_penalty: float = -0.1,
        debug_info: bool = False,
    ):
        """
        Initialize the CLI environment.

        Args:
            render_mode: The render mode ("human" or "ansi")
            max_steps: Maximum steps per episode
            workspace_dir: Working directory in sandbox
            sandbox_image: Docker image for sandbox
            sandbox_base_url: Base URL for sandbox API
            auto_clear_seconds: Auto clear timeout for sandbox
            format_penalty: Format penalty
        """
        self.max_steps = max_steps
        self.workspace_dir = workspace_dir
        self.sandbox_image = sandbox_image
        self.sandbox_base_url = sandbox_base_url
        self.auto_clear_seconds = auto_clear_seconds
        self.debug_info = debug_info

        self.render_mode = render_mode
        self.format_penalty = format_penalty

        # State tracking
        self.current_step = 0
        self.xrl_client = None
        self.current_observation = ""
        self.task_description = ""
        self.target_files = []

        # Action mappings
        self.action_names = [
            "list_directory",
            "read_file",
            "read_abs_path_file",
            "write_file",
            "run_shell_command",
            "search_file_content",
            "web_search",
            "todo_write",
            "iflow_call",
        ]

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        Env.reset(self, seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0

        # Initialize environment (xrl client or local mode)
        self._init_sandbox_sync()

        # Generate new task
        (
            self.task_description,
            self.target_files,
            self.annotated_actions,
            self.task_completion_description,
        ) = self._generate_task()

        self.task_completion_description += "If completed, return task_done, otherwise return task_fail."

        # Set up initial workspace
        self._setup_workspace_sync()

        # Get initial observation
        self.current_observation = self._format_initial_observation(self.task_description, self.workspace_dir)

        info = {
            "task": self.task_description,
            "target_files": self.target_files,
            "workspace": self.workspace_dir,
            "step": self.current_step,
            "env_instruction": "",
        }

        return self.current_observation, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.

        Args:
            action: action string

        Returns:
            observation: Result of the action
            reward: Reward for the action
            terminated: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.current_step += 1
        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }
        # Parse action
        action_str = str(action).strip()
        match = re.search(r"<answer>(.*?)</answer>", action_str, re.DOTALL)
        if match:
            # 使用正则表达式抽取<answer>和</answer>中间的内容
            action_str = match.group(1).strip()

        # Map action name to index
        action_name = action_str.split()[0]
        if action_name not in self.action_names:
            self.current_observation = f"Invalid action: {action_name}. Available: {', '.join(self.action_names)}"
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "format_penalty": -self.format_penalty,
            }
            info = {"metrics": metrics, "step": self.current_step, "metrics_agg_mode": metrics_agg_mode}
            return (
                self.current_observation,
                -self.format_penalty,
                False,
                False,
                info,
            )

        try:
            result = self._execute_cli_action_sync(action_str)
            self.current_observation = result

            terminated = self._check_task_completion(result)

            if terminated:
                reward = 1.0
            else:
                reward = self._calculate_reward(action_str, result)

            truncated = self.current_step >= self.max_steps

        except Exception as e:
            self.current_observation = f"Error: {str(e)}"
            reward = -1.0
            terminated = False
            truncated = False

        metrics = {
            "action_is_effective": False,
            "action_is_valid": False,
            "success": False,
            "format_penalty": -self.format_penalty,
        }
        info = {"metrics": metrics, "step": self.current_step, "action": action_str, "metrics_agg_mode": metrics_agg_mode}
        return self.current_observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Task: {self.task_description}")
            print(f"Current observation: {self.current_observation}")
        elif self.render_mode == "ansi":
            return f"Step: {self.current_step}\nTask: {self.task_description}\nObservation: {self.current_observation}"

    def close(self):
        if self.xrl_client:
            self.xrl_client.close()
            self.xrl_client = None

    def _setup_workspace_sync(self):
        if self.xrl_client:
            # Use xrl client for workspace setup
            self.xrl_client.run(f"mkdir -p {self.workspace_dir}")
            self.xrl_client.run(f"cd {self.workspace_dir}")
        else:
            raise Exception("env not ready")

    def _execute_cli_action_sync(self, action_str: str) -> str:
        parts = action_str.split(maxsplit=1)
        action_name = parts[0]
        params = parts[1] if len(parts) > 1 else ""

        if self.debug_info:
            print(f"action_str: {action_str}")
            print(f"action_name: {action_name}")
            print(f"params: {params}")

        # TODO validate action检查，避免危险的shell的命令
        if not validate_cli_command(params):
            print(f"dangerous command: {params}")

        # TODO 当前的封装感觉还是不够优雅
        if self.xrl_client:
            # Use xrl client for all operations
            if action_name == "list_directory":
                target_dir = params if params else "."
                return self.xrl_client.run(f"ls -al {target_dir}")
            elif action_name == "write_file":
                if not params:
                    return "Error: Please specify filename and content"

                parts = params.split(maxsplit=1)
                if len(parts) < 2:
                    return "Error: Please specify both filename and content"
                filename, content = parts
                # Use echo for simple content writing
                escaped_content = content.replace('"', '\\"')
                return self.xrl_client.run(f'echo "{escaped_content}" > {self.workspace_dir}/{filename}')
            elif action_name == "search_file_content":
                pattern = params if params else ""
                if not pattern:
                    return "Error: Please specify search pattern"
                return self.xrl_client.run(f"grep -r '{pattern}' {self.workspace_dir} || echo 'No matches found'")
            elif action_name == "read_file":
                # 拼上workspace_dir
                return self.xrl_client.run(create_iflow_read_file(f"{self.workspace_dir}/{params}"))
            elif action_name == "read_abs_path_file":
                # 直接就是绝对路径
                return self.xrl_client.run(create_iflow_read_file(params))
            elif action_name == "run_shell_command":
                return self.xrl_client.run(create_iflow_shell(params))
            elif action_name == "web_search":
                return self.xrl_client.run(create_iflow_search(params))
            elif action_name == "iflow_call":
                return self.xrl_client.run(create_iflow_call(params))

            else:
                return f"Unknown action: {action_name}"

        raise Exception("sandbox not ready")

    def _calculate_reward(self, action: str, result: str) -> float:
        """Calculate reward based on action and result."""
        if "Error" in result:
            return -1.0
        elif "No matches found" in result or "No such file" in result:
            return -0.5
        elif any(target in result for target in self.target_files):
            return 1.0
        else:
            return 0.1  # Small positive reward for successful actions

    def _check_task_completion(self, result: str) -> bool:
        #     # 利用iflow call的能力，所以之类就是一个agentic能力
        #     task_result = self._execute_cli_action_sync(
        #         f"iflow_call '{self.task_completion_description}'"
        #     )
        """检查任务是否完成 - 使用simple_task_checker.py的能力"""
        try:
            from .simple_task_checker import SimpleTaskChecker

            checker = SimpleTaskChecker(self)
            return checker.check_task_completion(self.task_description, self.target_files)

        except ImportError:
            print("simple_task_checker.py 不可用，使用内置检查逻辑")
            return self._check_task_completion_fallback(result)
        except Exception as e:
            print(f"任务检查错误: {str(e)}")
            return False

    def _init_sandbox_sync(self, session_id: str = None):
        """Initialize synchronous mode using xrl wrapper or local fallback."""
        try:
            from .xrl_wrapper import create_sync_client

            self.xrl_client = create_sync_client(
                sandbox_image=self.sandbox_image,
                sandbox_base_url=self.sandbox_base_url,
                auto_clear_seconds=self.auto_clear_seconds,
                debug_info=self.debug_info,
            )
            # Start the client
            success = self.xrl_client.start()
            if not success:
                print("xrl client failed to start, using local mode")
                self.xrl_client = None
        except ImportError:
            print("xrl wrapper not available, using local mode")
            self.xrl_client = None

    def _generate_task(self) -> Tuple[str, List[str], str]:
        return generate_random_cli_task()

    def _format_initial_observation(self, task_description, workspace_dir) -> str:
        return (
            "Welcome to the CLI Environment!\n"
            f"Task: {task_description}\n\n"
            "Available actions:\n"
            "0: list_directory - List the contents of the current directory\n"
            "1: read_file - Read file content (filename required)\n"
            "2: write_file - Write to file (filename and content required)\n"
            "3: run_shell_command - Run a shell command\n"
            "4: search_file_content - Search for content in files\n"
            "5: web_search - Search the web for information\n"
            "6: todo_write - Write a todo item\n"
            "7: iflow_call - Call an iflow function\n\n"
            f"Current directory: {workspace_dir}\n\n"
            "Please complete the task using the available CLI commands.\n"
            "Format your response as: <answer>action_name [parameters]</answer>\n"
        )

    def sample_random_action(self) -> str:
        """Samples a random action given the current state."""
        import random

        return random.choice(self.action_names)
