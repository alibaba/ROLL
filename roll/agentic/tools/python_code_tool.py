import re
from typing import Tuple, Optional

from gem.tools.python_code_tool import PythonCodeTool as GEMPythonCodeTool
from gem.utils.sandbox import run_python


class PythonCodeTool(GEMPythonCodeTool):

    def __init__(
        self,
        timeout: int = 5,
        sandbox_type: str = "none",
        keep_error_last_line: bool = False,
        tool_instruction=None,
        patterns=None,
    ):
        super().__init__(timeout, sandbox_type, keep_error_last_line)
        self.tool_instruction = ("Initially, when solving a question, you would need to think step by step, without the ability to use code for calculation. "
            "Now, you have the capability to write code to use the code interpreter for calculation. "
            "The code will be executed by a sandbox, and the result can be returned to enhance your reasoning process. your calculation while still maintaining the reasoning process."
            "The thinking process can ""have multiple code snippets. Each code snippet is wrapped with: <code>...</code>, and should be executable."
            "Details:"
            "1. Identify sections where code execution could speed up the reasoning process or make the calculation more accurate."
            "2. Replace the manual calculation steps with code snippets and the corresponding interpreter's execution results."
            "3. Keep the logical flow of the reasoning process intact, including any failed exploration attempts that were part of the initial process."
            "4. The code snippets should be complete scripts, including necessary imports, and should not contain markdown symbols like <python>...‹/python>."
            "5. Outputs in the code snippets must explicitly call the print function."
            "6. Execution results should match the model's output exactly, with no extra or missing tokens.")
        self.patterns = [r"<code>(.*?)</code>", r"```\n?python(.*?)```"]
        if tool_instruction:
            self.tool_instruction = tool_instruction
        if patterns:
            self.patterns = patterns

    def _parse_action(self, action: str) -> tuple[Optional[str], str, bool]:
        parsed_code = None
        parsed_action = action
        is_valid = False
        prev_end = len(action)
        for pattern in self.patterns:
            # Search for the first occurrence of the pattern
            matches = re.search(pattern, action, re.DOTALL)
            if matches:
                is_valid = True
                if matches.end() <= prev_end:
                    parsed_code = matches.group(1).strip()
                    parsed_action = action[: matches.end()]
                    prev_end = matches.end()
        return parsed_code, parsed_action, is_valid

    def instruction_string(self) -> str:
        return self.tool_instruction

    def execute_action(self, action):
        """
        Execute the parsed action
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_code, parsed_action, is_valid = self._parse_action(action)

        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = ""
            has_error = True
        else:
            success, stdout, stderr = run_python(
                parsed_code, self.sandbox_type, timeout=self.timeout
            )
            has_error = not success
            if stderr and self.keep_error_last_line:
                stderr = stderr.split("\n")[-1]
            execution_result = f"{stdout}\n{stderr}" if stderr else stdout

            observation = execution_result.lstrip(" \n")
            if len(observation) == 0:
                has_error = True

            observation = "Code execution result: " + observation + "\n"

        return is_valid, has_error, observation, parsed_action
