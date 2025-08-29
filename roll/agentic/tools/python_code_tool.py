from gem.tools.python_code_tool import PythonCodeTool as GEMPythonCodeTool
from gem.utils.sandbox import run_python


class PythonCodeTool(GEMPythonCodeTool):


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
