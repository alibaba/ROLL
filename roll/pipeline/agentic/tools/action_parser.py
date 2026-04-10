import json
import logging
import re
import time
from abc import ABC, abstractmethod

from roll.utils.logging import get_logger


class ActionParser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def parse_action(self, response: str):
        raise NotImplementedError


class Qwen3CoderActionParser(ActionParser):
    def __init__(
        self,
    ):
        self.logger: logging.Logger = get_logger()

    def parse_action(self, response: str):
        """
        Parse the model response to extract tool calls.
        """

        try:
            actions = []
            has_tool_format = False

            if self._is_incomplete_tool_call(response):
                self.logger.info("[ACTION_PARSE] Detected incomplete tool call")
                return False, "Please continue."

            if "<function" in response:
                has_tool_format = True
                function_pattern = r"<function\s*=\s*([^>]+)>(.*?)</function>"
                function_matches = re.findall(function_pattern, response, flags=re.DOTALL)

                if not function_matches:
                    self.logger.info("[ACTION_PARSE] No complete <function=...></function> blocks found.")
                else:
                    for i, (function_name, function_body) in enumerate(function_matches):
                        try:
                            function_name = function_name.strip()

                            param_pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
                            param_matches = re.findall(param_pattern, function_body, flags=re.DOTALL)

                            def _coerce_param_value(v: str):
                                v = v.strip()
                                if not v:
                                    return v

                                # json bool/null
                                if v in ("true", "false", "null"):
                                    try:
                                        return json.loads(v)
                                    except Exception:
                                        return v

                                # int/float
                                if re.fullmatch(r"-?\d+", v):
                                    return int(v)
                                if re.fullmatch(r"-?\d+\.\d+", v):
                                    return float(v)

                                # json container
                                if (v[0] == "[" and v[-1] == "]") or (v[0] == "{" and v[-1] == "}"):
                                    try:
                                        return json.loads(v)
                                    except Exception:
                                        return v

                                return v

                            if not param_matches:
                                self.logger.info(f"[ACTION_PARSE] No <parameter=> blocks found in function {function_name}.")
                                params = {}
                            else:
                                params = {
                                    key.strip(): _coerce_param_value(value)
                                    for key, value in param_matches
                                }

                            cur = {
                                "type": "function",
                                "id": f"{function_name}_{int(time.time() * 1000)}_{i}",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(params, ensure_ascii=False),
                                },
                            }
                            actions.append(cur)
                            self.logger.debug(
                                f"[ACTION_PARSE] Parsed function action {i + 1}: {function_name} with params: {list(params.keys())}"
                            )

                        except Exception as e:
                            self.logger.warning(f"[ACTION_PARSE] Failed to parse function block {i + 1}: {e}")
                            continue
            elif "<tool_call>" in response:
                has_tool_format = True
                if "<tool_call>" in response and "</tool_call>" not in response:
                    response = response + "</tool_call>"
                tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
                tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)

                for i, tool_call_str in enumerate(tool_call_matches):
                    try:
                        tool_call_str = tool_call_str.strip()
                        tool_call_json = json.loads(tool_call_str)
                        function_name = tool_call_json.get("name", "")
                        arguments = tool_call_json.get("arguments", {})
                        cur = {
                            "type": "function",
                            "id": f"{function_name}_{int(time.time() * 1000)}_{i}",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                        actions.append(cur)
                        self.logger.debug(f"[ACTION_PARSE] Parsed tool_call action {i + 1}: {function_name}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"[ACTION_PARSE] Failed to parse JSON in tool_call {i + 1}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"[ACTION_PARSE] Failed to parse tool_call action {i + 1}: {e}")
                        continue

            # 检查是否没有找到任何工具格式
            if not has_tool_format:
                return False, "action parse failed"

            if actions:
                self.logger.info(f"[ACTION_PARSE] Success! - Parsed {len(actions)} tool calls")
                # 打印每个action的详细信息用于调试
                for i, action in enumerate(actions):
                    self.logger.debug(f"Action {i + 1}: ID={action['id']}, Name={action['function']['name']}")
            else:
                self.logger.info("[ACTION_PARSE] No tool calls found in response")

            return True, actions

        except Exception as e:
            self.logger.error(f"[ACTION_PARSE] Failed! - Error parsing action: {e}")
            return False, "工具调用格式错误"

    def _is_incomplete_tool_call(self, response: str) -> bool:
        """
        Check if the response contains an incomplete tool call that should be continued.
        """
        if "<tool_call>" in response and "</tool_call>" not in response:
            if not response.strip().endswith((">")):
                return True

        if "<function" in response:
            incomplete_patterns = [
                r"<parameter\s*=\s*[^>]*>\s*[^<]*$",
                r"<function\s*=\s*[^>]*>\s*<parameter\s*=\s*[^>]*>\s*[^<]*</?\s*$",
                r"<function\s*=\s*[^>]*>\s*<parameter\s*=\s*[^>]*>\s*.*</\s*$",
            ]

            for pattern in incomplete_patterns:
                if re.search(pattern, response, re.DOTALL):
                    return True

        return False


if __name__ == "__main__":
    logger = get_logger()
    tool = Qwen3CoderActionParser()
    response = "Let me check the current directory.<tool_call><function=list_directory><parameter=path>.</parameter></function></tool_call>"

    print(tool.parse_action(response=response))
