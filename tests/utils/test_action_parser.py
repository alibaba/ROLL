import re

import pytest

from roll.pipeline.agentic.tools.action_parser import Qwen3CoderActionParser


def test_qwen3coder_action_parser_parse_action_single_call():
    tool = Qwen3CoderActionParser()
    response = (
        "Let me check the current directory."
        "<tool_call><function=list_directory><parameter=path>.</parameter></function></tool_call>"
    )

    ok, actions = tool.parse_action(response=response)

    assert ok is True
    assert isinstance(actions, list)
    assert len(actions) == 1

    action = actions[0]
    assert action["type"] == "function"
    assert action["function"]["name"] == "list_directory"
    assert action["function"]["arguments"] == '{"path": "."}'
