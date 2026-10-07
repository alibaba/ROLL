"""Qwen3.5 native XML tool responses use the real optional SGlang parser."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _enrich(text, tools, *, tokens=None):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    pipeline = module.TinkerBackendRuntimePipeline.__new__(module.TinkerBackendRuntimePipeline)
    pipeline.args = SimpleNamespace(runtime_config={"tool_call_parser": "qwen3_coder"})
    pipeline.roll_backend = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=248046, eos_token="<|im_end|>")
    )
    sampled_tokens = [101, 248046] if tokens is None else tokens
    sequence = {"tokens": sampled_tokens, "logprobs": [-0.125] * len(sampled_tokens),
                "text": text, "raw_text": text, "stop_reason": "stop"}
    payload = {"prompt": module.openai_chat_completion_request({
        "messages": [{"role": "user", "content": "Run the public test command."}],
        "tools": tools,
    })}
    result = pipeline._enrich_sample_response(payload, {"sequences": [sequence]},
                                             prompt_token_ids=[11, 12])
    assert result["prompt_token_ids"] == [11, 12]
    assert sequence["tokens"] == sampled_tokens
    assert sequence["output_token_ids"] == sampled_tokens
    assert sequence["logprobs"] == [-0.125] * len(sampled_tokens)
    assert sequence["raw_text"] == text
    return sequence


def _tool(name, properties):
    return {"type": "function", "function": {"name": name, "description": "Public tool",
            "parameters": {"type": "object", "properties": properties}}}


def test_qwen35_xml_bash_preserves_multiline_command_and_sampled_evidence():
    command = 'printf "a & b < c > d"\npython -c "print(1 + 2)"'
    text = ('<tool_call>\n<function=bash>\n<parameter=command>\n' + command +
            '\n</parameter>\n</function>\n</tool_call><|im_end|>\n')
    sequence = _enrich(text, [_tool("bash", {"command": {"type": "string"}})])
    assert sequence["finish_reason"] == "tool_calls"
    assert len(sequence["tool_calls"]) == 1
    call = sequence["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": command}
    assert "<|im_end|>" not in sequence["text"]


def test_qwen35_xml_parameters_follow_tool_schema_types():
    text = ('<tool_call><function=inspect><parameter=count>3</parameter>'
            '<parameter=enabled>true</parameter><parameter=items>[1, "two"]</parameter>'
            '<parameter=options>{"depth": 2}</parameter></function></tool_call><|im_end|>')
    tools = [_tool("inspect", {"count": {"type": "integer"}, "enabled": {"type": "boolean"},
                              "items": {"type": "array"}, "options": {"type": "object"}})]
    sequence = _enrich(text, tools)
    assert len(sequence["tool_calls"]) == 1
    assert json.loads(sequence["tool_calls"][0]["function"]["arguments"]) == {
        "count": 3, "enabled": True, "items": [1, "two"], "options": {"depth": 2}}


@pytest.mark.parametrize("tools", [[], [_tool("bash", {"command": {"type": "string"}})]])
def test_qwen35_unadvertised_tool_is_not_executable(tools):
    text = ('<tool_call><function=not_advertised><parameter=command>id</parameter>'
            '</function></tool_call><|im_end|>')
    sequence = _enrich(text, tools)
    assert not sequence["tool_calls"]
    assert sequence["finish_reason"] == "stop"


def test_qwen35_eos_content_requires_matching_sampled_terminal_token():
    text = "A literal marker <|im_end|>\n"
    sequence = _enrich(text, [], tokens=[101, 102])
    assert sequence["text"] == text
