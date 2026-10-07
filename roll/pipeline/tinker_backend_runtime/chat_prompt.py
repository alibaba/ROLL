"""Pure chat prompt normalization shared by rollout and CPU preflight."""
from __future__ import annotations

import json
from typing import Any


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize_tool_calls_for_chat_template(tool_calls: Any) -> Any:
    if not isinstance(tool_calls, list):
        return tool_calls
    normalized = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized.append(tool_call)
            continue
        call = dict(tool_call)
        function = call.get("function")
        if isinstance(function, dict):
            function = dict(function)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_arguments = None
                if isinstance(parsed_arguments, dict):
                    function["arguments"] = parsed_arguments
            call["function"] = function
        normalized.append(call)
    return normalized


def _normalize_messages_for_chat_template(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        item = dict(message)
        if "content" in item:
            item["content"] = _message_content_to_text(item.get("content"))
        if "tool_calls" in item:
            item["tool_calls"] = _normalize_tool_calls_for_chat_template(item.get("tool_calls"))
        normalized.append(item)
    return normalized


def _messages_to_prompt_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages or "")
    rendered = []
    for message in messages:
        if isinstance(message, dict):
            rendered.append(f"{message.get('role', 'user')}: {_message_content_to_text(message.get('content', ''))}")
        else:
            rendered.append(str(message))
    return "\n".join(rendered)


def _render_prompt_token_ids(
    tokenizer: Any,
    request: dict[str, Any],
    *,
    enable_thinking: bool = False,
) -> list[int]:
    if tokenizer is None:
        raise RuntimeError("structured sample prompt requires a tokenizer")
    messages = _normalize_messages_for_chat_template(request.get("messages"))
    if not isinstance(messages, list):
        raise ValueError("sample prompt.request.messages must be a list")
    kwargs = {
        "tools": request.get("tools"),
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            **kwargs,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"failed to tokenize structured sample prompt: {exc}") from exc
    if token_ids is None:
        raise RuntimeError("tokenizer.apply_chat_template returned None")
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids")
    elif hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if (
        isinstance(token_ids, (list, tuple))
        and len(token_ids) == 1
        and isinstance(token_ids[0], (list, tuple))
    ):
        token_ids = token_ids[0]
    if not isinstance(token_ids, (list, tuple)):
        raise RuntimeError("tokenizer.apply_chat_template did not return input_ids")
    return [int(token_id) for token_id in token_ids]
