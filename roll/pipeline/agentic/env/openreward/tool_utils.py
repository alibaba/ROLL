"""Reusable utilities for OpenReward tool call parsing and system prompt building.

Supports Qwen3.5's **native** tool-call format (``<function=name><parameter=key>...``)
as well as the JSON fallback (``{"name": ..., "arguments": {...}}``).
"""
import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool-spec conversion: OpenReward spec → Qwen chat-template dict
# ---------------------------------------------------------------------------

def openreward_spec_to_qwen_tool(spec: Any) -> Dict[str, Any]:
    """Convert an OpenReward tool spec to the dict format expected by
    ``tokenizer.apply_chat_template(tools=[...])``.

    The Qwen3.5 chat template expects each tool as::

        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Args:
        spec: OpenReward tool spec with ``.name``, ``.input_schema``, ``.description``.

    Returns:
        A dict compatible with the Qwen tokenizer's ``tools`` parameter.
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


# ---------------------------------------------------------------------------
# Tool-call parsing: Qwen native XML + JSON fallback
# ---------------------------------------------------------------------------

# Regex for Qwen3.5 native format: <function=name>...<parameter=key>\nvalue\n</parameter>...
_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<key>[^>]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from model output.

    Supports two formats:

    1. **Qwen3.5 native** (preferred)::

        <tool_call>
        <function=bash>
        <parameter=command>ls</parameter>
        <parameter=description>list files</parameter>
        </function>
        </tool_call>

    2. **JSON fallback** (cookbook style)::

        <tool_call>
        {"name": "bash", "arguments": {"command": "ls", "description": "list files"}}
        </tool_call>

    Args:
        text: Raw model output.

    Returns:
        ``None`` if no ``<tool_call>`` found.
        ``{"type": "success", "name": str, "arguments": dict}`` on success.
        ``{"type": "error", "error": str}`` on parse failure.
    """
    start_tag = "<tool_call>"
    si = text.find(start_tag)
    if si == -1:
        return None

    end_tag = "</tool_call>"
    ei = text.find(end_tag, si)
    inner = text[si + len(start_tag):ei].strip() if ei != -1 else text[si + len(start_tag):].strip()

    if not inner:
        return {"type": "error", "error": "empty tool call block"}

    # --- Try Qwen native XML format first ---
    func_match = _FUNCTION_RE.search(inner)
    if func_match:
        name = func_match.group("name").strip()
        body = func_match.group("body")
        arguments: Dict[str, str] = {}
        for param_match in _PARAMETER_RE.finditer(body):
            key = param_match.group("key").strip()
            value = param_match.group("value").strip()
            arguments[key] = value
        return {"type": "success", "name": name, "arguments": arguments}

    # --- Fallback: JSON format ---
    try:
        data = json.loads(inner)
        if not isinstance(data, dict):
            return {"type": "error", "error": f"parsed value is not a dict: {type(data).__name__}"}
        name = data.get("name")
        if not name:
            return {"type": "error", "error": "missing 'name' field in tool call"}
        args = data.get("arguments", {})
        if not isinstance(args, dict):
            return {"type": "error", "error": f"arguments is not a dict: {type(args).__name__}"}
        return {"type": "success", "name": name, "arguments": args}
    except (json.JSONDecodeError, KeyError) as exc:
        return {"type": "error", "error": str(exc)}


def reduce_rewards(rewards: List[float], method: str) -> float:
    """Reduce a list of per-step rewards to a single scalar."""
    if not rewards:
        return 0.0
    if method == "sum":
        return sum(rewards)
    elif method == "mean":
        return sum(rewards) / len(rewards)
    elif method == "max":
        return max(rewards)
    elif method == "min":
        return min(rewards)
    raise ValueError(f"Unknown reward reduction method: {method!r}")
