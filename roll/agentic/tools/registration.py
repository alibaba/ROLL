import importlib
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Union


@dataclass
class ToolSpec:
    """A specification for creating Tools."""

    id: str
    entry_point: Union[Callable, str]
    kwargs: Dict[str, Any] = field(default_factory=dict)


TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tools(tool_id: str, entry_point: Union[Callable, str], **kwargs: Any):
    """Register a tool with a given ID."""
    if tool_id in TOOL_REGISTRY:
        raise ValueError(f"Tool {tool_id} already registered.")
    TOOL_REGISTRY[tool_id] = ToolSpec(id=tool_id, entry_point=entry_point, kwargs=kwargs)


def make_tool(tool_id: str, **kwargs) -> Any:
    """Create an instance of a registered tool."""
    if tool_id not in TOOL_REGISTRY:
        raise ValueError(f"Tool {tool_id} not found in registry.")

    tool_spec = TOOL_REGISTRY[tool_id]

    if isinstance(tool_spec.entry_point, str):
        module_path, class_name = tool_spec.entry_point.split(":")
        try:
            module = importlib.import_module(module_path)
            tool_class: Callable = getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(
                f"Could not import {module_path}.{class_name}. Error: {e}"
            )
    else:
        tool_class: Callable = tool_spec.entry_point

    return tool_class(**{**tool_spec.kwargs, **kwargs})


def print_tools():
    """Print all registered tools."""
    if not TOOL_REGISTRY:
        print("No tools registered.")
    else:
        print("Detailed Registered Tools:")
        for tool_id, tool_spec in TOOL_REGISTRY.items():
            print(f"  - {tool_id}:")
            print(f"      Entry Point: {tool_spec.entry_point}")
            print(f"      Kwargs: {tool_spec.kwargs}")

