import io
import os
import re
import sys
import dataclasses
from typing import Any, Optional

from omegaconf import OmegaConf

def contains_renderable_field(s: str, key: str) -> bool:
    """
    Check whether the string `s` contains a renderable field named `key`.

    Args:
        s: The string to inspect.
        key: Name of the renderable field (e.g., "turn_idx").

    Returns:
        True if `s` contains patterns like `{key}`, `{key:format}`, `{key.attr}`,
        `{key[index]}`, etc.; otherwise False.
    """
    if not isinstance(s, str):
        raise TypeError("Input 's' must be a string.")
    if not isinstance(key, str):
        raise TypeError("Input 'key' must be a string.")

    pattern = r"\{" + re.escape(key) + r"(?!\w).*\}"
    return re.search(pattern, s) is not None


def print_pipeline_config(config_obj: Any, enable_color: bool = False) -> None:
    def convert_to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {f.name: convert_to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, (set, frozenset)):
            try:
                return sorted(list(obj), key=str)
            except TypeError:
                return list(obj)
        if isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_dict(item) for item in obj]
        return obj

    buf = io.StringIO()

    ANSI_RESET = "\033[0m"
    ANSI_MAGENTA = "\033[95m"
    ANSI_CYAN = "\033[96m"
    ANSI_GREEN = "\033[92m"
    ANSI_YELLOW = "\033[93m"
    COLORS_BY_LEVEL = [ANSI_CYAN, ANSI_GREEN, ANSI_YELLOW, ANSI_MAGENTA]

    use_color = enable_color and sys.stdout.isatty() and os.getenv("NO_COLOR") is None

    def wrap(text: str, color: Optional[str]) -> str:
        if use_color and color:
            return f"{color}{text}{ANSI_RESET}"
        return text

    def bprint(s: str, color: Optional[str] = None):
        print(wrap(s, color), file=buf)

    def colorize_yaml(yaml_text: str) -> str:
        colored_lines = []
        for line in yaml_text.splitlines():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            level_color = COLORS_BY_LEVEL[(indent // 2) % len(COLORS_BY_LEVEL)]
            if ":" in stripped:
                key, rest = stripped.split(":", 1)
                rest = rest.rstrip()
                suffix = f": {rest.strip()}" if rest.strip() else ":"
                colored_lines.append(f"{' ' * indent}{wrap(key, level_color)}{suffix}")
            else:
                colored_lines.append(f"{' ' * indent}{wrap(stripped, level_color)}")
        return "\n".join(colored_lines)

    bprint("\n====== Pipeline Config ======", ANSI_MAGENTA)
    bprint("------ merged & post-init ------", ANSI_CYAN)
    config_dict = convert_to_dict(config_obj)
    yaml_text = OmegaConf.to_yaml(OmegaConf.create(config_dict), resolve=True).rstrip()
    bprint(colorize_yaml(yaml_text))
    bprint("====== End Config ======", ANSI_MAGENTA)
    print(buf.getvalue())