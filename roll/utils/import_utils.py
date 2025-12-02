import importlib
from importlib.util import find_spec
from typing import Any, Optional
import traceback

from roll.utils.logging import get_logger


logger = get_logger()


def is_vllm_available() -> bool:
    return find_spec("vllm") is not None


def can_import_class(class_path: str) -> bool:
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        getattr(module, class_name)
        return True
    except Exception as e:
        logger.error(f"Failed to import class {class_path}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def safe_import_class(class_path: str) -> Optional[Any]:
    if can_import_class(class_path):
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    else:
        return None