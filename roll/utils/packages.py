import importlib.metadata
import importlib.util
from functools import lru_cache

from packaging import version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "0.0.0"


@lru_cache
def is_transformers_version_greater_than(content: str):
    return version.parse(_get_package_version("transformers")) >= version.parse(content)
