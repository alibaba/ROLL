import importlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any


class CapabilityUnavailableError(ImportError):
    pass


def apply_default_attention_config(kwargs: dict[str, Any], *, supports_attention_config: bool) -> None:
    """Preserve an explicit backend opt-out while supporting old vLLM builds."""
    if kwargs.get("attention_config", ... ) is None:
        kwargs.pop("attention_config")
        return
    if supports_attention_config and "attention_config" not in kwargs:
        kwargs["attention_config"] = {"backend": "FLASH_ATTN"}


def _target_module_is_missing(error: ModuleNotFoundError, module_name: str) -> bool:
    return error.name == module_name or module_name.startswith(f"{error.name}.")


def import_attribute(module_names: Sequence[str], attribute: str) -> Any:
    errors = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attribute)
        except AttributeError as error:
            errors.append(f"{module_name}: {error}")
        except ModuleNotFoundError as error:
            if not _target_module_is_missing(error, module_name):
                raise
            errors.append(f"{module_name}: {error}")
    attempted = "; ".join(errors)
    raise CapabilityUnavailableError(f"{attribute} is unavailable; attempted {attempted}")


def module_has_attributes(module_name: str, attributes: Sequence[str]) -> bool:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if not _target_module_is_missing(error, module_name):
            raise
        return False
    return all(hasattr(module, attribute) for attribute in attributes)


async def call_maybe_await(function: Callable[..., Any], *args, **kwargs) -> Any:
    result = function(*args, **kwargs)
    if isinstance(result, Awaitable):
        return await result
    return result


def native_sleep(worker: Any, level: int) -> None:
    worker.sleep(level)


def native_wake_up(worker: Any, tags: list[str]) -> None:
    worker.wake_up(tags)


def native_sleep_owns_buffers(worker: Any) -> bool:
    return hasattr(worker, "_sleep_saved_buffers") and hasattr(
        worker, "_sleep_saved_draft_buffers"
    )


def patch_moe_model_weight_loaders(model: Any) -> None:
    for layer in model.model.layers:
        mlp = layer.mlp
        for name, parameter in mlp.named_parameters():
            if "w13_weight" not in name and "w2_weight" not in name:
                continue
            if getattr(parameter, "roll_skip_patch_moe", False):
                continue

            experts = getattr(mlp, "experts", None)
            if experts is None:
                continue
            loader_owner = getattr(getattr(parameter, "weight_loader", None), "__self__", None)
            if loader_owner is not experts:
                parameter.weight_loader = experts.weight_loader


def process_weights_after_loading(
    model: Any,
    model_config: Any,
    target_device: Any,
    loader_modules: Sequence[str] = ("vllm.model_executor.model_loader.utils",),
    torch_utils_modules: Sequence[str] = (
        "vllm.utils.torch_utils",
        "vllm.model_executor.model_loader.utils",
    ),
) -> bool:
    try:
        process = import_attribute(loader_modules, "process_weights_after_loading")
        set_default_torch_dtype = import_attribute(torch_utils_modules, "set_default_torch_dtype")
    except CapabilityUnavailableError:
        return False
    if not callable(process) or not callable(set_default_torch_dtype):
        return False
    with set_default_torch_dtype(model_config.dtype):
        process(model, model_config, target_device)
    return True
