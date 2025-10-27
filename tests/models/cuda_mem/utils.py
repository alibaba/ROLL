import torch

from roll.platforms import current_platform


def log_gpu_memory_usage(head: str):
    memory_allocated = current_platform.memory_allocated() / 1024**3
    memory_reserved = current_platform.memory_reserved() / 1024**2
    memory_reserved_max = current_platform.max_memory_reserved() / 1024**3
    message = (
        f"{head}, memory allocated (GB): {memory_allocated}, memory reserved (MB): {memory_reserved}, "
        f"memory max reserved (GB): {memory_reserved_max}"
    )
    print(message)
