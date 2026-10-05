"""Reserve room for NCCL's first synchronous gradient communication."""
import torch


class FirstGradSyncCacheRelease:
    """Release unused activation cache once, before NCCL allocates its buffers.

    NCCL uses allocations outside PyTorch's caching allocator. A completed
    backward pass can leave enough cached (but unused) memory to make NCCL's
    first gradient collective fail. Only use this for synchronous gradient
    reduction: overlapped reductions have already begun before finalization.
    """

    def __init__(self, finalize):
        self.finalize = finalize
        self.initialized = False

    def __call__(self, *args, **kwargs):
        if not self.initialized:
            torch.cuda.empty_cache()
        result = self.finalize(*args, **kwargs)
        self.initialized = True
        return result
