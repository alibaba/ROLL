"""Reduce checkpoint seeks while retaining native DCP deserialization semantics."""
from dataclasses import replace
from functools import wraps


def order_checkpoint_reader(reader):
    """Sort a copy of each read plan, without caching tensor payloads.

    Model/optimizer traversal order need not match the writer's physical layout.
    Large MoE checkpoints can have thousands of interleaved chunks per file;
    passing that order through to FileSystemReader creates backwards HDD seeks.
    The native reader still owns transforms, tensor slices, copies and errors.
    """
    original = reader.read_data
    if getattr(original, '_mcore_storage_ordered', False):
        return reader

    @wraps(original)
    def read_data(plan, planner):
        def location(request):
            storage = reader.storage_data[request.storage_index]
            return storage.relative_path, storage.offset

        ordered = replace(plan, items=sorted(plan.items, key=location))
        return original(ordered, planner)

    read_data._mcore_storage_ordered = True
    reader.read_data = read_data
    return reader


def patch_mcore_checkpoint_read_order():
    """Apply ordering only to MCore's ordinary local filesystem readers."""
    from megatron.core.dist_checkpointing.strategies import torch as strategy

    original = getattr(strategy, '_get_filesystem_reader', None)
    if original is None or getattr(original, '_mcore_storage_ordered', False):
        return
    supported = (strategy.FileSystemReader, strategy.CachedMetadataFileSystemReader)

    @wraps(original)
    def get_reader(*args, **kwargs):
        reader = original(*args, **kwargs)
        # Keep external storage clients and custom reader implementations intact.
        if type(reader) in supported:
            order_checkpoint_reader(reader)
        return reader

    get_reader._mcore_storage_ordered = True
    strategy._get_filesystem_reader = get_reader
