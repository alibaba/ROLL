"""Synchronous native DCP writes with one tensor staged on CPU at a time.

Megatron's async writer preloads the entire checkpoint even for a synchronous
request. PyTorch's current serial writer also retains written tensors for its
safetensors branch. This torch.save-only writer retains native planning,
serialization, metadata, and collective failure handling without either list.
The staging bound is the largest individual write item, not a fixed byte cap.
"""
import os

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as filesystem
from torch.distributed.checkpoint.planner import WriteItemType


class StreamingFileSystemWriter(filesystem.FileSystemWriter):
    def __init__(self, path):
        super().__init__(path, single_file_per_rank=True, thread_count=1,
                         per_thread_copy_ahead=0, sync_files=True)

    def write_data(self, plan, planner):
        results = []
        if plan.items:
            storage_key = f'{plan.storage_data.prefix}0{filesystem.DEFAULT_SUFFIX}'
            path = self.fs.concat_path(self.path, storage_key)
            with self.fs.create_stream(path, 'wb') as stream:
                for item in plan.items:
                    data = planner.resolve_data(item)
                    try:
                        if item.type != WriteItemType.BYTE_IO:
                            data = data.detach().to(device='cpu', non_blocking=False)
                            if data.untyped_storage().nbytes() != data.nbytes:
                                # Packed optimizer/model buffers expose small
                                # views; torch.save otherwise serializes their
                                # entire backing storage for every write item.
                                data = data.clone()
                        results.append(filesystem._write_item(
                            self.transforms, stream, data, item, storage_key,
                            self.serialization_format))
                    finally:
                        del data
                stream.flush()
                os.fsync(stream.fileno())
        future = torch.futures.Future()
        future.set_result(results)
        return future


def streaming_save_strategy():
    """Keep native MCore key/shard conversion and use synchronous DCP IO.

The import is lazy so the tensor writer can also be exercised on CPU-only
installations without Megatron. Asynchronous requests are intentionally not
supported by this strategy; callers must finish saving before updating weights.
"""
    from megatron.core.dist_checkpointing.strategies.base import SaveShardedStrategy
    from megatron.core.dist_checkpointing.strategies.torch import (
        MCoreSavePlanner,
        _replace_state_dict_keys_with_sharded_keys,
        mcore_to_pyt_state_dict,
    )

    class StreamingSaveShardedStrategy(SaveShardedStrategy):
        def __init__(self):
            super().__init__('torch_dist', 1)

        @property
        def can_handle_sharded_objects(self):
            return True

        def save(self, sharded_state_dict, checkpoint_dir):
            sharded, _, _ = _replace_state_dict_keys_with_sharded_keys(
                sharded_state_dict, keep_only_main_replica=True)
            state = mcore_to_pyt_state_dict(sharded, False)
            dcp.save(state, storage_writer=StreamingFileSystemWriter(checkpoint_dir),
                     planner=MCoreSavePlanner(flatten_state_dict=False,
                                              dedup_replicated_tensors=False))

    return StreamingSaveShardedStrategy()
