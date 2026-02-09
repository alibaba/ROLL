from typing import Dict

import torch
from torch.multiprocessing import reductions

from roll.platforms import current_platform
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer

MAX_SHARD_SIZE = 5_000_000_000  # 5GB

# below is copy from huggingface-hub
SIZE_UNITS = {
    "TB": 10**12,
    "GB": 10**9,
    "MB": 10**6,
    "KB": 10**3,
}


def parse_size_to_int(size_as_str: str) -> int:
    """
    Parse a size expressed as a string with digits and unit (like `"5MB"`) to an integer (in bytes).

    Supported units are "TB", "GB", "MB", "KB".

    Args:
        size_as_str (`str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> parse_size_to_int("5MB")
    5000000
    ```
    """
    size_as_str = size_as_str.strip()

    # Parse unit
    unit = size_as_str[-2:].upper()
    if unit not in SIZE_UNITS:
        raise ValueError(f"Unit '{unit}' not supported. Supported units are TB, GB, MB, KB. Got '{size_as_str}'.")
    multiplier = SIZE_UNITS[unit]

    # Parse value
    try:
        value = float(size_as_str[:-2].strip())
    except ValueError as e:
        raise ValueError(f"Could not parse the size value from '{size_as_str}': {e}") from e

    return int(value * multiplier)


def get_tensor_size(tensor: "torch.Tensor") -> int:
    return tensor.numel() * tensor.element_size()


class TensorBucket:
    def __init__(self, bucket_size, device="cuda"):
        self.buffer = torch.empty(bucket_size, dtype=torch.int8, device=device)
        self.device = device
        self.bucket_size = bucket_size
        self.write_index = 0
        self.tensors_meta = []

    def push_tensor(self, tensor: "torch.Tensor", tensor_start: int, name: str):
        required_bytes = get_tensor_size(tensor) - tensor_start
        bucket_start = self.write_index
        save_bytes = min(required_bytes, self.bucket_size - bucket_start)
        tensor_bytes = tensor.view(-1).view(torch.int8)
        self.buffer[bucket_start : bucket_start + save_bytes].copy_(
            tensor_bytes[tensor_start : tensor_start + save_bytes]
        )
        self.tensors_meta.append(
            {
                "name": name,
                "bucket_start": bucket_start,
                "tensor_start": tensor_start,
                "save_bytes": save_bytes,
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
            }
        )
        self.write_index += save_bytes
        return save_bytes

    def pop_tensor(self, named_tensors: Dict[str, "torch.Tensor"]):
        named_tensors = self.pop_tensor_in_buffer(named_tensors, self.tensors_meta, self.buffer)
        self.drop()
        return named_tensors

    @staticmethod
    def pop_tensor_in_buffer(named_tensors: Dict[str, "torch.Tensor"], tensors_meta, buffer: "torch.Tensor"):
        for meta in tensors_meta:
            name = meta["name"]
            bucket_start, tensor_start, save_bytes = meta["bucket_start"], meta["tensor_start"], meta["save_bytes"]
            tensor = named_tensors.get(name, None)
            if tensor is None:
                tensor = torch.empty(
                    torch.Size(meta["shape"]),
                    dtype=meta["dtype"],
                    device=buffer.device,
                )
                named_tensors[name] = tensor
            tensor.view(-1).view(torch.int8)[tensor_start : tensor_start + save_bytes].copy_(
                buffer[bucket_start : bucket_start + save_bytes]
            )
        return named_tensors

    def drop(self):
        self.tensors_meta = []
        self.write_index = 0

    def is_full(self):
        return self.write_index == self.bucket_size


class SendBucketManager:
    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
        self.bucket = TensorBucket(bucket_size, current_platform.device_type)

    def push_tensor(self, tensor: "torch.Tensor", name: str):
        tensor_start = 0
        required_bytes = get_tensor_size(tensor)
        while tensor_start < required_bytes:
            save_bytes = self.bucket.push_tensor(tensor, tensor_start, name)
            tensor_start += save_bytes
            if self.bucket.is_full():
                yield self.bucket.tensors_meta, self.bucket.buffer
                self.bucket.drop()

    def pop_last_bucket(self):
        if self.bucket.write_index > 0:
            return self.bucket.tensors_meta, self.bucket.buffer
        return None, None


class RecvBucketManager:
    def __init__(self):
        self.waiting_tensors = {}

    def process_bucket(self, tensors_meta, buffer):
        self.waiting_tensors = TensorBucket.pop_tensor_in_buffer(self.waiting_tensors, tensors_meta, buffer)
        finished_tensors = {}
        for meta in tensors_meta:
            name = meta["name"]
            tensor_start, save_bytes = meta["tensor_start"], meta["save_bytes"]
            if tensor_start + save_bytes == get_tensor_size(self.waiting_tensors[name]):
                finished_tensors[name] = self.waiting_tensors.pop(name)
        return finished_tensors

    def clear(self):
        assert len(self.waiting_tensors) == 0


# ref: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/patch_torch.py
def monkey_patch_torch_reductions():
    """Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed"""

    # Currently, NPU does not support UUID. This has been temporarily commented out, with support expected in the fourth quarter.
    if current_platform.device_type == "npu":
        return

    if hasattr(reductions, "_reduce_tensor_original"):
        return

    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified

    reductions.init_reductions()


_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def _reduce_tensor_modified(*args, **kwargs):
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    output_args = _modify_tuple(output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid)
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def _device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(device_maybe_uuid) -> int:
    if isinstance(device_maybe_uuid, int):
        return device_maybe_uuid

    if isinstance(device_maybe_uuid, str):
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        raise Exception("Invalid device_uuid=" + device_maybe_uuid)

    raise Exception(f"Unknown type: {device_maybe_uuid=}")


def _modify_tuple(t, index: int, modifier):
    return *t[:index], modifier(t[index]), *t[index + 1 :]


def _bucket_named_tensors(named_tensors: list[tuple[str, torch.Tensor]]) -> tuple[torch.Tensor, list[dict]]:
    if not named_tensors:
        raise ValueError("Cannot create empty tensor bucket")

    tensors_meta = []
    flattened_tensors = []

    current_idx = 0
    for i, (name, tensor) in enumerate(named_tensors):
        flattened = tensor.flatten().view(torch.int8)

        numel = flattened.numel()
        metadata = {
            "name": name,
            "shape": list(tensor.shape),  # Convert to list for serialization
            "dtype": tensor.dtype,
            "start_idx": current_idx,
            "end_idx": current_idx + numel,
            "numel": numel,
        }
        tensors_meta.append(metadata)
        flattened_tensors.append(flattened)
        current_idx += numel

    flattened_tensor = torch.cat(flattened_tensors, dim=0)
    return flattened_tensor, tensors_meta


def named_tensors_from_bucket(bucket: "torch.Tensor", tensors_meta: list[dict]) -> list[tuple[str, torch.Tensor]]:
    reconstructed = []
    for i, meta in enumerate(tensors_meta):
        tensor = bucket[meta["start_idx"] : meta["end_idx"]].view(meta["dtype"]).reshape(torch.Size(meta["shape"]))
        reconstructed.append((meta["name"], tensor))
    return reconstructed


def serialize_named_weights(named_weights: list[tuple[str, torch.Tensor]], infer_strategy: str):
    if infer_strategy == "sglang":
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        try:
            from sglang.srt.utils.patch_torch import (
                monkey_patch_torch_reductions as sglang_monkey_patch_torch_reductions,
            )  # type: ignore
        except ImportError:
            from sglang.srt.patch_torch import (
                monkey_patch_torch_reductions as sglang_monkey_patch_torch_reductions,
            )  # type: ignore

        sglang_monkey_patch_torch_reductions()
        bucket = FlattenedTensorBucket(named_weights)
        flattened_tensor_data = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized_tensors = MultiprocessingSerializer.serialize(flattened_tensor_data)
        return serialized_tensors

    bucket, tensors_meta = _bucket_named_tensors(named_weights)

    # PumpkinComment:
    # FSDP2 will fail if using CPUOffload Policy without this check
    if not getattr(bucket, "is_cuda", False):
        bucket = bucket.to(current_platform.device_type).contiguous()

    monkey_patch_torch_reductions()

    serialized_tensors = MultiprocessingSerializer.serialize({"bucket": bucket, "tensors_meta": tensors_meta})
    return serialized_tensors
