from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from torch import Tensor


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


@dataclass
class StackedTensors:
    tensors: Optional[List["torch.Tensor"]]
    dim: int = 0


class TensorBucket:
    def __init__(self, bucket_size, device="cuda"):
        self.buffer = torch.empty(bucket_size, dtype=torch.int8, device=device)
        self.device = device
        self.bucket_size = bucket_size
        self.write_index = 0
        self.tensors_meta = {}

    def push_tensor(self, tensor: "torch.Tensor", tensor_start: int, name: str):
        required_bytes = get_tensor_size(tensor) - tensor_start
        bucket_start = self.write_index
        save_bytes = min(required_bytes, self.bucket_size - bucket_start)
        tensor_bytes = tensor.view(-1).view(torch.int8)
        self.buffer[bucket_start : bucket_start + save_bytes].copy_(
            tensor_bytes[tensor_start : tensor_start + save_bytes]
        )
        self.tensors_meta[name] = {
            "bucket_start": bucket_start,
            "tensor_start": tensor_start,
            "save_bytes": save_bytes,
            "tensor_meta": torch.empty_like(tensor, device="meta"),
        }
        self.write_index += save_bytes
        return save_bytes

    def pop_tensor(self, named_tensors: Dict[str, "torch.Tensor"]):
        named_tensors = self.pop_tensor_in_buffer(named_tensors, self.tensors_meta, self.buffer)
        self.drop()
        return named_tensors

    @staticmethod
    def pop_tensor_in_buffer(named_tensors: Dict[str, "torch.Tensor"], tensors_meta, buffer: "torch.Tensor"):
        for name, meta in tensors_meta.items():
            meta = tensors_meta[name]
            bucket_start, tensor_start, save_bytes = meta["bucket_start"], meta["tensor_start"], meta["save_bytes"]
            tensor = named_tensors.get(name, None)
            if tensor is None:
                tensor = torch.empty_like(meta["tensor_meta"], device=buffer.device)
                named_tensors[name] = tensor
            tensor.view(-1).view(torch.int8)[tensor_start : tensor_start + save_bytes].copy_(
                buffer[bucket_start : bucket_start + save_bytes]
            )
        return named_tensors

    def drop(self):
        self.tensors_meta = {}
        self.write_index = 0

    def is_full(self):
        return self.write_index == self.bucket_size


class SendBucketManager:
    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
        self.bucket = TensorBucket(bucket_size)

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

    @staticmethod
    def meta_to_dict(meta_infos):
        """
        Convert tensor_meta from torch.Tensor of meta device to dict
        """
        for _, meta_info in meta_infos.items():
            t = meta_info["tensor_meta"]
            tensor_meta = {
                "shape": list(t.shape),
                "dtype": t.dtype,
                "layout": t.layout,
                "device": t.device,
            }
            meta_info["tensor_meta"] = tensor_meta


class RecvBucketManager:
    def __init__(self):
        self.waiting_tensors = {}

    def process_bucket(self, tensors_meta, buffer):
        self.waiting_tensors = TensorBucket.pop_tensor_in_buffer(self.waiting_tensors, tensors_meta, buffer)
        finished_tensors = {}
        for name, meta in tensors_meta.items():
            tensor_start, save_bytes = meta["tensor_start"], meta["save_bytes"]
            if tensor_start + save_bytes == get_tensor_size(self.waiting_tensors[name]):
                finished_tensors[name] = self.waiting_tensors.pop(name)
        return finished_tensors

    def clear(self):
        assert len(self.waiting_tensors) == 0

    @staticmethod
    def dict_to_meta(meta_infos):
        for _, meta_info in meta_infos.items():
            tensor_meta = meta_info["tensor_meta"]
            assert tensor_meta["device"] == torch.device("meta")
            t = torch.empty(
                torch.Size(tensor_meta["shape"]),
                dtype=tensor_meta["dtype"],
                layout=tensor_meta["layout"],
                device=tensor_meta["device"],
            )
            meta_info["tensor_meta"] = t
