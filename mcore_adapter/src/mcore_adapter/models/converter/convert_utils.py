import re
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from megatron.core import mpu
from packaging.version import Version as PkgVersion


if TYPE_CHECKING:
    from torch import Tensor


MCA_LAYER_PREFIX = "decoder.layers."
MCA_MOE_PREFIX = ".mlp.experts.local_experts."
MAX_SHARD_SIZE = 5_000_000_000  # 5GB


def get_layer_index(weight_name: str, prefix: str):
    if not weight_name.startswith(prefix):
        return None
    return int(weight_name.replace(prefix, "").split(".")[0])


def get_weight_prefix(weight_name: str, prefix: str, moe_prefix: str = None):
    if not weight_name.startswith(prefix):
        return ""
    layer_index = get_layer_index(weight_name, prefix)
    layer_prefix = prefix + str(layer_index)
    if moe_prefix is None:
        return layer_prefix
    return layer_prefix + get_weight_prefix(weight_name[len(layer_prefix) :], prefix=moe_prefix)


def remove_weight_prefix(weight_name: str, prefix: str, moe_prefix: str = None):
    weight_prefix = get_weight_prefix(weight_name, prefix, moe_prefix)
    return weight_name.replace(weight_prefix, "", 1)


def get_moe_index(weight_name: str, prefix: str, moe_prefix: str = None):
    if not weight_name.startswith(prefix):
        return None
    mos_layer_name = remove_weight_prefix(weight_name, prefix)
    return get_layer_index(mos_layer_name, moe_prefix)


def add_layer_prefix(
    weight_name: str,
    layer_index: Union[int, str],
    prefix: str,
    moe_index: Union[int, str] = None,
    moe_prefix: str = None,
):
    if not weight_name.startswith("."):
        # not weight in layer
        return weight_name
    if moe_index is not None:
        weight_name = add_layer_prefix(weight_name, moe_index, moe_prefix)
    return prefix + str(layer_index) + weight_name


def convert_to_mca_prefix(weight_prefix: str, prefix: str, moe_prefix: str = None):
    weight_prefix = weight_prefix.replace(prefix, MCA_LAYER_PREFIX, 1)
    if moe_prefix is not None:
        weight_prefix = weight_prefix.replace(moe_prefix, MCA_MOE_PREFIX, 1)
    return weight_prefix


def convert_to_hf_prefix(weight_prefix: str, prefix: str, moe_prefix: str = None):
    weight_prefix = weight_prefix.replace(MCA_LAYER_PREFIX, prefix, 1)
    if moe_prefix is not None:
        weight_prefix = weight_prefix.replace(MCA_MOE_PREFIX, moe_prefix, 1)
    return weight_prefix


def get_mca_layer_index(weight_name: str):
    return get_layer_index(weight_name, MCA_LAYER_PREFIX)


def get_mca_weight_prefix(weight_name: str):
    return get_weight_prefix(weight_name, MCA_LAYER_PREFIX, MCA_MOE_PREFIX)


def remove_mca_weight_prefix(weight_name: str):
    return remove_weight_prefix(weight_name, MCA_LAYER_PREFIX, MCA_MOE_PREFIX)


def get_mca_moe_index(weight_name: str):
    return get_moe_index(weight_name, MCA_LAYER_PREFIX, MCA_MOE_PREFIX)


def add_mca_layer_prefix(weight_name: str, layer_index: Union[int, str], moe_index: Union[int, str] = None):
    return add_layer_prefix(weight_name, layer_index, MCA_LAYER_PREFIX, moe_index, MCA_MOE_PREFIX)


def extract_suffix_number(s):
    match = re.search(r"\d+$", s)
    return match.group() if match else None


def gather_tensor_parallel(tensor: "torch.Tensor", async_op: bool = False):
    group = mpu.get_tensor_model_parallel_group()
    dst = dist.get_process_group_ranks(group)[0]
    if mpu.get_tensor_model_parallel_rank() == 0:
        gathered_tensors = [torch.empty_like(tensor) for _ in range(mpu.get_tensor_model_parallel_world_size())]
    else:
        gathered_tensors = None
    handle = dist.gather(tensor, gathered_tensors, dst=dst, group=group, async_op=async_op)

    if async_op:
        return gathered_tensors, handle
    else:
        return gathered_tensors


def all_gather_tensors(tensor: "torch.Tensor", group, async_op: bool = False):
    gathered_tensors = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group=group))]
    handle = torch.distributed.all_gather(gathered_tensors, tensor, group=group, async_op=async_op)
    if async_op:
        return gathered_tensors, handle
    else:
        return gathered_tensors


def allgather_parallel_objs(obj: Any, group):
    world_size = dist.get_world_size(group)
    gathered_objs = [obj for _ in range(world_size)]
    dist.all_gather_object(gathered_objs, obj, group=group)
    return gathered_objs


@dataclass
class StateDictSplitState:
    filename_to_tensors: Dict[str, List[str]] = field(default_factory=dict)
    tensor_to_filename: Dict[str, str] = field(default_factory=dict)
    total_size: int = 0
    shard_list: List[Dict[str, "Tensor"]] = field(default_factory=list)
    current_shard: Dict[str, "Tensor"] = field(default_factory=dict)
    current_shard_size: int = 0
    max_shard_size: int = MAX_SHARD_SIZE

    @classmethod
    def merge_states(cls, states: List["StateDictSplitState"]):
        filename_to_tensors = {}
        tensor_to_filename = {}
        for state in states:
            assert all(
                file_name not in filename_to_tensors for file_name in state.filename_to_tensors
            ), f"file name conflict {filename_to_tensors} {state.filename_to_tensors}"
            assert all(
                tensor not in tensor_to_filename for tensor in state.tensor_to_filename
            ), f"tensor name conflict {tensor_to_filename} {state.tensor_to_filename}"
            filename_to_tensors.update(state.filename_to_tensors)
            tensor_to_filename.update(state.tensor_to_filename)
        return cls(
            filename_to_tensors=filename_to_tensors,
            tensor_to_filename=tensor_to_filename,
            total_size=sum(state.total_size for state in states),
        )


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


def te_grouped_moe_available():
    try:
        import transformer_engine as te
    except ImportError:
        return False

    def get_te_version():
        def get_te_version_str():
            if hasattr(te, "__version__"):
                return str(te.__version__)
            else:
                return version("transformer-engine")

        return PkgVersion(get_te_version_str())

    return get_te_version() >= PkgVersion("1.9.0.dev0")


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
