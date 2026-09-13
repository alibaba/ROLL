"""
ref: https://github.com/volcengine/verl/blob/main/verl/protocol.py
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import copy
import os
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Union, Set

import numpy as np
import ray
import tensordict
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from codetiming import Timer

from roll.distributed.scheduler.remote_protocol import RemoteBatch, BatchProxy
from roll.distributed.scheduler import transfer_backend
from roll.utils.functionals import union_two_dict, divide_by_chunk_size
from roll.platforms import current_platform
from roll.utils.logging import get_logger

logger = get_logger()

try:
    tensordict.set_lazy_legacy(False).set()
except:
    pass


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert (
        tensor_dict1.batch_size == tensor_dict2.batch_size
    ), f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            # Compare values - handle both tensors and NonTensorStack
            val1, val2 = tensor_dict1[key], tensor_dict2[key]
            assert type(val1) == type(val2), f"{key} has different types: {type(val1)} vs {type(val2)}"
            if isinstance(val1, torch.Tensor):
                assert val1.equal(val2), f"{key} in tensor_dict1 and tensor_dict2 are not the same"
            # For NonTensorStack and other types, skip equality check
            # (comparison would require iterating and checking each element)

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[np.ndarray], tensor_dict2: dict[np.ndarray]) -> dict[np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            assert np.all(
                tensor_dict2[key] == tensor_dict1[key]
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    """
    Convert a list of dictionaries into a dictionary of lists.

    Example:
        Input:  [{"a": 1, "b": 2}, {"a": 3}, {"b": 4}]
        Output: {"a": [1, 3], "b": [2, 4]}

    Only keys present in each dictionary are aggregated.
    Missing keys in a dictionary are simply skipped.
    """
    if not list_of_dict:
        return {}

    output = {}
    for d in list_of_dict:
        if d is None:
            continue
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, but got {type(d)}: {d}")
        for k, v in d.items():
            output.setdefault(k, []).append(v)

    return output


def collate_fn(x: list["DataProto"]):
    meta_info = x[-1].meta_info
    data = DataProto.concat(x)
    data.meta_info = meta_info
    return data

def move_tensors_to_device(data, device):
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = move_tensors_to_device(val, device)
    elif isinstance(data, list):
        for index, val in enumerate(data):
            data[index] = move_tensors_to_device(val, device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def custom_np_concatenate(val):
    concatenated_list = []
    for array in val:
        concatenated_list.extend(array)
    concatenated_array = np.empty(len(concatenated_list), dtype=object)
    concatenated_array[:] = concatenated_list
    return concatenated_array


class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    def __init__(
        self,
        batch: TensorDict = None,
        non_tensor_batch: Dict = None,
        remote_batch: RemoteBatch = None,
        meta_info: Dict = None,
    ):
        if batch is None and remote_batch is not None:
            batch = TensorDict({}, batch_size=[len(remote_batch)])
        self._batch = batch
        self._non_tensor_batch = non_tensor_batch if non_tensor_batch is not None else {}
        self._remote_batch = remote_batch
        self.meta_info = meta_info if meta_info is not None else {}
        self.__post_init__()

    @property
    def batch(self) -> "BatchProxy":
        """Hook: called before accessing batch.
        Returns a BatchProxy that supports fallback lookup to remote_batch."""
        return BatchProxy(self._batch, self._remote_batch, len(self))

    @batch.setter
    def batch(self, value: TensorDict | BatchProxy):
        assert isinstance(value, (TensorDict, BatchProxy))
        value = value.copy()
        if self._remote_batch is not None:
            for key in value.keys():
                if key in self._remote_batch:
                    del self._remote_batch[key]
        for key in self._non_tensor_batch.keys():
            if key in value:
                del value[key]
        if isinstance(value, BatchProxy):
            if self._remote_batch is not None:
                self._remote_batch.union(value._remote_batch)
            else:
                self._remote_batch = value._remote_batch
            self._batch = value._batch
        else:
            self._batch = value
        self.check_consistency()

    @property
    def non_tensor_batch(self) -> "BatchProxy":
        """Hook: called before accessing non_tensor_batch.
        Returns a BatchProxy that supports fallback lookup to remote_batch."""
        return BatchProxy(self._non_tensor_batch, self._remote_batch, len(self))

    @non_tensor_batch.setter
    def non_tensor_batch(self, value: dict | BatchProxy):
        assert isinstance(value, (dict, BatchProxy))
        value = value.copy()
        if self._remote_batch is not None:
            for key in value.keys():
                if key in self._remote_batch:
                    del self._remote_batch[key]
        for key in self._batch.keys():
            if key in value:
                del value[key]
        if isinstance(value, BatchProxy):
            if self._remote_batch is not None:
                self._remote_batch.union(value._remote_batch)
            else:
                self._remote_batch = value._remote_batch
            self._non_tensor_batch = value._batch
        else:
            self._non_tensor_batch = value
        self.check_consistency()

    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()

        if self._batch is not None and current_platform.is_npu():
            for key, val in self._batch.items():
                if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
                    logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                    self._batch[key] = val.to(torch.int32)

        assert self._remote_batch is None or not current_platform.is_npu()

    def __repr__(self) -> str:
        return f"DataProto(batch={self._batch}, non_tensor_batch={self._non_tensor_batch}, remote_batch={self._remote_batch}, meta_info={self.meta_info})"

    def __len__(self):
        if self._batch is not None:
            return len(self._batch)
        if self._non_tensor_batch:
            return len(next(iter(self._non_tensor_batch.values())))
        if self._remote_batch is not None:
            return len(self._remote_batch)
        return 0

    def __getitem__(self, item):
        """
        Enhanced indexing for DataProto objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices
                - str: A key to look up in batch or remote_batch

        Returns:
            DataProto: For slice/list/array/tensor/int indexing.
            torch.Tensor | np.ndarray: For string key lookup.
        """
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)
        elif isinstance(item, (int, np.integer)):
            return self.slice(item, item + 1, 1)
        elif isinstance(item, str):
            # Search batch first, then remote_batch
            if self._batch is not None and item in self._batch.keys():
                return self._batch[item]
            elif item in self._non_tensor_batch.keys():
                return self._non_tensor_batch[item]
            elif self._remote_batch is not None and item in self._remote_batch:
                return self._remote_batch[item]
            else:
                raise KeyError(f"Key '{item}' not found in batch or remote_batch")
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __setitem__(self, key: str, value) -> None:
        if isinstance(key, str):
            if self._remote_batch is not None and key in self._remote_batch:
                del self._remote_batch[key]
            if isinstance(value, torch.Tensor):
                if key in self._non_tensor_batch:
                    del self._non_tensor_batch[key]
                self._batch[key] = value
            elif isinstance(value, np.ndarray):
                if self._batch is not None and key in self._batch.keys():
                    del self._batch[key]
                self._non_tensor_batch[key] = value
            else:
                raise TypeError(f"Unsupported type for value: {type(value)}")
        else:
            raise TypeError(f"Key must be str, got {type(key)}")

    def __delitem__(self, key: str) -> None:
        """Delete key from batch or remote_batch."""
        if isinstance(key, str):
            if self._batch is not None and key in self._batch:
                del self._batch[key]
            elif key in self._non_tensor_batch:
                del self._non_tensor_batch[key]
            elif self._remote_batch is not None and key in self._remote_batch:
                del self._remote_batch[key]
            else:
                raise KeyError(f"Key '{key}' not found")
        else:
            raise TypeError(f"Key must be str, got {type(key)}")

    def __getstate__(self):
        import io

        buffer = io.BytesIO()
        if tensordict.__version__ >= "0.5.0" and self._batch is not None:
            self._batch = self._batch.contiguous()
            self._batch = self._batch.consolidate()
        torch.save(self._batch, buffer)
        return buffer, self._non_tensor_batch, self._remote_batch, self.meta_info

    def __setstate__(self, data):
        batch_deserialized, non_tensor_batch, remote_batch, meta_info = data
        batch_deserialized.seek(0)
        batch = torch.load(
            batch_deserialized, weights_only=False, map_location="cpu" if not current_platform.is_available() else None
        )
        self._batch = batch
        self._non_tensor_batch = non_tensor_batch
        self._remote_batch = remote_batch
        self.meta_info = meta_info

    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        existing_keys = set()

        if self._batch is not None:
            assert len(self._batch.batch_size) == 1, "only support num_batch_dims=1"
            existing_keys.update(self._batch.keys())

        if len(self._non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self._batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            assert existing_keys.isdisjoint(self._non_tensor_batch.keys()), "batch and non_tensor_batch cannot have overlapping keys"
            existing_keys.update(self._non_tensor_batch.keys())

            batch_size = self._batch.batch_size[0]
            for key, val in self._non_tensor_batch.items():
                assert (
                    isinstance(val, np.ndarray) and val.dtype == object
                ), "data in the non_tensor_batch must be a numpy.array with dtype=object"
                assert (
                    val.shape[0] == batch_size
                ), f"key {key} length {len(val)} is not equal to batch size {batch_size}"

        if self._remote_batch is not None:
            assert existing_keys.isdisjoint(
                self._remote_batch.keys()
            ), f"batch and remote_batch cannot have overlapping keys {existing_keys} {self._remote_batch.keys()}"
            if self._batch is not None:
                assert (
                    len(self._remote_batch) == self._batch.batch_size[0]
                ), f"remote_batch length {len(self._remote_batch)} is not equal to batch size {self._batch.batch_size[0]}"

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None):
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if current_platform.is_npu() and val.dtype == torch.int64:
                    logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                    val = val.to(torch.int32)
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1):
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """
        assert len(tensors) > 0, "tensors must not be empty"
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert (
                    batch_size == current_batch
                ), f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}"

        for key, val in non_tensors.items():
            non_tensors[key] = np.empty(len(val), dtype=object)
            non_tensors[key][:] = val

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> "DataProto":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self._batch is not None:
            self._batch = self._batch.to(device)
        if self._remote_batch is not None:
            self._remote_batch = self._remote_batch.to(device)
        if self.meta_info is not None:
            self.meta_info = move_tensors_to_device(self.meta_info, device)

        return self

    def clone(self) -> "DataProto":
        """
        Create a deep copy of this DataProto, including tensors,
        non-tensor data, and meta_info.

        The new DataProto will share no underlying storage with the original.

        Returns:
            DataProto: A new DataProto instance with the same content but
                       independent memory.
        """
        # Copy batch
        batch_copy = self._batch.clone() if self._batch is not None else None

        # Copy non-tensor objects (numpy arrays)
        non_tensor_copy = {k: np.copy(v) for k, v in self._non_tensor_batch.items()}

        remote_batch_copy = self._remote_batch.clone() if self._remote_batch is not None else None

        # Deep copy meta_info to avoid shared mutable objects
        meta_copy = copy.deepcopy(self.meta_info)

        # Return new DataProto instance
        return DataProto(
            batch=batch_copy,
            non_tensor_batch=non_tensor_copy,
            remote_batch=remote_batch_copy,
            meta_info=meta_copy
        )

    def update(self, other: dict):
        assert isinstance(other, dict)
        for key, value in other.items():
            if self._remote_batch is not None and key in self._remote_batch:
                del self._remote_batch[key]
            if isinstance(value, torch.Tensor):
                assert self._batch is not None
                if key in self._non_tensor_batch:
                    del self._non_tensor_batch[key]
                self._batch[key] = value
            elif isinstance(value, np.ndarray):
                if self._batch is not None and key in self._batch.keys():
                    del self._batch[key]
                self._non_tensor_batch[key] = value
            else:
                raise TypeError(f"Unsupported type {type(value)} for key '{key}': expected torch.Tensor or np.ndarray")

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        assert set(batch_keys).isdisjoint(non_tensor_batch_keys), "batch_keys and non_tensor_batch_keys cannot be overlapping"

        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self._batch.select(*batch_keys)
        else:
            batch_keys = []
            sub_batch = self._batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self._non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch_keys = []
            non_tensor_batch = self._non_tensor_batch

        if self._remote_batch is not None:
            assert not deepcopy, "remote_batch deepcopy is not supported yet"
            # FIXME: The behavior of select is changed when batch_keys or non_tensor_batch_keys is None.
            sub_remote_batch = self._remote_batch.select(batch_keys + non_tensor_batch_keys)
        else:
            sub_remote_batch = None

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)
            sub_remote_batch = sub_remote_batch.clone()
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(
            batch=sub_batch, non_tensor_batch=non_tensor_batch, remote_batch=sub_remote_batch, meta_info=sub_meta_info
        )

    def select_idxs(self, idxs):
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        batch_size = idxs_np.sum() if idxs_np.dtype == bool else idxs_np.shape[0]

        if self._batch is not None:
            # Use TensorDict's built-in indexing capabilities
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self._batch.items()}, batch_size=(batch_size,)
            )
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self._non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        if self._remote_batch is not None:
            remote_batch = self._remote_batch.select_idxs(idxs_torch)
        else:
            remote_batch = None

        return type(self)(
            batch=selected_batch,
            non_tensor_batch=selected_non_tensor,
            remote_batch=remote_batch,
            meta_info=self.meta_info,
        )

    def slice(self, start=None, end=None, step=None):
        """
        Slice the DataProto and return a new DataProto object.

        Args:
            start (int, optional): Start index. Defaults to None (start from beginning).
            end (int, optional): End index (exclusive). Defaults to None (go to end).
            step (int, optional): Step size. Defaults to None (step=1).

        Returns:
            DataProto: A new DataProto containing the sliced data

        Examples:
            # Using the slice method directly
            sliced_data = data_proto.slice(10, 20)

            # Using enhanced indexing (returns DataProto)
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # Every other element

            # Using list indexing (returns DataProto)
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # Single index returns DataProto too
            single_item = data_proto[5]
        """
        # Create a slice object
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self._batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self._batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self._non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        if self._remote_batch is not None:
            remote_batch = self._remote_batch[slice_obj]
        else:
            remote_batch = None

        # Return a new DataProto object
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, remote_batch=remote_batch, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        assert batch_keys is not None
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []
        assert set(batch_keys).isdisjoint(non_tensor_batch_keys), "batch_keys and non_tensor_batch_keys cannot be overlapping"
        batch_keys = self.validate_input(batch_keys)
        non_tensor_batch_keys = self.validate_input(non_tensor_batch_keys)
        meta_info_keys = self.validate_input(meta_info_keys)

        remote_batch_keys = set()

        tensors = {}
        for key in batch_keys:
            if key not in self._batch.keys():
                remote_batch_keys.add(key)
            else:
                tensors[key] = self._batch.pop(key)
        tensors = TensorDict(tensors, batch_size=len(self))

        non_tensors = {}
        for key in non_tensor_batch_keys:
            if key not in self._non_tensor_batch.keys():
                remote_batch_keys.add(key)
            else:
                non_tensors[key] = self._non_tensor_batch.pop(key)

        remote_batch = self._remote_batch.pop(remote_batch_keys) if self._remote_batch else None

        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)

        return DataProto(
            batch=tensors,
            non_tensor_batch=non_tensors,
            remote_batch=remote_batch,
            meta_info=meta_info,
        )

    @staticmethod
    def validate_input(keys):
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            elif isinstance(keys, list):
                pass
            else:
                raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
        return keys

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        Note that this function only rename the key in the batch

        WARNING: Rename will materialize the remote batch if the old key is in the remote batch.
        """

        old_keys = self.validate_input(old_keys)
        new_keys = self.validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        if self._remote_batch is None:
            for old_key, new_key in zip(old_keys, new_keys):
                self._batch.rename_key_(old_key, new_key)
            self.check_consistency()
            return self
        else:
            logger.warning(f"RemoteBatch renaming keys {old_keys} to {new_keys} is not efficient for remote data")
            if self._batch is None:
                self._batch = TensorDict({}, batch_size=len(self))

            local_old_keys = []
            local_new_keys = []
            remote_old_keys = []
            remote_new_keys = []
            for old_key, new_key in zip(old_keys, new_keys):
                if old_key in self._batch:
                    assert old_key not in self._remote_batch
                    local_old_keys.append(old_key)
                    local_new_keys.append(new_key)
                elif old_key in self._remote_batch:
                    remote_old_keys.append(old_key)
                    remote_new_keys.append(new_key)
                else:
                    raise KeyError(f"{old_key} not in batch")

            if local_old_keys:
                for old_key, new_key in zip(local_old_keys, local_new_keys):
                    self._batch.rename_key_(old_key, new_key)

            self._remote_batch.materialize(remote_old_keys)
            for old_key, new_key in zip(remote_old_keys, remote_new_keys):
                self._batch[new_key] = self._remote_batch[old_key]
                del self._remote_batch[old_key]

            self.check_consistency()
            return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if
        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        if self._batch is not None and other._batch is not None:
            self._batch = union_tensor_dict(self._batch, other._batch)
        elif other._batch is not None:
            self._batch = TensorDict(other._batch.to_dict(), batch_size=other._batch.batch_size)

        self._non_tensor_batch = union_numpy_dict(self._non_tensor_batch, other._non_tensor_batch)

        if self._remote_batch is not None and other._remote_batch is not None:
            self._remote_batch = self._remote_batch.union(other._remote_batch)
        elif self._remote_batch is None:
            self._remote_batch = other._remote_batch
        if self._remote_batch is not None:
            existing_keys = set(self._batch.keys() if self._batch is not None else []) | set(self._non_tensor_batch.keys())
            for key in existing_keys:
                # use local batch as golden source when key conflict
                if key in self._remote_batch:
                    del self._remote_batch[key]

        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        self.check_consistency()
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.

        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs: internally, it returns a DataLoader over the batch.
                The dataloader_kwargs is the kwargs passed to the DataLoader

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
            ``self._batch.batch_size * epochs // mini_batch_size``
        """
        assert self._batch.batch_size[0] % mini_batch_size == 0, f"{self._batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        # FIXME do not materialize all fields of remote batch
        if self._remote_batch is not None:
            self._remote_batch.materialize()

        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.
        要求:
            batch_size > chunks，调用方保证，此处保证每个chunk会返回一个DataProto

        np.array_split(val, chunks) 和 self._batch.chunk(chunks=chunks, dim=0) 在不能均分时行为不同
        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        chunks_sizes = None
        if len(self) > 0:
            assert len(self) >= chunks, f"batch_size {self._batch.batch_size[0]} < chunks {chunks}"
            index_array = np.arange(len(self))
            chunks_sizes = [len(b) for b in np.array_split(index_array, chunks)]

        if self._batch is not None:
            batch_lst = divide_by_chunk_size(self._batch, chunk_sizes=chunks_sizes)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self._non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            non_tensor_lst = divide_by_chunk_size(val, chunk_sizes=chunks_sizes)
            assert len(non_tensor_lst) == chunks, f"len(non_tensor_lst) {len(non_tensor_lst)} != chunks {chunks}"
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        if self._remote_batch:
            remote_batch_lst = self._remote_batch.chunk(chunks_sizes)
        else:
            remote_batch_lst = [None for _ in range(chunks)]

        output = []
        for i in range(chunks):
            output.append(
                DataProto(
                    batch=batch_lst[i].clone() if batch_lst[i] is not None else batch_lst[i],
                    non_tensor_batch=non_tensor_batch_lst[i],
                    remote_batch=remote_batch_lst[i],
                    meta_info=self.meta_info,
                )
            )

        return output

    @staticmethod
    def concat(
            data: List["DataProto"],
            *,
            global_keys: Optional[Set[str]] = None,
    ) -> "DataProto":
        """
        Concatenate a list of DataProto objects.

        Parameters
        ----------
        data : List[DataProto]
            List of DataProto instances to be concatenated.
        global_keys : Set[str], optional
            Keys in `meta_info` that should be **aggregated across ranks**.
            - If the value is a dict, each sub-key is concatenated across ranks.
            - Otherwise, values are collected into a list.
            Keys not listed retain only the value from rank 0.

        Returns
        -------
        DataProto
            A new DataProto with concatenated tensors, non-tensor data,
            and processed meta information.
        """
        if len(data) == 1:
            return data[0]

        global_keys = global_keys if global_keys is not None else {"metrics"}

        # ---------- 1. Concatenate tensor / non-tensor batches ----------
        batch_lst = [d._batch for d in data if d._batch is not None]
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst else None

        non_tensor_batch = list_of_dict_to_dict_of_list(
            [d._non_tensor_batch for d in data]
        )
        for k, v in non_tensor_batch.items():
            non_tensor_batch[k] = custom_np_concatenate(v)

        remote_batch_list = [d._remote_batch for d in data if d._remote_batch is not None]
        remote_batch = RemoteBatch.cat(remote_batch_list) if remote_batch_list else None

        # ---------- 2. Aggregate meta information ----------
        merged_meta = dict(data[0].meta_info)  # start with rank-0 values

        for key in global_keys:
            # Check if any data has this key, not just the first one
            has_key = any(key in d.meta_info and d.meta_info[key] is not None for d in data)
            if not has_key:
                continue

            values = [d.meta_info.get(key) for d in data]

            # Determine the type from first non-None value
            first_non_none_value = next((v for v in values if v is not None), None)

            # Case 1: dict — aggregate each sub-key across ranks
            if isinstance(first_non_none_value, dict):
                sub_dict = list_of_dict_to_dict_of_list(values)
                for sub_key, sub_list in sub_dict.items():
                    try:
                        if np.isscalar(sub_list[0]):
                            sub_dict[sub_key] = np.array(sub_list).tolist()
                        else:
                            sub_dict[sub_key] = np.concatenate(sub_list, axis=0).tolist()
                    except Exception:
                        # fallback: keep as list
                        sub_dict[sub_key] = sub_list
                merged_meta[key] = sub_dict

            # Case 2: non-dict — collect into list
            else:
                merged_meta[key] = values

        return DataProto(
            batch=new_batch,
            non_tensor_batch=non_tensor_batch,
            remote_batch=remote_batch,
            meta_info=merged_meta,
        )

    def reorder(self, indices):
        """
        Note that this operation is in-place
        """
        # Ensure that indices is at least a 1-D tensor.
        indices = indices.view(-1) if indices.dim() == 0 else indices
        indices_np = indices.detach().numpy()
        self._batch = self._batch[indices] if self._batch is not None else None
        self._non_tensor_batch = {key: val[indices_np] for key, val in self._non_tensor_batch.items()}
        self._remote_batch = self._remote_batch.select_idxs(indices) if self._remote_batch else None

    def group_by(self, keys: Union[List[str], str]) -> Dict[str, "DataProto"]:
        """
        Group the data by specified keys. Supports grouping by both tensor and non-tensor fields.

        Args:
            keys: Field names to group by. Can be either in batch (tensors) or non_tensor_batch

        Returns:
            Dictionary mapping group keys to DataProto instances containing matching data

        Example:
            Given data with field "category" having values ["A", "B", "A"],
            returns {"A": DataProto(A_data), "B": DataProto(B_data)}
        """
        keys = self.validate_input(keys)
        assert len(keys) > 0, "Must provide at least one grouping key"

        remote_keys = [key for key in keys if self._remote_batch is not None and key in self._remote_batch]
        if remote_keys and not self._remote_batch.cached(remote_keys):
            self._remote_batch.materialize(remote_keys)
            logger.warning(f"RemoteBatch implicit materialize key {remote_keys} for group by")

        # Collect grouping values across data types
        group_key_values = []
        for idx in range(len(self)):
            key_values = []
            for key in keys:
                # Check tensor data first
                if self._batch is not None and key in self._batch.keys():
                    key_values.append(str(self._batch[key][idx].numpy()))
                elif key in self._non_tensor_batch:
                    key_values.append(str(self._non_tensor_batch[key][idx]))
                elif self._remote_batch is not None and key in self._remote_batch:
                    key_values.append(str(self._remote_batch[key][idx]))
                else:
                    raise KeyError(f"Grouping key '{key}' not found in tensor or non-tensor data")

            # Create composite key for multi-field grouping
            group_key = "|".join(key_values) if len(key_values) > 1 else key_values[0]
            group_key_values.append(group_key)

        # Create index groups
        groups = defaultdict(list)
        for idx, group_key in enumerate(group_key_values):
            groups[group_key].append(idx)

        # Create grouped DataProtos
        grouped_data = {}
        for group_key, indices in groups.items():
            grouped_data[group_key] = self.select_idxs(indices)

        return grouped_data

    def repeat(self, repeat_times=2, interleave=True):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self._batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self._batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self._batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self._batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self._non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        repeated_remote_batch = self._remote_batch.repeat(repeat_times, interleave) if self._remote_batch else None

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            remote_batch=repeated_remote_batch,
            meta_info=self.meta_info,
        )

    @staticmethod
    def materialize_concat(
            data_refs: Union[List[ray.ObjectRef], ray.ObjectRef, List["ObjectRefWrap"]],
            *,
            global_keys: Optional[Set[str]] = None,
    ) -> "DataProto":
        """
        Fetch a collection of DataProto objects from Ray ObjectRef(s) and concatenate
        them into a single DataProto instance.

        Parameters
        ----------
        data_refs : Union[List[ray.ObjectRef], ray.ObjectRef, List[ObjectRefWrap]]
            Ray object references (or ObjectRefWrap) pointing to DataProto objects.
        global_keys : Optional[Set[str]], optional
            Keys in ``meta_info`` that should be aggregated across all ranks when
            concatenating.  If None, only rank-0 values are kept for all keys.

        Returns
        -------
        DataProto
            The concatenated DataProto instance.
        """
        # Normalize input to List[<reference>]
        if isinstance(data_refs, DataProto):
            data_refs = [data_refs]

        timeout = None
        if "roll_RPC_TIMEOUT" in os.environ:
            timeout = int(os.environ["roll_RPC_TIMEOUT"])

        # Fetch objects from Ray
        if isinstance(data_refs[0], ObjectRefWrap):
            data_refs: List[ObjectRefWrap]
            obj_refs = [ref.obj_ref for ref in data_refs]
            fetched = ray.get(obj_refs, timeout=timeout)
            data = [fetched[i] for i, ref in enumerate(data_refs) if ref.collected]
        else:
            data: List["DataProto"] = ray.get(data_refs, timeout=timeout)

        # Concatenate and apply global aggregation rules
        return DataProto.concat(data, global_keys=global_keys)

    @classmethod
    def to_remote(cls, data: "DataProto", partition = "train_eval", *, ref_data = None) -> "DataProto":
        with Timer(name="RemoteBatch to_remote", logger=None) as timer:
            batch_size = len(data)

            if ref_data is not None:
                if len(ref_data) != batch_size:
                    logger.warning(f"RemoteBatch to_remote ref_data batch size {len(ref_data)} does not match data batch size {batch_size}, {data=}")
                    return data
                if ref_data._remote_batch is None or ref_data._remote_batch.row_ids() is None:
                    logger.warning(f"RemoteBatch to_remote ref_data has no row ids")
                    return data
                row_ids = ref_data._remote_batch.row_ids()
                if len(row_ids) != batch_size:
                    logger.warning(f"RemoteBatch to_remote ref_data row ids batch size {len(row_ids)} does not match data batch size {batch_size}")
                    return data
                if data._remote_batch is not None and data._remote_batch.row_ids() != row_ids:
                    logger.warning(f"RemoteBatch to_remote ref_data and data have different row ids")
                    return data
                logger.info(f"RemoteBatch to_remote add to current rows, {partition=} {len(row_ids)=}")
            elif data._remote_batch is not None and data._remote_batch.row_ids() is not None:
                row_ids = data._remote_batch.row_ids()
            else:
                row_ids = [str(uuid.uuid4()) for _ in range(batch_size)]
                logger.info(f"RemoteBatch to_remote add {len(row_ids)} new rows, row ids {partition=} row_ids={row_ids[:10]}...")
            assert len(row_ids) == batch_size

            # Partition is used to group data of a step.
            # Some backend may support drop partition or delete keys using regex.
            assert isinstance(partition, str)
            partition = data._remote_batch.partition if data._remote_batch is not None else partition

            data_dict: dict = data._batch.to_dict() if data._batch is not None else {}
            data_dict.update(data._non_tensor_batch)

            if not data_dict:
                return data

            remote_batch = transfer_backend.put(partition, row_ids, data_dict, batch_size)
            if remote_batch is None: # transfer backend is not available
                assert data._remote_batch is None
                return data

            if data._remote_batch is not None:
                remote_batch = remote_batch.union(data._remote_batch)

        logger.info(f"RemoteBatch to_remote finished in {timer.last}s")

        return DataProto(
            batch=None,
            non_tensor_batch={},
            remote_batch=remote_batch,
            meta_info=data.meta_info,
        )

    def prefetch(self, keys: list[str] | str | None = None):
        if self._remote_batch is None:
            return
        if keys is not None:
            keys = [keys] if isinstance(keys, str) else keys
        if keys is None or not self._remote_batch.cached(keys):
            with Timer(name="RemoteBatch prefetch", logger=None) as timer:
                self._remote_batch.materialize(keys)
            logger.info(f"RemoteBatch prefetch finished in {timer.last}s")

    @classmethod
    def drop(cls, data: "DataProto"):
        if data._remote_batch is None:
            return
        with Timer(name="RemoteBatch drop", logger=None) as timer:
            partition = data._remote_batch.partition
            logger.info(f"RemoteBatch drop {partition=} row_ids={data._remote_batch.row_ids()}")
            data._remote_batch.drop()
        logger.info(f"RemoteBatch drop {partition=} finished in {timer.last}s")


class ObjectRefWrap:
    def __init__(self, obj_ref: ray.ObjectRef, collected=False):
        self.obj_ref = obj_ref
        self.collected = collected
