import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.utils import LinkedList
from codetiming import Timer

from roll.distributed.scheduler import transfer_backend
from roll.utils.logging import get_logger

logger = get_logger()


class RemoteBatch:
    def __init__(self, key_type: str, partition: str, device):
        self.key_type = key_type
        self.partition = partition
        self.device = None

    def __reduce__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)
        elif isinstance(item, str):
            td = self.materialize([item])
            assert isinstance(td, TensorDict), f"Expected TensorDict, got {type(td)}"
            value = td[item]
            assert isinstance(value, (torch.Tensor, LinkedList))
            if isinstance(value, LinkedList):
                items = list(value)
                return np.array(items, dtype=object)
            else:
                return value
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __contains__(self, key: str) -> bool:
        raise NotImplementedError

    def clone(self, recurse: bool = True):
        raise NotImplementedError

    def keys(self):
        """
        If not specified keys and fields are the same.
        (keys only reference to key to kv storage in materialize now)
        """
        raise NotImplementedError

    def row_ids(self):
        return None

    def to(self, device) -> "RemoteBatch":
        self.device = device
        return self

    def materialize(self, fields: list[str] = None) -> TensorDict:
        raise NotImplementedError

    def cached(self, fields: list[str]) -> bool:
        if self.cache is None:
            return False
        else:
            return all(field in self.cache for field in fields)

    def drop(self):
        raise NotImplementedError

    def select(self, fileds: list[str]) -> "RemoteBatch":
        raise NotImplementedError

    def select_idxs(self, index: torch.Tensor | np.ndarray | list) -> "RemoteBatch":
        raise NotImplementedError

    def slice(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ) -> "RemoteBatch":
        raise NotImplementedError

    def pop(self, fileds) -> "RemoteBatch":
        raise NotImplementedError

    def chunk(self, chunk_sizes: list[int]) -> list["RemoteBatch"]:
        raise NotImplementedError

    def repeat(self, repeat_times: int, interleave: bool) -> "RemoteBatch":
        raise NotImplementedError

    def union(self, rhs: "RemoteBatch") -> "RemoteBatch":
        """
        RemoteBatch.union will not check the following preconditions:
            - there are conflict keys in batch and they are not equal
            - the batch size of two data batch is not the same
        """
        raise NotImplementedError

    @classmethod
    def cat(cls, data: list["RemoteBatch"]) -> "RemoteBatch":
        assert data
        target_cls = type(data[0])
        assert all(
            type(d) is target_cls for d in data
        ), f"All batches must be of the same type, got {[type(d).__name__ for d in data]}"
        return target_cls._cat(data)

    @classmethod
    def _cat(cls, data: list["RemoteBatch"]) -> "RemoteBatch":
        raise NotImplementedError


class BatchProxy:
    """
    Proxy for batch that supports fallback lookup to remote_batch.

    Only support a minimal set of special methods and normal methods that works identically on
    both TensorDict and dict[np.ndarray]. Raises on other special methods (__len__, __iter__, ...).

    Only support properties of TensorDict currently used in codebase for backward compatibility.
    Use of properties of dict is not supported.
    """

    def __init__(self, batch: TensorDict | dict[np.ndarray] | None, remote_batch: RemoteBatch | None, batch_size: int):
        assert batch is None or isinstance(batch, (TensorDict, dict))
        self._batch = batch
        self._remote_batch = remote_batch
        self._batch_size = batch_size

    def __getitem__(self, key: str):
        if self._batch is not None and key in self._batch:
            return self._batch[key]
        elif self._remote_batch is not None and key in self._remote_batch:
            return self._remote_batch[key]
        else:
            raise KeyError(f"Key '{key}' not found in batch or remote_batch")

    def __setitem__(self, key: str, value):
        assert isinstance(value, (torch.Tensor, np.ndarray))
        if self._remote_batch is not None and key in self._remote_batch:
            # Just delete from local, does not delete from remote server.
            del self._remote_batch[key]
        if self._batch is not None:
            assert (
                len(value) == self._batch_size
            ), f"Value length {len(value)} does not match batch length {self._batch_size}"
            self._batch[key] = value
        else:
            raise RuntimeError("Cannot set item when batch is None")

    def __delitem__(self, key: str):
        if self._batch is not None and key in self._batch:
            del self._batch[key]
        elif self._remote_batch is not None and key in self._remote_batch:
            # Just delete from local, does not delete from remote server.
            del self._remote_batch[key]
        else:
            raise KeyError(f"Key '{key}' not found in batch or remote_batch")

    def __contains__(self, key: str) -> bool:
        in_batch = self._batch is not None and key in self._batch
        in_remote = self._remote_batch is not None and key in self._remote_batch
        return in_batch or in_remote

    def copy(self) -> "BatchProxy":
        """Shallow copy of the BatchProxy."""
        batch_copy = self._batch.copy() if self._batch is not None else None
        remote_copy = self._remote_batch.clone(recurse=False) if self._remote_batch is not None else None
        return BatchProxy(batch_copy, remote_copy, self._batch_size)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        result = set()
        if self._batch is not None:
            result |= set(self._batch.keys())
        if self._remote_batch is not None:
            result |= set(self._remote_batch.keys())
        return result

    def items(self):
        """
        WARNING: this function will materializes remote_batch if exists and does not guarantee the order of items.
        """
        # Yield from _batch first
        if self._batch is not None:
            if isinstance(self._batch, TensorDict):
                for key in self._batch.keys():
                    yield (key, self._batch[key])
            else:
                # dict
                for key, val in self._batch.items():
                    yield (key, val)
        # Yield from _remote_batch for keys not in _batch
        if self._remote_batch is not None:
            logger.warning("RemoteBatch materializing remote batch for items()")
            self._remote_batch.materialize()
            for key in self._remote_batch.keys():
                yield (key, self._remote_batch[key])

    _POP_SENTINEL = object()

    def pop(self, key: str, default=_POP_SENTINEL):
        if key not in self:
            if default is BatchProxy._POP_SENTINEL:
                raise KeyError(f"Key '{key}' not found in batch or remote_batch")
            return default
        res = self[key]
        if self._remote_batch is not None and key in self._remote_batch:
            del self._remote_batch[key]
        if self._batch is not None and key in self._batch:
            del self._batch[key]
        return res

    def update(self, other: dict):
        assert isinstance(other, dict)
        assert self._batch is not None, "Update with batch is None is not supported, use DataProto.update insted."
        for key in other.keys():
            if self._remote_batch is not None and key in self._remote_batch:
                del self._remote_batch[key]
        self._batch.update(other)

    def __getattr__(self, name: str):
        """
        Raise AttributeError for all other attributes.
        Because the semantics of returning getattr(self._batch, name) is undefined.
        """
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ============================================================================
    # Below properties are for backward compatibility only.
    # Only supported when:
    #   - self._batch is not None and isinstance(self._batch, TensorDict), OR
    #   - self._batch is None and self._remote_batch is not None
    # ============================================================================

    @property
    def batch_size(self) -> torch.Size:
        """Return batch size. Only supported when _batch is TensorDict or _remote_batch exists."""
        if isinstance(self._batch, TensorDict):
            return self._batch.batch_size
        if self._batch is None and self._remote_batch is not None:
            return torch.Size([len(self._remote_batch)])
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute 'batch_size' "
            f"(batch is {type(self._batch).__name__ if self._batch else 'None'})"
        )

    @property
    def shape(self) -> torch.Size:
        """Return batch shape. Only supported when _batch is TensorDict or _remote_batch exists."""
        if isinstance(self._batch, TensorDict):
            return self._batch.shape
        if self._batch is None and self._remote_batch is not None:
            return torch.Size([len(self._remote_batch)])
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute 'shape' "
            f"(batch is {type(self._batch).__name__ if self._batch else 'None'})"
        )

    @property
    def device(self):
        """Return device. Only supported when _batch is TensorDict or _remote_batch exists."""
        if isinstance(self._batch, TensorDict):
            return self._batch.device
        if self._batch is None and self._remote_batch is not None:
            return self._remote_batch.device
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute 'device' "
            f"(batch is {type(self._batch).__name__ if self._batch else 'None'})"
        )


class RowRemoteBatch(RemoteBatch):
    """
    A remote batch stored in a key-value store with row id as keys.
    """

    def __init__(self, partition: str, device, fields, row_ids: list[str], cache: TensorDict):
        super().__init__("row", partition, device)
        self.fields = set(fields)  # str, stores column names
        self._row_ids = row_ids.copy()
        self.cache = cache.clone() if cache is not None else None

    def __reduce__(self):
        return (
            RowRemoteBatch,
            (self.partition, self.device, self.fields, self._row_ids, None),
        )

    def __repr__(self):
        return f"RowRemoteBatch(partition={self.partition}, device={self.device}, fields={self.fields}, row_ids={self._row_ids}, cache={self.cache})"

    def __len__(self):
        return len(self._row_ids)

    def __delitem__(self, key: str):
        self.fields.remove(key)
        if self.cache is not None and key in self.cache:
            del self.cache[key]

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def clone(self, recurse: bool = True):
        return RowRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=self.fields.copy(),
            row_ids=self._row_ids.copy(),
            cache=self.cache.clone(recurse=recurse) if self.cache is not None else None,
        )

    def keys(self):
        return self.fields

    def row_ids(self):
        return self._row_ids

    def to(self, device) -> "RowRemoteBatch":
        super().to(device)
        if self.cache is not None:
            self.cache = self.cache.to(device)
        return self

    def materialize(self, fields: list[str] = None) -> TensorDict:
        if fields is None:
            fields = self.fields
        else:
            assert set(fields) <= self.fields, f"Fields {set(fields)} is not subset of {self.fields}"
        existing_fields = set(self.cache.keys()) if self.cache is not None else set()
        fetch_fields = [field for field in fields if field not in existing_fields]
        if len(fetch_fields) > 0:
            with Timer(name="remote_batch_materialize", logger=None) as timer:
                data: TensorDict = transfer_backend.get(partition=self.partition, keys=self._row_ids, fields=fetch_fields)
                assert set(data.keys()) == set(fetch_fields)

                if self.cache is None:
                    self.cache = data
                else:
                    from roll.distributed.scheduler.protocol import union_tensor_dict

                    self.cache = union_tensor_dict(self.cache, data)
                if self.device is not None:
                    self.cache.to(self.device)
            logger.info(f"RemoteBatch materialize cost {timer.last}s, partition={self.partition}, new materialized {sorted(fetch_fields)}, cached fields {sorted(list(existing_fields))}")

        return self.cache.select(*fields)

    def drop(self):
        transfer_backend.delete(partition=self.partition, keys=self._row_ids, fields=list(self.fields))

    def select(self, fileds: list[str]) -> "RowRemoteBatch":
        assert all(key in self.fields for key in fileds), f"Keys {fileds} not in {self.fields}"
        cache = self.cache
        if cache is not None:
            keys_in_cache = [k for k in fileds if k in cache.keys()]
            if keys_in_cache:
                cache = cache.select(*keys_in_cache)
            else:
                cache = None
        return RowRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=fileds,
            row_ids=self._row_ids,
            cache=cache,
        )

    def select_idxs(self, index: torch.Tensor | np.ndarray | list) -> "RowRemoteBatch":
        assert isinstance(index, (torch.Tensor, np.ndarray, list))
        if isinstance(index, np.ndarray):
            index_list = index.tolist()
            index = torch.from_numpy(index)
        elif isinstance(index, list):
            index_list = index
            index = torch.tensor(index)
        else:
            index_list = index.tolist()

        if index.dtype == torch.bool:
            selected_row_ids = [self._row_ids[i] for i, mask in enumerate(index_list) if mask]
        else:
            selected_row_ids = [self._row_ids[i] for i in index_list]

        cache = self.cache
        if cache is not None:
            cache = cache[index]
            assert isinstance(cache, TensorDict)

        return RowRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=self.fields,
            row_ids=selected_row_ids,
            cache=cache,
        )

    def slice(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ) -> "RowRemoteBatch":
        sliced_row_ids = self._row_ids[start:end:step]

        cache = self.cache
        if cache is not None:
            cache = cache[slice(start, end, step)]

        return RowRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=self.fields,
            row_ids=sliced_row_ids,
            cache=cache,
        )

    def pop(self, filed) -> "RowRemoteBatch":
        assert len(filed) == len(set(filed)), "Fields must be unique"
        assert set(filed) <= self.fields, f"Fields {set(filed) - self.fields} not in batch"
        ret = self.select(filed)

        self.fields -= set(filed)
        if self.cache is not None:
            remaining_keys = [k for k in self.fields if k in self.cache.keys()]
            if remaining_keys:
                self.cache = self.cache.select(*remaining_keys)
            else:
                self.cache = None

        return ret

    def chunk(self, chunk_sizes: list[int]) -> list["RowRemoteBatch"]:
        assert sum(chunk_sizes) == len(
            self
        ), f"Sum of chunk_sizes {sum(chunk_sizes)} does not match batch size {len(self)}"
        chunks = []
        offset = 0
        for size in chunk_sizes:
            chunks.append(self.slice(offset, offset + size))
            offset += size
        return chunks

    def repeat(self, repeat_times: int, interleave: bool) -> "RowRemoteBatch":
        if interleave:
            repeated_row_ids = [row_id for row_id in self._row_ids for _ in range(repeat_times)]
        else:
            repeated_row_ids = self._row_ids * repeat_times

        cache = self.cache
        if cache is not None:
            if interleave:
                cache = cache.repeat_interleave(repeat_times)
            else:
                cache = cache.repeat(repeat_times)

        return RowRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=self.fields,
            row_ids=repeated_row_ids,
            cache=cache,
        )

    def union(self, rhs: "RowRemoteBatch") -> "RowRemoteBatch":
        assert isinstance(rhs, RemoteBatch)
        assert len(self) == len(rhs), f"Two tensor dict must have identical batch size. Got {len(self)} and {len(rhs)}"
        if self.cache is not None and rhs.cache is not None:
            from roll.distributed.scheduler.protocol import union_tensor_dict

            union_tensor_dict(self.cache, rhs.cache)
        elif self.cache is None:
            self.cache = rhs.cache.clone() if rhs.cache is not None else None

        for field in rhs.fields:
            if field in self.fields:
                assert set(self._row_ids) == set(rhs._row_ids), f"Row ids must be the same. Got {self._row_ids} and {rhs._row_ids}"
                continue
            self.fields.add(field)

        return self

    @classmethod
    def _cat(cls, data: list["RowRemoteBatch"]) -> "RowRemoteBatch":
        assert data
        if len(data) == 1:
            return data[0]

        fields = data[0].fields
        assert all(d.fields == fields for d in data), "All batches must have the same fields"
        partition = data[0].partition
        assert all(d.partition == partition for d in data), "All batches must have the same partition"

        row_ids = [row_id for d in data for row_id in d._row_ids]

        caches = [d.cache for d in data]
        if all(c is not None for c in caches):
            first_keys = set(caches[0].keys())
            if all(set(c.keys()) == first_keys for c in caches[1:]):
                cache = TensorDict.cat(caches, dim=0)
            else:
                cache = None
        else:
            cache = None

        return RowRemoteBatch(
            partition=partition,
            device=data[0].device,
            fields=fields,
            row_ids=row_ids,
            cache=cache,
        )


class PlanNode(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @abstractmethod
    def execute(self, data):
        pass


class SelectPlan(PlanNode):
    def __init__(self, index: torch.Tensor):
        super().__init__()
        assert isinstance(index, torch.Tensor) and index.dim() == 1
        self.index = index

    @property
    def batch_size(self):
        return int(self.index.sum().item()) if self.index.dtype == torch.bool else self.index.shape[0]

    def execute(self, data):
        return data[self.index.to(data.device)]


class SlicePlan(PlanNode):
    def __init__(self, start: int, end: int, step: int, batch_size: int):
        super().__init__()
        self.slice_obj = slice(start, end, step)
        self.source_batch_size = batch_size

    @property
    def batch_size(self):
        return len(range(*self.slice_obj.indices(self.source_batch_size)))

    def to_select(self):
        start, stop, step = self.slice_obj.indices(self.source_batch_size)
        return SelectPlan(np.arange(start, stop, step))

    def execute(self, data):
        return data[self.slice_obj]


class RepeatPlan(PlanNode):
    def __init__(self, repeat_times: int, interleave: bool, source_batch_size: int):
        super().__init__()
        self.repeat_times = repeat_times
        self.interleave = interleave
        self.source_batch_size = source_batch_size

    @property
    def batch_size(self):
        return self.source_batch_size * self.repeat_times

    def execute(self, data):
        assert isinstance(data, TensorDict)
        if self.interleave:
            return data.repeat_interleave(self.repeat_times)
        else:
            return data.repeat(self.repeat_times)


class CatPlan(PlanNode):
    def __init__(self, batch_size: int):
        super().__init__()
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def execute(self, data):
        assert isinstance(data, list) and all(isinstance(d, TensorDict) for d in data)
        return TensorDict.cat(data, dim=0)


# TODO shigao: use Box to share materialized remote object
class Box:
    """
    Can not used in asyncio context, threading.Lock will block event loop.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.value = None

    def get(self):
        with self.lock:
            return self.value

    def set(self, value):
        with self.lock:
            if self.value is None:
                self.value = value


class ColumnRemoteBatch(RemoteBatch):
    """
    A remote batch stored in a key-value store with column id as keys.
    """

    def __init__(
        self,
        partition: str,
        device,
        fields: dict[str, Any | list["ColumnRemoteBatch"]],
        is_nested: bool,
        cache: TensorDict,
        batch_size: int,
        pipeline: tuple[PlanNode] = tuple(),
    ):
        """
        fields contains any meta need to be hold and pass to transfer backend during get
        or a list of ColumnRemoteBatch if is nested.
        """
        super().__init__("column", partition, device)
        self.fields = fields
        self.is_nested = is_nested
        self.cache = cache
        self.batch_size = batch_size
        self.pipeline = pipeline

    def __reduce__(self):
        return (
            ColumnRemoteBatch,
            (self.partition, self.device, self.fields, self.is_nested, None, self.batch_size, self.pipeline),
        )

    def __len__(self) -> int:
        return self.batch_size

    def __delitem__(self, key: str):
        self.fields.pop(key)
        if self.cache is not None and key in self.cache:
            del self.cache[key]

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def clone(self, recurse: bool = True):
        if self.is_nested:
            cloned_fields = {k: [batch.clone() for batch in v] for k, v in self.fields.items()}
        else:
            cloned_fields = self.fields.copy()
        return ColumnRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=cloned_fields,
            is_nested=self.is_nested,
            cache=self.cache.clone(recurse=recurse) if self.cache is not None else None,
            batch_size=self.batch_size,
            pipeline=self.pipeline,
        )

    def keys(self):
        return self.fields.keys()

    def to(self, device) -> "ColumnRemoteBatch":
        super().to(device)
        if self.cache is not None:
            self.cache = self.cache.to(device)
        return self

    def materialize(self, fields: list[str] = None) -> TensorDict:
        if fields is None:
            fields = self.fields.keys()
        else:
            assert set(fields) <= set(self.fields.keys())
        existing_fields = set(self.cache.keys()) if self.cache is not None else set()

        data = None
        if self.is_nested:
            assert len(self.pipeline) == 1 and isinstance(self.pipeline[0], CatPlan)
            chunks: list["ColumnRemoteBatch"] = next(iter(self.fields.values()))
            # TODO shigao: use batch get
            # TODO: parallel gather
            data: list[TensorDict] = [chunk.materialize(fields) for chunk in chunks]
        else:
            # pass column(key) meta back to transfer backend
            fetch_fields = {field: self.fields[field] for field in fields if field not in existing_fields}
            if len(fetch_fields) > 0:
                data: TensorDict = transfer_backend.get(
                    partition=self.partition, keys=list(fetch_fields.keys()), fields=list(fetch_fields.values())
                )

        if data is not None:
            for operator in self.pipeline:
                data = operator.execute(data)
            assert len(data) == self.batch_size

            if self.device is not None:
                data.to(self.device)

            if self.cache is None:
                self.cache = data
            else:
                from roll.distributed.scheduler.protocol import union_tensor_dict

                self.cache = union_tensor_dict(self.cache, data)

        return self.cache.select(*fields)

    def drop(self):
        transfer_backend.delete(
            partition=self.partition, keys=list(self.fields.keys()), fields=list(self.fields.values())
        )

    def select(self, fileds: list[str]) -> "ColumnRemoteBatch":
        assert all(key in self.fields for key in fileds), f"Keys {fileds} not in {self.fields.keys()}"
        fields = {key: self.fields[key] for key in fileds}
        cache = self.cache
        if cache is not None:
            keys_in_cache = [k for k in fileds if k in cache.keys()]
            if keys_in_cache:
                cache = cache.select(*keys_in_cache)
            else:
                cache = None
        return ColumnRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=fields,
            cache=cache,
            batch_size=self.batch_size,
            is_nested=self.is_nested,
            pipeline=self.pipeline,
        )

    def _selection(self, plan: PlanNode) -> "ColumnRemoteBatch":
        batch_size = plan.batch_size

        cache = self.cache
        if cache is not None:
            cache = plan.execute(cache)
            assert isinstance(cache, TensorDict)

        if not self.pipeline:
            pipeline = self.pipeline + (plan,)
        else:
            # TODO: heuristic optimization
            # TODO: predicate pushdown
            if isinstance(plan, SelectPlan) and isinstance(self.pipeline[-1], SelectPlan):
                pipeline = self.pipeline + (plan,)
            else:
                pipeline = self.pipeline + (plan,)

        return ColumnRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=self.fields,
            cache=cache,
            batch_size=batch_size,
            is_nested=self.is_nested,
            pipeline=pipeline,
        )

    def select_idxs(self, index: torch.Tensor | np.ndarray | list) -> "ColumnRemoteBatch":
        assert isinstance(index, (torch.Tensor, np.ndarray, list))
        if isinstance(index, np.ndarray):
            index = torch.from_numpy(index)
        elif isinstance(index, list):
            index = torch.tensor(index)
            if index.dtype != torch.bool:
                index = index.type(torch.int32)

        plan = SelectPlan(index)
        return self._selection(plan)

    def slice(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ) -> "ColumnRemoteBatch":
        plan = SlicePlan(start, end, step, self.batch_size)
        return self._selection(plan)

    def pop(self, fileds) -> "ColumnRemoteBatch":
        assert len(fileds) == len(set(fileds)), "Fields must be unique"
        assert set(fileds) <= self.fields.keys(), f"Fields {set(fileds) - self.fields.keys()} not in batch"
        ret = self.select(fileds)

        for key in fileds:
            del self.fields[key]
        if self.cache is not None:
            remaining_keys = [k for k in self.fields.keys() if k in self.cache.keys()]
            if remaining_keys:
                self.cache = self.cache.select(*remaining_keys)
            else:
                self.cache = None

        return ret

    def chunk(self, chunk_sizes: list[int]) -> list["ColumnRemoteBatch"]:
        assert sum(chunk_sizes) == len(
            self
        ), f"Sum of chunk_sizes {sum(chunk_sizes)} does not match batch size {len(self)}"
        chunks = []
        offset = 0
        for size in chunk_sizes:
            chunks.append(self.slice(offset, offset + size))
            offset += size
        return chunks

    def repeat(self, repeat_times: int, interleave: bool) -> "ColumnRemoteBatch":
        plan = RepeatPlan(repeat_times, interleave, self.batch_size)
        return self._selection(plan)

    def union(self, rhs: "ColumnRemoteBatch") -> "ColumnRemoteBatch":
        assert isinstance(rhs, RemoteBatch)
        assert len(self) == len(rhs), f"Two tensor dict must have identical batch size. Got {len(self)} and {len(rhs)}"
        if self.cache is not None and rhs.cache is not None:
            from roll.distributed.scheduler.protocol import union_tensor_dict

            union_tensor_dict(self.cache, rhs.cache)
        elif self.cache is None:
            self.cache = rhs.cache.clone() if rhs.cache is not None else None

        for field, value in rhs.fields.items():
            if field in self.fields:
                # assert self.cache[field].equal(rhs.cache[field]), f"{field=}"
                continue
            self.fields[field] = value

        return self

    @classmethod
    def _cat(cls, data: list["ColumnRemoteBatch"]) -> "ColumnRemoteBatch":
        """
        ColumnRemoteBatch._cat will not check type and shape[1:] of fields.
        """
        assert data
        if len(data) == 1:
            return data[0]

        keys = set(data[0].fields.keys())
        assert all(set(d.fields.keys()) == keys for d in data), "All batches must have the same fields"
        partition = data[0].partition
        assert all(d.partition == partition for d in data), "All batches must have the same partition"
        device = data[0].device

        batch_size = sum(d.batch_size for d in data)
        plan = CatPlan(batch_size)

        caches = [d.cache for d in data]
        if all(c is not None for c in caches):
            first_keys = set(caches[0].keys())
            if all(set(c.keys()) == first_keys for c in caches[1:]):
                cache = plan.execute(caches)
            else:
                cache = None
        else:
            cache = None

        return ColumnRemoteBatch(
            partition=partition,
            device=device,
            fields={field: data for field in keys},
            is_nested=True,
            cache=cache,
            batch_size=batch_size,
            pipeline=(plan,),
        )
