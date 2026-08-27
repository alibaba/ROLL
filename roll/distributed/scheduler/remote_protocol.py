import threading
import uuid
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch
from codetiming import Timer
from tensordict import TensorDict
from tensordict.utils import LinkedList

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

    def _ensure_active(self) -> None:
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
                data: TensorDict = transfer_backend.get(
                    partition=self.partition, keys=self._row_ids, fields=fetch_fields
                )
                assert set(data.keys()) == set(fetch_fields)

                if self.cache is None:
                    self.cache = data
                else:
                    from roll.distributed.scheduler.protocol import union_tensor_dict

                    self.cache = union_tensor_dict(self.cache, data)
                if self.device is not None:
                    self.cache.to(self.device)
            logger.info(
                f"RemoteBatch materialize cost {timer.last}s, partition={self.partition}, new materialized {sorted(fetch_fields)}, cached fields {sorted(list(existing_fields))}"
            )

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
                assert set(self._row_ids) == set(
                    rhs._row_ids
                ), f"Row ids must be the same. Got {self._row_ids} and {rhs._row_ids}"
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


class _ColumnRemoteState:
    """Shared lifetime for one physical column bundle."""

    def __init__(self, partition: str, keys: list[str], fields: list[Any], state_id: str | None = None):
        self.state_id = state_id or uuid.uuid4().hex
        self.partition = partition
        self.keys = tuple(keys)
        self.fields = tuple(fields)
        self.active = True
        self.delete_pending = True
        self._lock = threading.Lock()
        self._pickle_snapshot = threading.local()

    def _snapshot(self):
        return {
            "state_id": self.state_id,
            "partition": self.partition,
            "keys": self.keys,
            "fields": self.fields,
            "active": self.active,
            "delete_pending": self.delete_pending,
        }

    def __getstate__(self):
        with self._lock:
            snapshot = self._snapshot()
            active, delete_pending = getattr(
                self._pickle_snapshot,
                "value",
                (self.active, self.delete_pending),
            )
            if hasattr(self._pickle_snapshot, "value"):
                del self._pickle_snapshot.value
            snapshot["active"] = active
            snapshot["delete_pending"] = delete_pending
            return snapshot

    def __setstate__(self, snapshot):
        restored = self._from_snapshot(snapshot)
        self.__dict__.update(restored.__dict__)

    @classmethod
    def _from_snapshot(cls, snapshot):
        state = cls(
            partition=snapshot["partition"],
            keys=snapshot["keys"],
            fields=snapshot["fields"],
            state_id=snapshot["state_id"],
        )
        state.active = snapshot["active"]
        state.delete_pending = snapshot["delete_pending"]
        return state


class _ColumnRemoteLifetimeDomain:
    """Connected cleanup states serialized as one atomic snapshot."""

    _merge_lock = threading.Lock()

    def __init__(self, states: list[_ColumnRemoteState]):
        self._parent = None
        self._states = {state.state_id: state for state in states}
        for state in self._states.values():
            state._domain_ref = weakref.ref(self)

    @classmethod
    def merge(cls, left, right):
        return cls.merge_many((left, right))

    @classmethod
    def merge_many(cls, domains):
        with cls._merge_lock:
            return cls._merge_many_unlocked(domains)

    @classmethod
    def _merge_many_unlocked(cls, domains):
        roots = []
        seen = set()
        for domain in domains:
            root = domain._root_unlocked()
            if id(root) in seen:
                continue
            seen.add(id(root))
            roots.append(root)
        if not roots:
            raise ValueError("At least one lifetime domain is required")
        if len(roots) == 1:
            return roots[0]
        states = {}
        for root in roots:
            for state_id, state in root._states.items():
                states.setdefault(state_id, state)
        merged = cls(list(states.values()))
        for root in roots:
            root._parent = merged
        return merged

    def root(self):
        with self._merge_lock:
            return self._root_unlocked()

    def states(self, state_ids) -> list[_ColumnRemoteState]:
        with self._merge_lock:
            root = self._root_unlocked()
            missing = [state_id for state_id in state_ids if state_id not in root._states]
            if missing:
                raise RuntimeError("ColumnRemoteBatch was structurally modified during serialization")
            return [root._states[state_id] for state_id in state_ids]

    def __getstate__(self):
        with self._locked_states(freeze_domain=True) as states:
            for state in states:
                snapshot = state._snapshot()
                state._pickle_snapshot.value = (
                    snapshot["active"],
                    snapshot["delete_pending"],
                )
            return {"states": states}

    def __setstate__(self, snapshot):
        states = snapshot["states"]
        connected_domains = []
        for state in states:
            domain_ref = getattr(state, "_domain_ref", None)
            domain = domain_ref() if domain_ref is not None else None
            if domain is not None:
                connected_domains.append(domain)
        self._parent = None
        self._states = {state.state_id: state for state in states}
        with self._merge_lock:
            root = self._merge_many_unlocked((self, *connected_domains))
            for state in root._states.values():
                state._domain_ref = weakref.ref(root)

    def _root_unlocked(self):
        root = self
        while root._parent is not None:
            root = root._parent
        current = self
        while current._parent is not None and current._parent is not root:
            parent = current._parent
            current._parent = root
            current = parent
        return root

    @contextmanager
    def _locked_states(self, state_ids=None, freeze_domain=False):
        while True:
            with self._merge_lock:
                root = self._root_unlocked()
                selected_ids = tuple(root._states) if state_ids is None else tuple(state_ids)
                states = [root._states[state_id] for state_id in selected_ids]

            acquired = []
            try:
                for state in sorted(states, key=lambda state: state.state_id):
                    state._lock.acquire()
                    acquired.append(state)
            except BaseException:
                for state in reversed(acquired):
                    state._lock.release()
                raise

            try:
                self._merge_lock.acquire()
            except BaseException:
                for state in reversed(acquired):
                    state._lock.release()
                raise
            current_root = self._root_unlocked()
            valid = current_root is root
            if valid:
                if not freeze_domain:
                    self._merge_lock.release()
                break
            self._merge_lock.release()
            for state in reversed(acquired):
                state._lock.release()
        try:
            yield states
        finally:
            if freeze_domain:
                self._merge_lock.release()
            for state in reversed(acquired):
                state._lock.release()


class _ColumnRemoteLifetime:
    """Cleanup state shared by local aliases."""

    def __init__(self, states: list[_ColumnRemoteState], domain=None, state_ids=None):
        self._domain = domain or _ColumnRemoteLifetimeDomain(states)
        self._state_ids = tuple(state_ids or (state.state_id for state in states))

    def __reduce__(self):
        return (_restore_column_remote_lifetime, (self._domain.root(), self._state_ids))

    @property
    def states(self):
        return self._domain.states(self._state_ids)

    def connect_many(self, others):
        self._domain = _ColumnRemoteLifetimeDomain.merge_many((self._domain, *(other._domain for other in others)))

    def extend(self, other, states):
        if not states:
            return self
        domain = _ColumnRemoteLifetimeDomain.merge(self._domain, other._domain)
        state_ids = dict.fromkeys((*self._state_ids, *(state.state_id for state in states)))
        return _ColumnRemoteLifetime([], domain=domain, state_ids=state_ids)

    def ensure_active(self) -> None:
        with self._locked_states() as states:
            if any(not state.active for state in states):
                raise RuntimeError("RemoteBatch has already been dropped")

    def drop(self) -> None:
        first_error = None
        with self._locked_states() as states:
            for state in states:
                state.active = False
                if not state.delete_pending:
                    continue
                try:
                    transfer_backend.delete(
                        partition=state.partition,
                        keys=list(state.keys),
                        fields=list(state.fields),
                    )
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                else:
                    state.delete_pending = False
        if first_error is not None:
            raise first_error.with_traceback(first_error.__traceback__)

    @contextmanager
    def _locked_states(self):
        with self._domain._locked_states(self._state_ids) as states:
            yield states


def _restore_column_remote_lifetime(domain, state_ids):
    domain.states(state_ids)
    return _ColumnRemoteLifetime([], domain=domain, state_ids=state_ids)


def _restore_column_remote_batch(
    partition,
    device,
    lifetime,
    fields,
    batch_size,
    row_ids,
    field_pipelines,
):
    return ColumnRemoteBatch(
        partition=partition,
        device=device,
        fields=fields,
        cache=None,
        batch_size=batch_size,
        row_ids=row_ids,
        field_pipelines=field_pipelines,
        lifetime=lifetime,
    )


class ColumnRemoteBatch(RemoteBatch):
    """
    A remote batch stored in a key-value store with column id as keys.

    Structural mutations are not safe to run concurrently with serialization
    or cleanup.
    """

    def __init__(
        self,
        partition: str,
        device,
        fields: dict[str, Any | list["ColumnRemoteBatch"]],
        cache: TensorDict | None,
        batch_size: int,
        row_ids: list[str] | None = None,
        field_pipelines: dict[str, tuple[PlanNode, ...]] | None = None,
        lifetime: _ColumnRemoteLifetime | None = None,
    ):
        """
        fields contains any meta need to be hold and pass to transfer backend during get
        or a list of ColumnRemoteBatch for concatenated fields.
        """
        super().__init__("column", partition, device)
        self._structure_lock = threading.RLock()
        self.fields = fields
        self.cache = cache
        self.batch_size = batch_size
        self._row_ids = list(row_ids) if row_ids is not None else None
        if self._row_ids is not None:
            assert len(self._row_ids) == batch_size
        self._field_pipelines = field_pipelines if field_pipelines is not None else {key: tuple() for key in fields}
        if lifetime is None:
            children = self._children_from_fields(fields)
            lifetime = _ColumnRemoteLifetime(self._states_from_fields(fields, children))
            lifetime.connect_many(child._lifetime for child in children)
        self._lifetime = lifetime

    def __reduce__(self):
        with self._structure_lock:
            return (
                _restore_column_remote_batch,
                (
                    self.partition,
                    self.device,
                    self._lifetime,
                    dict(self.fields),
                    self.batch_size,
                    list(self._row_ids) if self._row_ids is not None else None,
                    dict(self._field_pipelines),
                ),
            )

    @property
    def _states(self) -> list[_ColumnRemoteState]:
        return self._lifetime.states

    def __len__(self) -> int:
        return self.batch_size

    def __delitem__(self, key: str):
        with self._structure_lock:
            self.fields.pop(key)
            self._field_pipelines.pop(key)
            if self.cache is not None and key in self.cache:
                del self.cache[key]

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def row_ids(self):
        return self._row_ids

    def clone(self, recurse: bool = True):
        """Clone local state while retaining the remote object's shared lifetime."""
        self._ensure_active()
        cloned_fields = {}
        cloned_children = {}
        for key, value in self.fields.items():
            if not isinstance(value, list):
                cloned_fields[key] = value
                continue
            children = cloned_children.get(id(value))
            if children is None:
                children = [batch.clone(recurse=recurse) for batch in value]
                cloned_children[id(value)] = children
            cloned_fields[key] = children
        return ColumnRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=cloned_fields,
            cache=self.cache.clone(recurse=recurse) if self.cache is not None else None,
            batch_size=self.batch_size,
            row_ids=self._row_ids,
            field_pipelines={key: self._field_pipelines[key] for key in cloned_fields},
            # A clone may copy its local cache, but it still references the same
            # remote object and therefore shares its cleanup boundary.
            lifetime=self._lifetime,
        )

    def keys(self):
        return self.fields.keys()

    def to(self, device) -> "ColumnRemoteBatch":
        self._ensure_active()
        super().to(device)
        if self.cache is not None:
            self.cache = self.cache.to(device)
        return self

    def materialize(self, fields: list[str] = None) -> TensorDict:
        self._ensure_active()
        if fields is None:
            fields = list(self.fields)
        else:
            assert set(fields) <= set(self.fields.keys())
        if fields == []:
            return TensorDict({}, batch_size=[self.batch_size], device=self.device)
        existing_fields = set(self.cache.keys()) if self.cache is not None else set()
        missing_fields = [field for field in fields if field not in existing_fields]
        groups: dict[tuple, list[str]] = {}
        for field in missing_fields:
            source = self.fields[field]
            pipeline = self._field_pipelines[field]
            source_key = ("nested", id(source)) if isinstance(source, list) else ("direct",)
            groups.setdefault(source_key + tuple(id(op) for op in pipeline), []).append(field)

        for group_fields in groups.values():
            source = self.fields[group_fields[0]]
            pipeline = self._field_pipelines[group_fields[0]]
            if isinstance(source, list):
                data = [chunk.materialize(group_fields) for chunk in source]
            else:
                data = transfer_backend.get(
                    partition=self.partition,
                    keys=group_fields,
                    fields=[self.fields[field] for field in group_fields],
                )

            for operator in pipeline:
                data = operator.execute(data)
            assert len(data) == self.batch_size
            if self.device is not None:
                data = data.to(self.device)
            self._cache_data(data)

        return self.cache.select(*fields)

    def drop(self):
        try:
            self._lifetime.drop()
        finally:
            self.cache = None

    def select(self, fileds: list[str]) -> "ColumnRemoteBatch":
        self._ensure_active()
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
            row_ids=self._row_ids,
            field_pipelines={key: self._field_pipelines[key] for key in fields},
            lifetime=self._lifetime,
        )

    def _selection(self, plan: PlanNode, *, ensure_active: bool = True) -> "ColumnRemoteBatch":
        if ensure_active:
            self._ensure_active()
        batch_size = plan.batch_size

        cache = self.cache
        if cache is not None:
            cache = plan.execute(cache)
            assert isinstance(cache, TensorDict)

        return ColumnRemoteBatch(
            partition=self.partition,
            device=self.device,
            fields=dict(self.fields),
            cache=cache,
            batch_size=batch_size,
            row_ids=self._apply_row_plan(plan),
            field_pipelines={key: field_pipeline + (plan,) for key, field_pipeline in self._field_pipelines.items()},
            lifetime=self._lifetime,
        )

    def select_idxs(self, index: torch.Tensor | np.ndarray | list) -> "ColumnRemoteBatch":
        assert isinstance(index, (torch.Tensor, np.ndarray, list))
        if isinstance(index, np.ndarray):
            index = torch.from_numpy(index)
        elif isinstance(index, list):
            index = torch.tensor(index)
            if index.dtype != torch.bool:
                index = index.type(torch.int32)

        if index.dtype == torch.bool:
            assert (
                len(index) == self.batch_size
            ), f"Boolean index length {len(index)} does not match batch size {self.batch_size}"

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
        with self._structure_lock:
            assert len(fileds) == len(set(fileds)), "Fields must be unique"
            assert set(fileds) <= self.fields.keys(), f"Fields {set(fileds) - self.fields.keys()} not in batch"
            ret = self.select(fileds)

            for key in fileds:
                del self.fields[key]
                del self._field_pipelines[key]
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
        self._ensure_active()
        chunks = []
        offset = 0
        for size in chunk_sizes:
            chunks.append(
                self._selection(
                    SlicePlan(offset, offset + size, None, self.batch_size),
                    ensure_active=False,
                )
            )
            offset += size
        return chunks

    def repeat(self, repeat_times: int, interleave: bool) -> "ColumnRemoteBatch":
        plan = RepeatPlan(repeat_times, interleave, self.batch_size)
        return self._selection(plan)

    def union(self, rhs: "ColumnRemoteBatch") -> "ColumnRemoteBatch":
        assert isinstance(rhs, ColumnRemoteBatch)
        with self._locked_structure(self, rhs):
            self._ensure_active()
            rhs._ensure_active()
            assert len(self) == len(
                rhs
            ), f"Two tensor dict must have identical batch size. Got {len(self)} and {len(rhs)}"
            assert self.partition == rhs.partition, "Two remote batches must have the same partition"
            assert (self._row_ids is None) == (rhs._row_ids is None), "Both remote batches must provide row ids"
            if self._row_ids is not None:
                assert self._row_ids == rhs._row_ids, "Row ids must have the same order"
            adopted_fields = {field: value for field, value in rhs.fields.items() if field not in self.fields}
            adopted_cache_keys = [field for field in adopted_fields if rhs.cache is not None and field in rhs.cache]
            if adopted_cache_keys:
                adopted_cache = rhs.cache.select(*adopted_cache_keys).clone(recurse=False)
                if self.cache is None:
                    self.cache = adopted_cache
                else:
                    from roll.distributed.scheduler.protocol import union_tensor_dict

                    self.cache = union_tensor_dict(self.cache, adopted_cache)

            for field, value in adopted_fields.items():
                self.fields[field] = value
                self._field_pipelines[field] = rhs._field_pipelines[field]

            self._lifetime = self._lifetime.extend(
                rhs._lifetime,
                rhs._states_for_fields(adopted_fields),
            )

        return self

    @staticmethod
    @contextmanager
    def _locked_structure(*batches):
        locks = sorted({id(batch._structure_lock): batch._structure_lock for batch in batches}.values(), key=id)
        acquired = []
        try:
            for lock in locks:
                lock.acquire()
                acquired.append(lock)
        except BaseException:
            for lock in reversed(acquired):
                lock.release()
            raise
        try:
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

    @classmethod
    def _cat(cls, data: list["ColumnRemoteBatch"]) -> "ColumnRemoteBatch":
        """
        ColumnRemoteBatch._cat will not check type and shape[1:] of fields.
        """
        assert data
        if len(data) == 1:
            data[0]._ensure_active()
            return data[0]

        keys = set(data[0].fields.keys())
        assert all(set(d.fields.keys()) == keys for d in data), "All batches must have the same fields"
        partition = data[0].partition
        assert all(d.partition == partition for d in data), "All batches must have the same partition"
        have_row_ids = [batch._row_ids is not None for batch in data]
        assert all(have_row_ids) or not any(have_row_ids), "All batches must either provide row ids or omit them"
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

        result = ColumnRemoteBatch(
            partition=partition,
            device=device,
            fields={field: data for field in keys},
            cache=cache,
            batch_size=batch_size,
            row_ids=[row_id for batch in data for row_id in batch._row_ids] if all(have_row_ids) else None,
            field_pipelines={field: (plan,) for field in keys},
        )
        result._ensure_active()
        return result

    def _ensure_active(self) -> None:
        self._lifetime.ensure_active()

    @staticmethod
    def _collect_states(batches: list["ColumnRemoteBatch"]) -> list[_ColumnRemoteState]:
        states_by_domain = {}
        lifetimes = {id(batch._lifetime): batch._lifetime for batch in batches}
        for lifetime in lifetimes.values():
            root = lifetime._domain.root()
            domain, state_ids = states_by_domain.setdefault(id(root), (root, {}))
            for state_id in lifetime._state_ids:
                state_ids.setdefault(state_id, None)
        return [state for domain, state_ids in states_by_domain.values() for state in domain.states(state_ids)]

    def _states_from_fields(
        self,
        fields: dict[str, Any | list["ColumnRemoteBatch"]],
        children: list["ColumnRemoteBatch"] | None = None,
    ) -> list[_ColumnRemoteState]:
        states = []
        direct_keys = [key for key, value in fields.items() if not isinstance(value, list)]
        if direct_keys:
            states.append(
                _ColumnRemoteState(
                    self.partition,
                    direct_keys,
                    [fields[key] for key in direct_keys],
                )
            )
        if children is None:
            children = self._children_from_fields(fields)
        states.extend(self._collect_states(children))
        return self._unique_states(states)

    @staticmethod
    def _children_from_fields(fields):
        children = []
        seen = set()
        for value in fields.values():
            if not isinstance(value, list) or id(value) in seen:
                continue
            seen.add(id(value))
            children.extend(value)
        return children

    def _states_for_fields(
        self,
        fields: dict[str, Any | list["ColumnRemoteBatch"]],
    ) -> list[_ColumnRemoteState]:
        direct_values = {id(value) for value in fields.values() if not isinstance(value, list)}
        states = [state for state in self._states if any(id(value) in direct_values for value in state.fields)]
        nested_fields = {}
        for field, value in fields.items():
            if not isinstance(value, list):
                continue
            _, child_fields = nested_fields.setdefault(id(value), (value, []))
            child_fields.append(field)
        for children, child_fields in nested_fields.values():
            for child in children:
                selected = {field: child.fields[field] for field in child_fields if field in child.fields}
                states.extend(child._states_for_fields(selected))
        return self._unique_states(states)

    @staticmethod
    def _unique_states(states: list[_ColumnRemoteState]) -> list[_ColumnRemoteState]:
        unique = {}
        for state in states:
            unique.setdefault(state.state_id, state)
        return list(unique.values())

    def _cache_data(self, data: TensorDict) -> None:
        if self.cache is None:
            self.cache = data
            return
        from roll.distributed.scheduler.protocol import union_tensor_dict

        self.cache = union_tensor_dict(self.cache, data)

    def _apply_row_plan(self, plan: PlanNode) -> list[str] | None:
        if self._row_ids is None:
            return None
        if isinstance(plan, SelectPlan):
            index = plan.index.tolist()
            if plan.index.dtype == torch.bool:
                return [row_id for row_id, selected in zip(self._row_ids, index) if selected]
            return [self._row_ids[i] for i in index]
        if isinstance(plan, SlicePlan):
            return self._row_ids[plan.slice_obj]
        if isinstance(plan, RepeatPlan):
            if plan.interleave:
                return [row_id for row_id in self._row_ids for _ in range(plan.repeat_times)]
            return self._row_ids * plan.repeat_times
        raise TypeError(f"Unsupported row plan: {type(plan)}")
