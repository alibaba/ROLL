import os
import sys
import threading
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

if sys.version_info < (3, 13):
    import transfer_queue as tq
else:
    tq = None

from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict
from tensordict.utils import LinkedList

from roll.configs.base_config import TransferBackendArguments
from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.constants import RAY_NAMESPACE, STORAGE_NAME
from roll.utils.logging import get_logger

logger = get_logger()

MOONCAKE_CLIENT_SCOPE_NODE = "node"
MOONCAKE_NODE_ACTOR_NAME_PREFIX = "MooncakeNodeTransfer"
_MOONCAKE_RELEASE_RESULTS_ATTR = "_mooncake_release_results"

# Global reference to keep SharedStorage actor alive
_shared_storage = None


def _check_transfer_queue_available():
    if tq is None:
        raise ImportError(
            "TransferQueue is not available on Python 3.13+. "
            "Please use an alternative transfer backend or downgrade to Python <= 3.12."
        )


def _check_mooncake_available():
    try:
        import mooncake.store  # noqa: F401
        import mooncake.structured_object_store  # noqa: F401
    except ImportError as exc:
        raise ImportError("Mooncake transfer backend requires the mooncake Python package.") from exc


def _mooncake_client_scope(config: dict[str, Any] | None) -> str:
    scope = (config or {}).get("client_scope", MOONCAKE_CLIENT_SCOPE_NODE)
    if scope != MOONCAKE_CLIENT_SCOPE_NODE:
        raise ValueError(f"Mooncake currently supports only client_scope={MOONCAKE_CLIENT_SCOPE_NODE!r}")
    return scope


def _prepare_mooncake_backend_config(config: TransferBackendArguments) -> None:
    if config.backend_name != "Mooncake":
        return
    if config.backend_config is None:
        config.backend_config = {}
    _mooncake_client_scope(config.backend_config)
    config.backend_config.setdefault("node_actor_session_id", uuid.uuid4().hex)


def init_transfer_backend(config: TransferBackendArguments | None):
    global _shared_storage

    _shared_storage = SharedStorage.options(name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE).remote()

    if config is None:
        config = TransferBackendArguments()
    _prepare_mooncake_backend_config(config)
    ray.get(_shared_storage.put.remote(key="transfer_backend_config", data=config))

    backend_name = config.backend_name
    backend_config = config.backend_config
    if backend_name is None:
        logger.info(f"Initialized dummy transfer backend: {config}")
    elif backend_name == "TransferQueue":
        _check_transfer_queue_available()
        init_transfer_queue_server(backend_config)
        logger.info(f"Initialized TransferQueue transfer backend: {config}")
    elif backend_name == "Mooncake":
        _check_mooncake_available()
        logger.info(f"Initialized Mooncake transfer backend: {config}")
    else:
        raise ValueError(f"Unsupported transfer backend: {backend_name}")


_client = None
_client_lock = threading.Lock()


def reinit_after_fork():
    global _client, _client_lock
    _client_lock = threading.Lock()
    _client = None


os.register_at_fork(after_in_child=reinit_after_fork)


def init_client():
    global _client
    if _client is not None:
        return
    with _client_lock:
        if _client is not None:
            return
        shared_storage = ray.get_actor(name=STORAGE_NAME, namespace=RAY_NAMESPACE)
        config = ray.get(shared_storage.get.remote(key="transfer_backend_config"))
        assert config is not None
        if config.backend_name is None:
            _client = DummyClient()
        elif config.backend_name == "TransferQueue":
            _client = TransferQueueClient()
        elif config.backend_name == "Mooncake":
            _mooncake_client_scope(config.backend_config)
            _client = MooncakeNodeClientProxy(config.backend_config)
        else:
            raise ValueError(f"Unsupported transfer backend: {config.backend_name}")
        logger.info(f"Initialized transfer client: {_client.__class__.__name__}")


def put(partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int):
    init_client()
    return _client.put(partition, row_ids, fields, batch_size)


def get(partition, keys: list[str], fields: list[Any]):
    init_client()
    return _client.get(partition, keys, fields)


def delete(partition, keys: list[str], fields: list[Any]):
    init_client()
    return _client.delete(partition, keys, fields)


def create_tensordict(fields: dict[str, torch.Tensor | np.ndarray]) -> TensorDict:
    assert fields
    td_dict = {}
    batch_size = None
    for key, val in fields.items():
        if isinstance(val, torch.Tensor):
            td_dict[key] = val
        elif isinstance(val, np.ndarray):
            td_dict[key] = NonTensorStack(*val)
        else:
            raise TypeError(f"Unsupported type: {type(val)}")
        if batch_size is None:
            batch_size = val.shape[0]
        elif batch_size != val.shape[0]:
            raise ValueError("Batch size mismatch")
    return TensorDict(td_dict, batch_size=[batch_size])


def _detach_mooncake_pool_value(value: Any) -> Any:
    """Remove process-local pool owners without copying numeric payloads."""
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype != object:
            return array
        detached = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            detached[index] = _detach_mooncake_pool_value(array[index])
        return detached
    if isinstance(value, list):
        return [_detach_mooncake_pool_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_mooncake_pool_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _detach_mooncake_pool_value(item) for key, item in value.items()}
    return value


def _detach_mooncake_pool_owners(data: TensorDict) -> TensorDict:
    """Build a Ray-serializable view before returning data from the node actor."""
    detached = {}
    for key, value in data.items():
        if isinstance(value, (NonTensorStack, LinkedList)):
            items = value.tolist() if isinstance(value, NonTensorStack) else value
            detached[key] = NonTensorStack(*(_detach_mooncake_pool_value(item) for item in items))
        else:
            detached[key] = value
    return TensorDict(detached, batch_size=data.batch_size, device=data.device)


class DummyClient:
    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
    ):
        return None

    def get(self, partition, keys: list[str], fields: list[Any]):
        raise RuntimeError("unexpected code path")

    def delete(self, partition, keys: list[str], fields: list[Any]):
        raise RuntimeError("unexpected code path")


@ray.remote
class RayMemoryStoreServer:
    def __init__(self):
        super().__init__()
        self.objects: dict[str, torch.Tensor | np.ndarray] = {}

    async def put(self, keys, values):
        for key, data in zip(keys, values):
            self.objects[key] = data

    async def get(self, keys):
        return [self.objects[key] for key in keys]

    async def delete(self, keys):
        for key in keys:
            del self.objects[key]


class RayMemoryStoreClient:
    def __init__(self):
        self.client = RayMemoryStoreServer.options(
            name="RayMemoryStore",
            get_if_exists=True,
        ).remote()

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
    ):
        # TODO move RayMemoryStoreClient to another file
        from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch

        column_ids = [str(uuid.uuid4()) for _ in range(len(fields))]
        ray.get(self.client.put.remote(keys=column_ids, values=list(fields.values())))

        meta_dict = {field: column_id for field, column_id in zip(fields.keys(), column_ids)}
        data = create_tensordict(fields)
        assert len(data) == batch_size
        return ColumnRemoteBatch(
            partition=partition,
            device=None,
            fields=meta_dict,
            cache=data,
            batch_size=batch_size,
            row_ids=row_ids,
        )

    def get(self, partition, keys: list[str], fields: list[Any]):
        data_list = ray.get(self.client.get.remote(fields))
        data_dict = {field: tensor for field, tensor in zip(keys, data_list)}
        return create_tensordict(data_dict)

    def delete(self, partition, keys: list[str], fields: list[Any]):
        pass


@ray.remote
class MooncakeNodeTransferActor:
    def __init__(self, config: dict[str, Any] | None = None):
        self.client = MooncakeClient(config)

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
    ):
        return self.client.put(partition, row_ids, fields, batch_size)

    def get(self, partition, keys: list[str], fields: list[Any]):
        data = self.client.get(partition, keys, fields)
        object_ref = None
        serialization_error = None
        try:
            # The pool lease belongs to this actor process. ray.put() finishes
            # serialization before the lease is returned to the BufferPool.
            object_ref = ray.put(_detach_mooncake_pool_owners(data))
        except Exception as exc:
            serialization_error = exc
        try:
            self.client.release(data)
        except Exception:
            if serialization_error is None:
                raise
            logger.exception("Failed to release a Mooncake GET result after serialization failed")
        if serialization_error is not None:
            raise serialization_error.with_traceback(serialization_error.__traceback__)
        return object_ref

    def delete(self, partition, keys: list[str], fields: list[Any]):
        return self.client.delete(partition, keys, fields)


class MooncakeNodeClientProxy:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.actor = self._get_node_actor()

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
    ):
        remote = ray.get(self.actor.put.remote(partition, row_ids, fields, batch_size))
        remote.cache = create_tensordict(fields)
        return remote

    def get(self, partition, keys: list[str], fields: list[Any]):
        return ray.get(ray.get(self.actor.get.remote(partition, keys, fields)))

    def delete(self, partition, keys: list[str], fields: list[Any]):
        return ray.get(self.actor.delete.remote(partition, keys, fields))

    def _get_node_actor(self):
        node_id = ray.get_runtime_context().get_node_id()
        session_id = self.config.get("node_actor_session_id", "default")
        actor_name = f"{MOONCAKE_NODE_ACTOR_NAME_PREFIX}-{session_id}-{node_id[:16]}"
        return MooncakeNodeTransferActor.options(
            name=actor_name,
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
            # One client owns the node BufferPool and release-retry queue;
            # serialize operations so materialization cannot race cleanup.
            max_concurrency=1,
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(self.config)


class MooncakeClient:
    def __init__(self, config: dict[str, Any] | None = None):
        _check_mooncake_available()
        from mooncake.store import MooncakeDistributedStore
        from mooncake.structured_object_store import (
            BundleTransferPolicy,
            FieldSchema,
            MooncakeBundleTransfer,
        )

        config = config or {}
        store = MooncakeDistributedStore()
        ret = self._setup_store(store, config)
        if ret != 0:
            raise RuntimeError(f"Mooncake store setup failed, return code={ret}")

        self.backend = MooncakeBundleTransfer(store, key_prefix=config.get("key_prefix", "roll"))
        policy_config = config.get("transfer_policy")
        self.transfer_policy = BundleTransferPolicy(**policy_config) if policy_config else None
        self.field_schemas = self._build_field_schemas(config.get("field_schemas"), FieldSchema)
        self._pending_release_results = []

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
    ):
        self._retry_pending_release_results()
        from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch

        batch_fields, non_tensor_fields = self._split_fields(fields)
        if len(row_ids) != batch_size:
            raise ValueError(f"Expected {batch_size} row ids, got {len(row_ids)}")
        invalid_lengths = {key: len(value) for key, value in fields.items() if len(value) != batch_size}
        if invalid_lengths:
            raise ValueError(f"Mooncake field lengths do not match batch_size={batch_size}: {invalid_lengths}")
        data = {
            "batch": batch_fields,
            "non_tensor_batch": non_tensor_fields,
            "meta_info": {},
        }

        put_kwargs = {"type": "dataproto", "partition": partition, "policy": self.transfer_policy}
        if self.field_schemas:
            put_kwargs["field_schemas"] = self.field_schemas
        ref = self.backend.put(data, **put_kwargs)
        field_refs = {
            key: {"ref": ref, "kind": "batch" if key in batch_fields else "non_tensor"} for key in fields.keys()
        }
        return ColumnRemoteBatch(
            partition=partition,
            device=next(iter(batch_fields.values())).device if batch_fields else None,
            fields=field_refs,
            cache=None,
            batch_size=batch_size,
            row_ids=row_ids,
        )

    def get(self, partition, keys: list[str], fields: list[Any]):
        self._retry_pending_release_results()
        if not fields:
            return TensorDict({}, batch_size=[0])

        grouped: dict[Any, dict[str, Any]] = {}
        for key, field in zip(keys, fields):
            ref = field["ref"]
            group = grouped.setdefault(self._ref_key(ref), {"ref": ref, "batch": [], "non_tensor": []})
            group[field["kind"]].append(key)

        raw_results = []
        try:
            data_dict = {}
            for group in grouped.values():
                data = self.backend.get(
                    group["ref"],
                    type="dataproto",
                    batch_fields=group["batch"],
                    non_tensor_fields=group["non_tensor"],
                    meta_info_keys=[],
                    data_cls=dict,
                )
                raw_results.append(data)
                data_dict.update(data.get("batch", {}))
                data_dict.update(data.get("non_tensor_batch", {}))
            result = create_tensordict({key: data_dict[key] for key in keys})
        except Exception:
            self._release_raw_results(raw_results, suppress_errors=True)
            self._pending_release_results.extend(raw_results)
            raise

        setattr(result, _MOONCAKE_RELEASE_RESULTS_ATTR, raw_results)
        return result

    def release(self, data: TensorDict):
        raw_results = getattr(data, _MOONCAKE_RELEASE_RESULTS_ATTR, None)
        if raw_results is None:
            self._retry_pending_release_results()
            return
        try:
            self._release_raw_results(raw_results)
        except Exception:
            self._pending_release_results.extend(raw_results)
            raise
        finally:
            delattr(data, _MOONCAKE_RELEASE_RESULTS_ATTR)
        self._retry_pending_release_results()

    def delete(self, partition, keys: list[str], fields: list[Any]):
        refs = []
        seen = set()
        for field in fields:
            if not isinstance(field, Mapping) or "ref" not in field:
                raise TypeError(f"Unsupported Mooncake field reference: {type(field)}")
            ref = field["ref"]
            ref_key = self._ref_key(ref)
            if ref_key in seen:
                continue
            seen.add(ref_key)
            refs.append(ref)

        try:
            self._retry_pending_release_results()
        except Exception:
            logger.warning("Pending Mooncake GET release still failed; continuing object cleanup", exc_info=True)

        first_error = None
        for ref in refs:
            try:
                self.backend.cleanup_dataproto(ref)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error.with_traceback(first_error.__traceback__)

    def _setup_store(self, store, config: dict[str, Any]) -> int:
        setup_args = config.get("setup_args")
        setup_kwargs = config.get("setup_kwargs")
        if setup_args is not None:
            return store.setup(*setup_args)
        if setup_kwargs is not None:
            return store.setup(**setup_kwargs)
        protocol = config.get("protocol", "tcp")
        local_hostname = config.get("local_hostname")
        device_name = config.get("rdma_devices") or config.get("device_name")
        master_address = config.get("master_server_addr") or config.get("master_server_address")
        mooncake_config = None
        if not local_hostname or not master_address or (protocol == "rdma" and not device_name):
            from mooncake.mooncake_config import MooncakeConfig

            mooncake_config = MooncakeConfig.load_from_env()

        def configured(name: str, default=None):
            value = config.get(name)
            if value is not None:
                return value
            return getattr(mooncake_config, name, default) if mooncake_config is not None else default

        return store.setup(
            configured("local_hostname"),
            configured("metadata_server", "P2PHANDSHAKE"),
            configured("global_segment_size", 3355443200),
            configured("local_buffer_size", 1073741824),
            configured("protocol", "tcp"),
            config.get("rdma_devices") or configured("device_name", ""),
            config.get("master_server_addr") or configured("master_server_address"),
        )

    @staticmethod
    def _split_fields(
        fields: dict[str, torch.Tensor | np.ndarray],
    ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        unsupported = {
            key: type(value) for key, value in fields.items() if not isinstance(value, (torch.Tensor, np.ndarray))
        }
        if unsupported:
            raise TypeError(f"Unsupported Mooncake fields: {unsupported}")
        tensors = {key: value for key, value in fields.items() if isinstance(value, torch.Tensor)}
        non_tensors = {key: value for key, value in fields.items() if isinstance(value, np.ndarray)}
        return tensors, non_tensors

    @staticmethod
    def _build_field_schemas(schema_config: Mapping[str, Any] | None, field_schema_cls) -> dict[str, Any]:
        schemas = {}
        for name, spec in (schema_config or {}).items():
            if not isinstance(spec, Mapping):
                raise TypeError(f"Mooncake field schema {name!r} must be a mapping")
            schemas[name] = field_schema_cls(**dict(spec))
        return schemas

    def _release_raw_results(self, results: list[Any], suppress_errors: bool = False) -> None:
        pending = []
        first_error = None
        for result in results:
            try:
                self.backend.release_result(result)
            except Exception as exc:
                pending.append(result)
                if first_error is None:
                    first_error = exc
                if suppress_errors:
                    logger.exception("Failed to release a Mooncake GET result")
        results[:] = pending
        if first_error is not None and not suppress_errors:
            raise first_error.with_traceback(first_error.__traceback__)

    def _retry_pending_release_results(self) -> None:
        pending, self._pending_release_results = self._pending_release_results, []
        try:
            self._release_raw_results(pending)
        finally:
            self._pending_release_results.extend(pending)

    @staticmethod
    def _ref_key(ref) -> Any:
        stage_refs = getattr(ref, "stage_refs", None)
        if stage_refs is not None:
            return tuple(sorted((stage, bundle.manifest_key) for stage, bundle in stage_refs.items()))
        return getattr(ref, "object_id", repr(ref))


def init_transfer_queue_server(config):
    # Must create enough storage units or may encounter:
    # EncodeError: Can't encode Ext objects with data longer than 2**32 - 1.
    # But also cannot set too many storage units that exceed the number of cores of ray cluster.
    config = OmegaConf.create(config)
    tq.init(config)


class TransferQueueClient:
    def __init__(self):
        _check_transfer_queue_available()
        tq.init()

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
    ):
        # TODO move TransferQueueClient to another file
        from roll.distributed.scheduler.remote_protocol import RowRemoteBatch

        data = create_tensordict(fields)
        assert len(data) == batch_size
        tq.kv_batch_put(
            keys=row_ids,
            fields=data,
            partition_id=partition,
        )
        return RowRemoteBatch(
            partition=partition,
            device=data.device,
            fields=list(fields.keys()),
            row_ids=row_ids,
            cache=data,
        )

    def get(self, partition, keys: list[str], fields: list[Any]):
        return tq.kv_batch_get(keys=keys, select_fields=fields, partition_id=partition)

    def delete(self, partition, keys: list[str], fields: list[Any]):
        return tq.kv_clear(keys=keys, partition_id=partition)


__all__ = ["init_transfer_backend", "put", "get", "delete"]
