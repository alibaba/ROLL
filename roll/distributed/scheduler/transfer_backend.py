import os
import threading
import uuid
from typing import Any

import ray
import torch
import numpy as np
import sys
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

if sys.version_info < (3, 13):
    import transfer_queue as tq
else:
    tq = None
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

from roll.configs.base_config import TransferBackendArguments
from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.constants import STORAGE_NAME, RAY_NAMESPACE
from roll.utils.logging import get_logger

logger = get_logger()

MOONCAKE_CLIENT_SCOPE_NODE = "node"
MOONCAKE_CLIENT_SCOPE_PROCESS = "process"
MOONCAKE_NODE_ACTOR_NAME_PREFIX = "MooncakeNodeTransfer"

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
        import mooncake.dataproto_transfer  # noqa: F401
        import mooncake.store  # noqa: F401
    except ImportError as exc:
        raise ImportError("Mooncake transfer backend requires the mooncake Python package.") from exc


def _mooncake_client_scope(config: dict[str, Any] | None) -> str:
    return (config or {}).get("client_scope", MOONCAKE_CLIENT_SCOPE_NODE)


def _prepare_mooncake_backend_config(config: TransferBackendArguments) -> None:
    if config.backend_name != "Mooncake":
        return
    if config.backend_config is None:
        config.backend_config = {}
    if _mooncake_client_scope(config.backend_config) != MOONCAKE_CLIENT_SCOPE_NODE:
        return
    config.backend_config.setdefault("node_actor_session_id", uuid.uuid4().hex)


def init_transfer_backend(config: TransferBackendArguments | None):
    global _shared_storage

    _shared_storage = SharedStorage.options(
        name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
    ).remote()

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
            if _mooncake_client_scope(config.backend_config) == MOONCAKE_CLIENT_SCOPE_NODE:
                _client = MooncakeNodeClientProxy(config.backend_config)
            else:
                _client = MooncakeClient(config.backend_config)
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


class DummyClient:

    def put(self, partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int):
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

    def put(self, partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int):
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
            is_nested=False,
            cache=data,
            batch_size=batch_size,
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

    def put(self, partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int):
        return self.client.put(partition, row_ids, fields, batch_size)

    def get(self, partition, keys: list[str], fields: list[Any]):
        return self.client.get(partition, keys, fields)

    def delete(self, partition, keys: list[str], fields: list[Any]):
        return self.client.delete(partition, keys, fields)


class MooncakeNodeClientProxy:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.actor = self._get_node_actor()

    def put(self, partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int):
        return ray.get(self.actor.put.remote(partition, row_ids, fields, batch_size))

    def get(self, partition, keys: list[str], fields: list[Any]):
        return ray.get(self.actor.get.remote(partition, keys, fields))

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
            max_concurrency=1,
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(self.config)


class MooncakeClient:
    def __init__(self, config: dict[str, Any] | None = None):
        _check_mooncake_available()
        from mooncake.dataproto_transfer import MooncakeDataProtoTransferBackend
        from mooncake.store import MooncakeDistributedStore
        from roll.distributed.scheduler.protocol import DataProto

        config = config or {}
        store = MooncakeDistributedStore()
        setup_args = config.get("setup_args")
        setup_kwargs = config.get("setup_kwargs")
        if setup_args is not None:
            ret = store.setup(*setup_args)
        elif setup_kwargs is not None:
            ret = store.setup(**setup_kwargs)
        else:
            ret = self._setup_store_from_env(store)
        if ret != 0:
            raise RuntimeError(f"Mooncake store setup failed, return code={ret}")

        self.backend = MooncakeDataProtoTransferBackend(
            store,
            key_prefix=config.get("key_prefix", "roll"),
            data_cls=DataProto,
        )
        self.shard_policy = self._create_shard_policy(config.get("shard_policy"))

    def put(self, partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int):
        from roll.distributed.scheduler.protocol import DataProto
        from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch

        tensors = {key: value for key, value in fields.items() if isinstance(value, torch.Tensor)}
        non_tensors = {key: value for key, value in fields.items() if isinstance(value, np.ndarray)}
        if len(tensors) + len(non_tensors) != len(fields):
            unsupported = {
                key: type(value) for key, value in fields.items() if key not in tensors and key not in non_tensors
            }
            raise TypeError(f"Unsupported Mooncake fields: {unsupported}")

        if tensors:
            data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={})
        else:
            data = DataProto(batch=None, non_tensor_batch=non_tensors, meta_info={})
        assert len(data) == batch_size
        ref = self.backend.put_dataproto(data, partition=partition, shard_policy=self.shard_policy)
        field_refs = {
            key: {"ref": ref, "kind": "batch" if key in tensors else "non_tensor"}
            for key in fields.keys()
        }
        return ColumnRemoteBatch(
            partition=partition,
            device=data.batch.device if tensors else None,
            fields=field_refs,
            is_nested=False,
            cache=create_tensordict(fields),
            batch_size=batch_size,
        )

    def get(self, partition, keys: list[str], fields: list[Any]):
        if not fields:
            return TensorDict({}, batch_size=[0])
        ref = fields[0]["ref"]
        if any(field["ref"].object_id != ref.object_id for field in fields):
            raise ValueError("Mooncake backend cannot materialize fields from different refs in one get")
        batch_fields = [key for key, field in zip(keys, fields) if field["kind"] == "batch"]
        non_tensor_fields = [key for key, field in zip(keys, fields) if field["kind"] == "non_tensor"]
        data = self.backend.materialize_dataproto(
            ref,
            batch_fields=batch_fields,
            non_tensor_fields=non_tensor_fields,
            include_meta_info=False,
        )
        data_dict = {}
        if data._batch is not None:
            data_dict.update(data._batch.to_dict())
        data_dict.update(data._non_tensor_batch)
        return create_tensordict({key: data_dict[key] for key in keys})

    def delete(self, partition, keys: list[str], fields: list[Any]):
        deleted = set()
        for field in fields:
            ref = field["ref"]
            if ref.object_id in deleted:
                continue
            self.backend.remove_dataproto(ref)
            deleted.add(ref.object_id)

    def _setup_store_from_env(self, store):
        from mooncake.mooncake_config import MooncakeConfig

        config = MooncakeConfig.load_from_env()
        return store.setup(
            config.local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name or "",
            config.master_server_address,
        )

    def _create_shard_policy(self, config: dict[str, Any] | None):
        if not config:
            return None
        from mooncake.dataproto_transfer import DataProtoShardPolicy

        return DataProtoShardPolicy(**config)


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

    def put(self, partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int):
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
