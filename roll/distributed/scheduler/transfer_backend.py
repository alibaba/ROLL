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
        import mooncake.store  # noqa: F401
        import mooncake.structured_object_store  # noqa: F401
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

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
    ):
        return self.client.put(partition, row_ids, fields, batch_size)

    def get(self, partition, keys: list[str], fields: list[Any]):
        return self.client.get(partition, keys, fields)

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
        from mooncake.store import MooncakeDistributedStore
        from mooncake.structured_object_store import BundleTransferPolicy, MooncakeBundleTransfer

        config = config or {}
        store = MooncakeDistributedStore()
        ret = self._setup_store(store, config)
        if ret != 0:
            raise RuntimeError(f"Mooncake store setup failed, return code={ret}")

        self.backend = MooncakeBundleTransfer(store, key_prefix=config.get("key_prefix", "roll"))
        policy_config = config.get("transfer_policy")
        self.transfer_policy = BundleTransferPolicy(**policy_config) if policy_config else None

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
    ):
        from roll.distributed.scheduler.protocol import DataProto
        from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch

        batch_fields, non_tensor_fields = self._split_fields(fields)
        meta_info = {"roll_row_ids": row_ids}
        if batch_fields:
            data = DataProto.from_dict(tensors=batch_fields, non_tensors=non_tensor_fields, meta_info=meta_info)
        else:
            data = DataProto(
                batch=TensorDict({}, batch_size=[batch_size]),
                non_tensor_batch=non_tensor_fields,
                meta_info=meta_info,
            )
        assert len(data) == batch_size

        ref = self.backend.put(data, type="dataproto", partition=partition, policy=self.transfer_policy)
        field_refs = {
            key: {"ref": ref, "kind": "batch" if key in batch_fields else "non_tensor"}
            for key in fields.keys()
        }
        return ColumnRemoteBatch(
            partition=partition,
            device=data.batch.device if batch_fields else None,
            fields=field_refs,
            is_nested=False,
            cache=create_tensordict(fields),
            batch_size=batch_size,
        )

    def get(self, partition, keys: list[str], fields: list[Any]):
        if not fields:
            return TensorDict({}, batch_size=[0])

        grouped: dict[str, dict[str, Any]] = {}
        for key, field in zip(keys, fields):
            ref = field["ref"]
            group = grouped.setdefault(self._ref_key(ref), {"ref": ref, "batch": [], "non_tensor": []})
            group[field["kind"]].append(key)

        data_dict = {}
        for group in grouped.values():
            data = self.backend.get(
                group["ref"],
                type="dataproto",
                batch_fields=group["batch"],
                non_tensor_fields=group["non_tensor"],
                data_cls=dict,
            )
            data_dict.update(data.get("batch", {}))
            data_dict.update(data.get("non_tensor_batch", {}))
        return create_tensordict({key: data_dict[key] for key in keys})

    def delete(self, partition, keys: list[str], fields: list[Any]):
        deleted = set()
        for field in fields:
            ref = field["ref"]
            ref_key = self._ref_key(ref)
            if ref_key in deleted:
                continue
            self.backend.cleanup_dataproto(ref)
            deleted.add(ref_key)

    def _setup_store(self, store, config: dict[str, Any]) -> int:
        setup_args = config.get("setup_args")
        setup_kwargs = config.get("setup_kwargs")
        if setup_args is not None:
            return store.setup(*setup_args)
        if setup_kwargs is not None:
            return store.setup(**setup_kwargs)
        if config.get("master_server_addr") or config.get("master_server_address"):
            return store.setup(
                self._require_config(config, "local_hostname"),
                self._require_config(config, "metadata_server"),
                config.get("global_segment_size", 3355443200),
                config.get("local_buffer_size", 1073741824),
                config.get("protocol", "tcp"),
                config.get("rdma_devices") or config.get("device_name", ""),
                config.get("master_server_addr") or config.get("master_server_address"),
            )
        return self._setup_store_from_env(store, config)

    def _setup_store_from_env(self, store, config: dict[str, Any]) -> int:
        from mooncake.mooncake_config import MooncakeConfig

        mooncake_config = MooncakeConfig.load_from_env()
        return store.setup(
            config.get("local_hostname", mooncake_config.local_hostname),
            config.get("metadata_server", mooncake_config.metadata_server),
            config.get("global_segment_size", mooncake_config.global_segment_size),
            config.get("local_buffer_size", mooncake_config.local_buffer_size),
            config.get("protocol", mooncake_config.protocol),
            config.get("rdma_devices") or config.get("device_name", mooncake_config.device_name or ""),
            config.get("master_server_addr")
            or config.get("master_server_address", mooncake_config.master_server_address),
        )

    @staticmethod
    def _split_fields(
        fields: dict[str, torch.Tensor | np.ndarray],
    ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        tensors = {key: value for key, value in fields.items() if isinstance(value, torch.Tensor)}
        non_tensors = {key: value for key, value in fields.items() if isinstance(value, np.ndarray)}
        MooncakeClient._validate_fields(tensors, non_tensors, fields)
        return tensors, non_tensors

    @staticmethod
    def _validate_fields(
        batch_fields: dict[str, Any],
        non_tensor_fields: dict[str, Any],
        all_fields: dict[str, Any] | None = None,
    ) -> None:
        unsupported = {}
        for key, value in batch_fields.items():
            if not isinstance(value, torch.Tensor):
                unsupported[key] = type(value)
        for key, value in non_tensor_fields.items():
            if not isinstance(value, np.ndarray):
                unsupported[key] = type(value)
        if all_fields is not None:
            known = set(batch_fields) | set(non_tensor_fields)
            unsupported.update(
                {key: type(value) for key, value in all_fields.items() if key not in known}
            )
        if unsupported:
            raise TypeError(f"Unsupported Mooncake fields: {unsupported}")

    @staticmethod
    def _ref_key(ref) -> str:
        return getattr(ref, "object_id", repr(ref))

    @staticmethod
    def _require_config(config: dict[str, Any], key: str) -> Any:
        value = config.get(key)
        if value is None or value == "":
            raise ValueError(f"Mooncake backend_config requires {key!r} when master_server_addr is set")
        return value

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
