import os
import threading
import uuid
from typing import Any

import ray
import torch
import numpy as np
import sys

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

# Global reference to keep SharedStorage actor alive
_shared_storage = None


def _check_transfer_queue_available():
    if tq is None:
        raise ImportError(
            "TransferQueue is not available on Python 3.13+. "
            "Please use an alternative transfer backend or downgrade to Python <= 3.12."
        )


def init_transfer_backend(config: TransferBackendArguments | None):
    global _shared_storage

    _shared_storage = SharedStorage.options(
        name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
    ).remote()

    if config is None:
        config = TransferBackendArguments()
    ray.get(_shared_storage.put.remote(key="transfer_backend_config", data=config))

    backend_name = config.backend_name
    backend_config = config.backend_config
    if backend_name is None:
        logger.info(f"Initialized dummy transfer backend: {config}")
    elif backend_name == "TransferQueue":
        _check_transfer_queue_available()
        init_transfer_queue_server(backend_config)
        logger.info(f"Initialized TransferQueue transfer backend: {config}")
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
