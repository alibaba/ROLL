import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

from roll.configs.base_config import TransferBackendArguments
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.transfer_backend import (
    MOONCAKE_CLIENT_SCOPE_NODE,
    MOONCAKE_CLIENT_SCOPE_PROCESS,
    MooncakeClient,
    _mooncake_client_scope,
    _prepare_mooncake_backend_config,
)


class FakeMooncakeStore:
    def setup(self, *args, **kwargs):
        return 0


class FakeMooncakeBackend:
    refs = {}
    removed = []

    def __init__(self, store, key_prefix="dataproto", data_cls=None):
        self.data_cls = data_cls

    def put_dataproto(self, data, partition="default", shard_policy=None):
        object_id = f"{partition}/ref"
        ref = SimpleNamespace(object_id=object_id, row_count=len(data), manifest_key="manifest", manifest={})
        self.refs[object_id] = data
        return ref

    def materialize_dataproto(
        self,
        ref,
        batch_fields=None,
        non_tensor_fields=None,
        include_meta_info=True,
    ):
        data = self.refs[ref.object_id]
        tensors = {}
        if data._batch is not None:
            selected = set(batch_fields or [])
            tensors = {key: value for key, value in data._batch.to_dict().items() if key in selected}
        non_tensors = {
            key: value for key, value in data._non_tensor_batch.items() if key in set(non_tensor_fields or [])
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors) if tensors else DataProto(
            batch=None, non_tensor_batch=non_tensors
        )

    def remove_dataproto(self, ref):
        self.removed.append(ref.object_id)


class FakeShardPolicy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_mooncake_client_scope_defaults_to_node():
    config = TransferBackendArguments(backend_name="Mooncake", backend_config={})

    _prepare_mooncake_backend_config(config)

    assert _mooncake_client_scope(config.backend_config) == MOONCAKE_CLIENT_SCOPE_NODE
    assert config.backend_config["node_actor_session_id"]


def test_mooncake_process_scope_keeps_config_small():
    config = TransferBackendArguments(
        backend_name="Mooncake",
        backend_config={"client_scope": MOONCAKE_CLIENT_SCOPE_PROCESS},
    )

    _prepare_mooncake_backend_config(config)

    assert "node_actor_session_id" not in config.backend_config


def test_mooncake_client_round_trip(monkeypatch):
    mooncake = types.ModuleType("mooncake")
    store_mod = types.ModuleType("mooncake.store")
    transfer_mod = types.ModuleType("mooncake.dataproto_transfer")
    store_mod.MooncakeDistributedStore = FakeMooncakeStore
    transfer_mod.MooncakeDataProtoTransferBackend = FakeMooncakeBackend
    transfer_mod.DataProtoShardPolicy = FakeShardPolicy
    monkeypatch.setitem(sys.modules, "mooncake", mooncake)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_mod)
    monkeypatch.setitem(sys.modules, "mooncake.dataproto_transfer", transfer_mod)

    FakeMooncakeBackend.refs = {}
    FakeMooncakeBackend.removed = []
    client = MooncakeClient({"setup_args": [], "shard_policy": {"enabled": True}})
    fields = {
        "tokens": torch.tensor([[1, 2], [3, 4]]),
        "prompt": np.array(["a", "b"], dtype=object),
    }

    remote = client.put("rollout", ["0", "1"], fields, batch_size=2)
    materialized = client.get("rollout", ["tokens", "prompt"], [remote.fields["tokens"], remote.fields["prompt"]])

    assert torch.equal(materialized["tokens"], fields["tokens"])
    assert list(materialized["prompt"]) == ["a", "b"]

    client.delete("rollout", list(remote.fields.keys()), list(remote.fields.values()))
    assert FakeMooncakeBackend.removed == ["rollout/ref"]
