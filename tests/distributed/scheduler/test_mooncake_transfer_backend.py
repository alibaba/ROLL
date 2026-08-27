import os
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import uuid
from collections import UserDict

import numpy as np
import pytest
import ray
import torch
from tensordict import TensorDict

from roll.configs.base_config import TransferBackendArguments
from roll.distributed.scheduler import transfer_backend
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch
from roll.distributed.scheduler.transfer_backend import (
    MOONCAKE_CLIENT_SCOPE_NODE,
    MooncakeClient,
    MooncakeNodeClientProxy,
    MooncakeNodeTransferActor,
    _detach_mooncake_pool_owners,
    _mooncake_client_scope,
    _prepare_mooncake_backend_config,
)


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _mooncake_master_endpoint() -> tuple[str, int]:
    master = os.environ.get("MOONCAKE_MASTER", "")
    if not master:
        pytest.skip("Set MOONCAKE_MASTER to run the Mooncake RDMA backend test")
    host, port = master.rsplit(":", 1)
    return host, int(port)


@pytest.fixture(scope="module")
def mooncake_master():
    host, port = _mooncake_master_endpoint()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, port)) == 0:
            yield f"{host}:{port}"
            return

    if shutil.which("mooncake_master") is None:
        pytest.skip("mooncake_master is not available in PATH and the configured master is unreachable")

    with tempfile.TemporaryFile(mode="w+") as master_log:
        process = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_address={host}",
                f"--rpc_port={port}",
                "--logtostderr=true",
            ],
            stdout=master_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            try:
                _wait_for_port(host, port)
            except TimeoutError as exc:
                master_log.seek(0)
                raise RuntimeError(f"mooncake_master did not start:\n{master_log.read()}") from exc
            yield f"{host}:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _column_remote(fields, batch_size, *, cache=None, row_ids=None, partition="rollout"):
    return ColumnRemoteBatch(
        partition=partition,
        device=None,
        fields=fields,
        cache=cache,
        batch_size=batch_size,
        row_ids=row_ids,
    )


def test_mooncake_client_scope_defaults_to_node():
    config = TransferBackendArguments(backend_name="Mooncake", backend_config={})

    _prepare_mooncake_backend_config(config)

    assert _mooncake_client_scope(config.backend_config) == MOONCAKE_CLIENT_SCOPE_NODE
    assert config.backend_config["node_actor_session_id"]


@pytest.mark.parametrize("scope", ["process", "per_worker"])
def test_mooncake_client_scope_rejects_unsupported_values(scope):
    with pytest.raises(ValueError, match="client_scope"):
        _mooncake_client_scope({"client_scope": scope})


def test_mooncake_client_splits_roll_fields():
    fields = {
        "tokens": torch.tensor([[1], [2]]),
        "prompt": np.array(["a", "b"], dtype=object),
    }

    tensors, non_tensors = MooncakeClient._split_fields(fields)

    assert tensors == {"tokens": fields["tokens"]}
    assert non_tensors == {"prompt": fields["prompt"]}


def test_mooncake_client_rejects_unsupported_fields():
    with pytest.raises(TypeError, match="Unsupported Mooncake fields"):
        MooncakeClient._split_fields({"bad": ["a", "b"]})


def test_mooncake_client_builds_and_validates_field_schemas():
    class FakeFieldSchema:
        def __init__(self, codec, nullable=True, metadata=None):
            self.codec = codec
            self.nullable = nullable
            self.metadata = metadata or {}

    schemas = MooncakeClient._build_field_schemas(
        UserDict(
            {
                "tokens": UserDict(
                    {
                        "codec": "ragged_tensor",
                        "nullable": False,
                        "metadata": {"section": "non_tensor_batch"},
                    }
                ),
            }
        ),
        FakeFieldSchema,
    )

    assert schemas["tokens"].codec == "ragged_tensor"
    assert schemas["tokens"].nullable is False
    assert schemas["tokens"].metadata == {"section": "non_tensor_batch"}

    with pytest.raises(TypeError):
        MooncakeClient._build_field_schemas({"tokens": {}}, FakeFieldSchema)
    with pytest.raises(TypeError):
        MooncakeClient._build_field_schemas(
            {"tokens": {"codec": "ragged_tensor", "unknown": True}},
            FakeFieldSchema,
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        MooncakeClient._build_field_schemas({"tokens": "ragged_tensor"}, FakeFieldSchema)


def test_mooncake_pool_owners_are_detached_without_copying_numeric_payloads():
    class PoolBackedArray(np.ndarray):
        pass

    row = np.arange(4, dtype=np.int64).view(PoolBackedArray)
    row._mooncake_pool_owner = object()
    rows = np.empty(1, dtype=object)
    rows[0] = row
    data = transfer_backend.create_tensordict({"items": torch.tensor([[1]]), "rows": rows})

    detached = _detach_mooncake_pool_owners(data)
    detached_row = detached["rows"][0]

    assert np.array_equal(detached_row, row)
    assert np.shares_memory(detached_row, row)
    assert not hasattr(detached_row, "_mooncake_pool_owner")


def test_mooncake_client_uses_unified_dataproto_api(monkeypatch):
    calls = []

    class FakeStore:
        def setup(self, *args, **kwargs):
            return 0

    class FakePolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeFieldSchema:
        def __init__(self, codec, nullable=True, metadata=None):
            self.codec = codec
            self.nullable = nullable
            self.metadata = metadata or {}

    class FakeTransfer:
        cleanup_calls = []
        fail_release = False
        release_calls = []

        def __init__(self, store, key_prefix):
            self.store = store
            self.key_prefix = key_prefix

        def put(self, data, **kwargs):
            calls.append(("put", kwargs, data))
            return "ref"

        def get(self, ref, **kwargs):
            calls.append(("get", kwargs, ref))
            return {
                "batch": {"tokens": torch.tensor([[1], [2]])},
                "non_tensor_batch": {"prompt": np.array(["a", "b"], dtype=object)},
            }

        def cleanup_dataproto(self, ref):
            self.cleanup_calls.append(ref)
            if ref == "bad":
                raise RuntimeError("cleanup failed")

        def release_result(self, result):
            self.release_calls.append(result)
            if self.fail_release or result.get("release_error"):
                raise RuntimeError("release failed")

    mooncake_module = types.ModuleType("mooncake")
    store_module = types.ModuleType("mooncake.store")
    structured_module = types.ModuleType("mooncake.structured_object_store")
    store_module.MooncakeDistributedStore = FakeStore
    structured_module.BundleTransferPolicy = FakePolicy
    structured_module.FieldSchema = FakeFieldSchema
    structured_module.MooncakeBundleTransfer = FakeTransfer
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setitem(sys.modules, "mooncake.structured_object_store", structured_module)

    client = MooncakeClient(
        {
            "local_hostname": "127.0.0.1",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
            "master_server_addr": "127.0.0.1:50051",
            "field_schemas": {
                "multi_modal_inputs": {
                    "codec": "ragged_tensor_dict",
                    "metadata": {"section": "non_tensor_batch", "keys": {"image": None}},
                }
            },
        }
    )

    prompt = np.array(["a", "b"], dtype=object)
    remote = client.put(
        "rollout",
        ["0", "1"],
        {
            "tokens": torch.tensor([[1], [2]]),
            "prompt": prompt,
        },
        batch_size=2,
    )
    materialized = client.get("rollout", ["tokens", "prompt"], [remote.fields["tokens"], remote.fields["prompt"]])

    put_name, put_kwargs, put_data = calls[0]
    get_name, get_kwargs, get_ref = calls[1]
    assert put_name == "put"
    assert put_kwargs["type"] == "dataproto"
    assert put_kwargs["partition"] == "rollout"
    schema = put_kwargs["field_schemas"]["multi_modal_inputs"]
    assert schema.codec == "ragged_tensor_dict"
    assert schema.metadata == {"section": "non_tensor_batch", "keys": {"image": None}}
    assert put_data["meta_info"] == {}
    assert set(put_data["batch"]) == {"tokens"}
    assert set(put_data["non_tensor_batch"]) == {"prompt"}
    assert put_data["non_tensor_batch"]["prompt"] is prompt
    assert get_name == "get"
    assert get_ref == "ref"
    assert get_kwargs == {
        "type": "dataproto",
        "batch_fields": ["tokens"],
        "non_tensor_fields": ["prompt"],
        "meta_info_keys": [],
        "data_cls": dict,
    }
    assert torch.equal(materialized["tokens"], torch.tensor([[1], [2]]))
    assert list(materialized["prompt"]) == ["a", "b"]

    client.release(materialized)
    client.release(materialized)
    assert len(FakeTransfer.release_calls) == 1

    non_tensor_remote = client.put(
        "rollout",
        ["0", "1"],
        {"prompt": np.array(["a", "b"], dtype=object)},
        batch_size=2,
    )
    assert non_tensor_remote.fields["prompt"]["kind"] == "non_tensor"
    assert len(calls[-1][2]["non_tensor_batch"]["prompt"]) == 2
    non_tensor_data = client.get(
        "rollout",
        ["prompt"],
        [non_tensor_remote.fields["prompt"]],
    )
    assert list(non_tensor_data["prompt"]) == ["a", "b"]
    client.release(non_tensor_data)

    with pytest.raises(ValueError, match="Expected 2 row ids"):
        client.put("rollout", ["0"], {"tokens": torch.tensor([[1], [2]])}, batch_size=2)
    with pytest.raises(ValueError, match="field lengths"):
        client.put("rollout", ["0", "1"], {"tokens": torch.tensor([[1]])}, batch_size=2)

    retry_data = client.get("rollout", ["tokens"], [remote.fields["tokens"]])
    FakeTransfer.fail_release = True
    with pytest.raises(RuntimeError, match="release failed"):
        client.release(retry_data)
    assert len(client._pending_release_results) == 1
    client.delete(
        "rollout",
        ["tokens"],
        [{"ref": "release-cleanup", "kind": "batch"}],
    )
    assert FakeTransfer.cleanup_calls == ["release-cleanup"]
    FakeTransfer.fail_release = False
    client.release(retry_data)
    assert client._pending_release_results == []

    release_count = len(FakeTransfer.release_calls)
    FakeTransfer.fail_release = True
    with pytest.raises(KeyError, match="missing"):
        client.get("rollout", ["missing"], [{"ref": "ref", "kind": "batch"}])
    assert len(FakeTransfer.release_calls) == release_count + 1
    assert len(client._pending_release_results) == 1
    FakeTransfer.fail_release = False
    client._retry_pending_release_results()
    assert client._pending_release_results == []

    client.delete("rollout", list(remote.fields), list(remote.fields.values()))
    assert FakeTransfer.cleanup_calls == ["release-cleanup", "ref"]

    with pytest.raises(RuntimeError, match="cleanup failed"):
        client.delete(
            "rollout",
            ["bad", "other"],
            [{"ref": "bad", "kind": "batch"}, {"ref": "other", "kind": "batch"}],
        )
    assert FakeTransfer.cleanup_calls[-2:] == ["bad", "other"]

    cleanup_count = len(FakeTransfer.cleanup_calls)
    with pytest.raises(TypeError, match="Unsupported Mooncake field reference"):
        client.delete(
            "rollout",
            ["valid", "invalid"],
            [{"ref": "valid", "kind": "batch"}, object()],
        )
    assert len(FakeTransfer.cleanup_calls) == cleanup_count


def test_column_remote_batch_selected_view_shares_cleanup(monkeypatch):
    values = {
        "tokens": torch.tensor([[1], [2]]),
        "rewards": torch.tensor([[0.1], [0.2]]),
    }
    delete_calls = []
    monkeypatch.setattr(
        transfer_backend,
        "get",
        lambda **kwargs: TensorDict(
            {key: values[key] for key in kwargs["keys"]},
            batch_size=[2],
        ),
    )
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    fields = {key: {"ref": "ref", "kind": "batch"} for key in ("tokens", "rewards")}
    remote = _column_remote(fields, 2)

    empty_view = remote.select([])
    for empty in (empty_view.materialize([]), empty_view.materialize()):
        assert len(empty) == 2
        assert list(empty.keys()) == []
    remote.materialize(["tokens"])
    assert set(remote.cache.keys()) == {"tokens"}
    remote.materialize(["rewards"])
    assert set(remote.cache.keys()) == {"tokens", "rewards"}
    selected = remote.select(list(fields))
    selected.drop()

    assert len(delete_calls) == 1
    assert selected.cache is None
    with pytest.raises(RuntimeError, match="already been dropped"):
        remote.materialize(["tokens"])


def test_nested_column_remote_batch_materializes_and_drops_children(monkeypatch):
    fetched = {
        "first": TensorDict(
            {"tokens": torch.tensor([[1]]), "labels": torch.tensor([[10]])},
            batch_size=[1],
        ),
        "second": TensorDict(
            {"tokens": torch.tensor([[2]]), "labels": torch.tensor([[20]])},
            batch_size=[1],
        ),
        "combined": TensorDict({"rewards": torch.tensor([[0.1], [0.2]])}, batch_size=[2]),
    }
    delete_calls = []

    def get(**kwargs):
        return fetched[kwargs["fields"][0]["ref"]]

    monkeypatch.setattr(transfer_backend, "get", get)
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    children = [
        _column_remote(
            {
                "tokens": {"ref": ref, "kind": "batch"},
                "labels": {"ref": ref, "kind": "batch"},
            },
            1,
            row_ids=[str(index)],
        )
        for index, ref in enumerate(("first", "second"))
    ]
    nested = pickle.loads(pickle.dumps(ColumnRemoteBatch.cat(children)))
    remote = _column_remote(
        {"rewards": {"ref": "combined", "kind": "batch"}},
        2,
        row_ids=["0", "1"],
    ).union(nested)
    cloned = remote.clone()
    assert cloned.fields["tokens"] is cloned.fields["labels"]
    selected = cloned.slice(1, 2)
    token_data = selected.materialize(["tokens"])
    materialized = selected.materialize(["labels", "rewards"])
    assert selected.row_ids() == ["1"]
    assert torch.equal(token_data["tokens"], torch.tensor([[2]]))
    assert torch.equal(materialized["labels"], torch.tensor([[20]]))
    assert torch.equal(materialized["rewards"], torch.tensor([[0.2]]))
    assert set(selected.cache.keys()) == {"tokens", "labels", "rewards"}
    cloned.drop()

    assert len(delete_calls) == 3
    assert cloned.cache is None
    with pytest.raises(RuntimeError, match="already been dropped"):
        remote.materialize(["tokens"])


def test_dp2_nested_union_preserves_order_through_select_repeat(monkeypatch):
    def get(**kwargs):
        data = {key: field["data"] for key, field in zip(kwargs["keys"], kwargs["fields"])}
        return TensorDict(data, batch_size=[len(next(iter(data.values())))])

    def make_data(field, values):
        return DataProto(
            batch=None,
            remote_batch=_column_remote(
                {field: {"data": torch.tensor(values).view(4, 1)}},
                4,
                row_ids=["r0", "r1", "r2", "r3"],
            ),
        )

    monkeypatch.setattr(transfer_backend, "get", get)
    inputs = DataProto.concat(make_data("tokens", [10, 11, 12, 13]).chunk(2))
    outputs = DataProto.concat(make_data("rewards", [20, 21, 22, 23]).chunk(2))

    result = inputs.union(outputs).select_idxs([3, 0, 2]).repeat(2, interleave=False)
    materialized = result._remote_batch.materialize(["tokens", "rewards"])

    assert result._remote_batch.row_ids() == ["r3", "r0", "r2", "r3", "r0", "r2"]
    assert materialized["tokens"].squeeze(-1).tolist() == [13, 10, 12, 13, 10, 12]
    assert materialized["rewards"].squeeze(-1).tolist() == [23, 20, 22, 23, 20, 22]


def test_many_chunk_cat_merges_lifetimes_once(monkeypatch):
    chunks = [
        _column_remote(
            {"tokens": {"ref": f"ref-{index}", "kind": "batch"}},
            1,
            row_ids=[str(index)],
        )
        for index in range(64)
    ]
    domain_type = type(chunks[0]._lifetime._domain)
    original_merge_many = domain_type.merge_many.__func__
    merge_sizes = []

    def merge_many(cls, domains):
        domains = tuple(domains)
        merge_sizes.append(len(domains))
        return original_merge_many(cls, domains)

    monkeypatch.setattr(domain_type, "merge_many", classmethod(merge_many))
    combined = ColumnRemoteBatch.cat(chunks)

    assert merge_sizes == [65]
    assert len(combined._states) == 64
    assert combined.row_ids() == [str(index) for index in range(64)]

    state_reads = []
    original_states = domain_type.states

    def states(self, state_ids):
        state_ids = tuple(state_ids)
        state_reads.append(len(state_ids))
        return original_states(self, state_ids)

    rechunked = combined.chunk([1] * 64)
    state_id_iterations = []

    class CountingStateIds(tuple):
        def __iter__(self):
            state_id_iterations.append(len(self))
            return super().__iter__()

    combined._lifetime._state_ids = CountingStateIds(combined._lifetime._state_ids)
    monkeypatch.setattr(domain_type, "states", states)
    recombined = ColumnRemoteBatch.cat(rechunked)

    assert state_id_iterations == [64]
    assert state_reads == [64]
    assert len(recombined._states) == 64


def test_column_remote_batch_preserves_row_order_through_transforms():
    remote = _column_remote(
        {"tokens": {"ref": "ref", "kind": "batch"}},
        4,
        cache=TensorDict({"tokens": torch.arange(4).view(4, 1)}, batch_size=[4]),
        row_ids=["0", "1", "2", "3"],
    )

    restored = pickle.loads(pickle.dumps(remote))
    assert restored.row_ids() == ["0", "1", "2", "3"]
    assert restored.select_idxs([3, 1]).row_ids() == ["3", "1"]
    assert restored.slice(1, 4, 2).row_ids() == ["1", "3"]
    assert restored.repeat(2, interleave=True).row_ids() == ["0", "0", "1", "1", "2", "2", "3", "3"]
    assert restored.repeat(2, interleave=False).row_ids() == ["0", "1", "2", "3", "0", "1", "2", "3"]
    assert ColumnRemoteBatch.cat([remote.slice(0, 2), remote.slice(2, 4)]).row_ids() == ["0", "1", "2", "3"]

    assert len(remote.select_idxs([False, False, False, False])) == 0
    assert len(remote.select_idxs([])) == 0
    with pytest.raises(AssertionError, match="Boolean index length"):
        remote.select_idxs([True])

    view = remote.slice(1, 3)
    view.union(
        _column_remote(
            {"rewards": {"ref": "rewards", "kind": "batch"}},
            2,
            row_ids=["1", "2"],
        )
    )
    assert "rewards" in view
    assert "rewards" not in remote
    del view["tokens"]
    assert "tokens" in remote
    assert torch.equal(remote.materialize(["tokens"])["tokens"], remote.cache["tokens"])


def test_union_only_adopts_cleanup_for_new_fields(monkeypatch):
    delete_calls = []
    monkeypatch.setattr(
        transfer_backend,
        "get",
        lambda **kwargs: TensorDict({"tokens": torch.tensor([[1]])}, batch_size=[1]),
    )
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    left = _column_remote({"tokens": {"ref": "left", "kind": "batch"}}, 1)
    right = _column_remote(
        {"tokens": {"ref": "right", "kind": "batch"}},
        1,
        cache=TensorDict({"tokens": torch.tensor([[9]])}, batch_size=[1]),
    )

    left.union(right)
    assert left.cache is None
    assert torch.equal(left.materialize(["tokens"])["tokens"], torch.tensor([[1]]))
    left.drop()
    assert [call["fields"][0]["ref"] for call in delete_calls] == ["left"]

    right.drop()
    assert [call["fields"][0]["ref"] for call in delete_calls] == ["left", "right"]


def test_nested_union_only_adopts_cleanup_for_new_fields(monkeypatch):
    delete_calls = []
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))

    def child(index):
        row_ids = [str(index)]
        tokens = _column_remote(
            {"tokens": {"ref": f"tokens-{index}", "kind": "batch"}},
            1,
            row_ids=row_ids,
        )
        rewards = _column_remote(
            {"rewards": {"ref": f"rewards-{index}", "kind": "batch"}},
            1,
            row_ids=row_ids,
        )
        return tokens.union(rewards)

    rhs = pickle.loads(pickle.dumps(ColumnRemoteBatch.cat([child(0), child(1)])))
    left = _column_remote(
        {"rewards": {"ref": "left-rewards", "kind": "batch"}},
        2,
        row_ids=["0", "1"],
    )

    left.union(rhs.clone())
    state_count = len(left._states)
    left.union(rhs.clone())
    assert len(left._states) == state_count
    left.drop()
    assert [call["fields"][0]["ref"] for call in delete_calls] == [
        "left-rewards",
        "tokens-0",
        "tokens-1",
    ]

    rhs.drop()
    assert [call["fields"][0]["ref"] for call in delete_calls] == [
        "left-rewards",
        "tokens-0",
        "tokens-1",
        "rewards-0",
        "rewards-1",
    ]


def test_column_remote_batch_union_validates_identity_and_reuses_cache(monkeypatch):
    monkeypatch.setattr(
        transfer_backend,
        "get",
        lambda **kwargs: pytest.fail(f"unexpected GET for cached fields: {kwargs}"),
    )
    left = _column_remote(
        {"tokens": {"ref": "left", "kind": "batch"}},
        2,
        cache=TensorDict({"tokens": torch.tensor([[1], [2]])}, batch_size=[2]),
        row_ids=["0", "1"],
    )
    right = _column_remote(
        {"rewards": {"ref": "right", "kind": "batch"}},
        2,
        cache=TensorDict({"rewards": torch.tensor([[0.1], [0.2]])}, batch_size=[2]),
        row_ids=["0", "1"],
    )

    materialized = left.union(right).materialize(["tokens", "rewards"])
    assert torch.equal(materialized["tokens"], torch.tensor([[1], [2]]))
    assert torch.equal(materialized["rewards"], torch.tensor([[0.1], [0.2]]))

    cached = right.clone()
    copied = _column_remote({}, 2, row_ids=["0", "1"]).union(cached)
    assert copied.cache is not cached.cache
    assert copied.cache["rewards"].data_ptr() == cached.cache["rewards"].data_ptr()
    del copied.cache["rewards"]
    assert "rewards" in cached.cache

    wrong_partition = right.clone()
    wrong_partition.partition = "other"
    with pytest.raises(AssertionError, match="same partition"):
        left.clone().union(wrong_partition)

    missing_row_ids = right.clone()
    missing_row_ids._row_ids = None
    with pytest.raises(AssertionError, match="provide row ids"):
        left.clone().union(missing_row_ids)

    wrong_order = right.clone()
    wrong_order._row_ids = ["1", "0"]
    with pytest.raises(AssertionError, match="same order"):
        left.clone().union(wrong_order)

    without_ids = right.clone()
    without_ids._row_ids = None
    with pytest.raises(AssertionError, match="either provide row ids or omit them"):
        ColumnRemoteBatch.cat([without_ids, right])


def test_pickled_views_share_cleanup_state(monkeypatch):
    delete_calls = []
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    remote = _column_remote({"tokens": {"ref": "ref", "kind": "batch"}}, 1, row_ids=["0"])
    serialized = pickle.dumps(remote)
    left = pickle.loads(serialized)
    right = pickle.loads(serialized)
    assert len(left.union(right)._states) == 1

    remote, selected = pickle.loads(pickle.dumps((remote, remote.select(["tokens"]))))

    assert remote._states[0] is selected._states[0]
    remote.drop()
    selected.drop()
    assert len(delete_calls) == 1

    left = _column_remote({"tokens": {"ref": "left", "kind": "batch"}}, 1)
    left_alias = left.clone()
    left.union(_column_remote({"rewards": {"ref": "right", "kind": "batch"}}, 1))
    left_alias, left = pickle.loads(pickle.dumps((left_alias, left)))
    shared_state = {state.state_id: state for state in left._states}
    assert shared_state[left_alias._states[0].state_id] is left_alias._states[0]
    left.drop()
    left_alias.drop()
    assert len(delete_calls) == 3


def test_union_does_not_add_rhs_cleanup_to_existing_aliases(monkeypatch):
    delete_calls = []
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    left = _column_remote({"tokens": {"ref": "left", "kind": "batch"}}, 1)
    left_alias = left.clone()
    right = _column_remote({"rewards": {"ref": "right", "kind": "batch"}}, 1)

    left.union(right)
    left_alias.drop()

    assert len(left._states) == 2
    assert len(left_alias._states) == 1
    assert [call["fields"][0]["ref"] for call in delete_calls] == ["left"]


def test_union_adopts_an_indivisible_state_for_a_new_field(monkeypatch):
    delete_calls = []
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    left = _column_remote({"tokens": {"ref": "left", "kind": "batch"}}, 1)
    right = _column_remote(
        {
            "tokens": {"ref": "right", "kind": "batch"},
            "rewards": {"ref": "right", "kind": "batch"},
        },
        1,
    )

    left.union(right)
    left.drop()
    right.drop()

    assert [call["fields"][0]["ref"] for call in delete_calls] == ["left", "right"]


def test_to_remote_reuses_mooncake_row_ids(monkeypatch):
    calls = []

    def put(partition, row_ids, fields, batch_size):
        calls.append((partition, row_ids, list(fields), batch_size))
        return _column_remote(
            {key: {"ref": "output", "kind": "batch"} for key in fields},
            batch_size,
            row_ids=row_ids,
            partition=partition,
        )

    monkeypatch.setattr(transfer_backend, "put", put)
    input_remote = _column_remote(
        {"tokens": {"ref": "input", "kind": "batch"}},
        2,
        row_ids=["sample-0", "sample-1"],
    )
    input_data = DataProto(batch=None, remote_batch=input_remote)
    output = DataProto.to_remote(
        DataProto.from_dict(tensors={"rewards": torch.tensor([[1.0], [2.0]])}),
        ref_data=input_data,
    )

    assert calls == [("rollout", ["sample-0", "sample-1"], ["rewards"], 2)]
    assert output._remote_batch.row_ids() == ["sample-0", "sample-1"]
    assert set(output._remote_batch.fields) == {"rewards"}

    other_partition = input_remote.clone()
    other_partition.partition = "other"
    mismatched = DataProto.to_remote(
        DataProto.from_dict(
            tensors={"rewards": torch.tensor([[1.0], [2.0]])},
            meta_info={},
        ).union(DataProto(batch=None, remote_batch=other_partition)),
        ref_data=input_data,
    )
    assert mismatched._remote_batch.partition == "other"
    assert len(calls) == 1


def test_to_remote_cleans_new_bundle_when_union_fails(monkeypatch):
    delete_calls = []
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    old_remote = _column_remote({"tokens": {"ref": "old", "kind": "batch"}}, 1, row_ids=["0"])
    new_remote = _column_remote({"rewards": {"ref": "new", "kind": "batch"}}, 1, row_ids=["0"])

    def fail_union(self, rhs):
        raise RuntimeError("union failed")

    new_remote.union = types.MethodType(fail_union, new_remote)
    monkeypatch.setattr(transfer_backend, "put", lambda *args, **kwargs: new_remote)
    data = DataProto(
        batch=TensorDict({"rewards": torch.tensor([[1.0]])}, batch_size=[1]),
        remote_batch=old_remote,
    )

    with pytest.raises(RuntimeError, match="union failed"):
        DataProto.to_remote(data)

    assert [call["fields"][0]["ref"] for call in delete_calls] == ["new"]


def test_to_remote_rejects_dropped_input_before_put(monkeypatch):
    put_calls = []
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: None)
    monkeypatch.setattr(transfer_backend, "put", lambda *args, **kwargs: put_calls.append(args))
    old_remote = _column_remote({"tokens": {"ref": "old", "kind": "batch"}}, 1, row_ids=["0"])
    old_remote.drop()
    data = DataProto(
        batch=TensorDict({"rewards": torch.tensor([[1.0]])}, batch_size=[1]),
        remote_batch=old_remote,
    )

    with pytest.raises(RuntimeError, match="already been dropped"):
        DataProto.to_remote(data)

    assert put_calls == []


def test_cleanup_survives_field_removal_and_pickle(monkeypatch):
    delete_calls = []
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    remote = _column_remote({"tokens": {"ref": "ref", "kind": "batch"}}, 1, row_ids=["0"])
    del remote["tokens"]

    pickle.loads(pickle.dumps(remote)).drop()

    assert len(delete_calls) == 1
    assert delete_calls[0]["fields"] == [{"ref": "ref", "kind": "batch"}]


def test_drop_is_idempotent_and_retries_failed_cleanup(monkeypatch):
    delete_calls = []
    delete_fails = [True]

    def delete(**kwargs):
        delete_calls.append(kwargs)
        if delete_fails[0]:
            raise RuntimeError("delete failed")

    monkeypatch.setattr(transfer_backend, "delete", delete)
    remote = _column_remote({"tokens": {"ref": "ref", "kind": "batch"}}, 1)

    with pytest.raises(RuntimeError, match="delete failed"):
        remote.drop()
    assert len(delete_calls) == 1
    delete_fails[0] = False
    remote.drop()
    remote.drop()
    assert len(delete_calls) == 2


def test_multistate_drop_retries_only_failed_cleanup(monkeypatch):
    delete_calls = []
    fail_tokens = [True]

    def delete(**kwargs):
        ref = kwargs["fields"][0]["ref"]
        delete_calls.append(ref)
        if ref == "tokens" and fail_tokens[0]:
            raise RuntimeError("delete failed")

    monkeypatch.setattr(transfer_backend, "delete", delete)
    remote = _column_remote({"tokens": {"ref": "tokens", "kind": "batch"}}, 1)
    remote.union(_column_remote({"rewards": {"ref": "rewards", "kind": "batch"}}, 1))

    with pytest.raises(RuntimeError, match="delete failed"):
        remote.drop()
    fail_tokens[0] = False
    remote.drop()
    remote.drop()

    assert delete_calls == ["tokens", "rewards", "tokens"]


def test_concurrent_alias_drop_cleans_up_once(monkeypatch):
    delete_calls = []
    delete_started = threading.Event()
    finish_delete = threading.Event()

    def delete(**kwargs):
        delete_calls.append(kwargs)
        delete_started.set()
        assert finish_delete.wait(timeout=2)

    monkeypatch.setattr(transfer_backend, "delete", delete)
    remote = _column_remote({"tokens": {"ref": "ref", "kind": "batch"}}, 1)
    alias = remote.clone()
    first = threading.Thread(target=remote.drop)
    second = threading.Thread(target=alias.drop)

    first.start()
    assert delete_started.wait(timeout=2)
    second.start()
    finish_delete.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(delete_calls) == 1


def test_pickle_waits_for_inflight_drop(monkeypatch):
    delete_calls = []
    delete_started = threading.Event()
    finish_delete = threading.Event()
    pickle_started = threading.Event()
    serialized = []

    def delete(**kwargs):
        delete_calls.append(kwargs)
        delete_started.set()
        assert finish_delete.wait(timeout=2)

    def serialize(remote):
        pickle_started.set()
        serialized.append(pickle.dumps(remote))

    monkeypatch.setattr(transfer_backend, "delete", delete)
    remote = _column_remote({"tokens": {"ref": "ref", "kind": "batch"}}, 1)
    drop_thread = threading.Thread(target=remote.drop)
    pickle_thread = threading.Thread(target=serialize, args=(remote,))

    drop_thread.start()
    assert delete_started.wait(timeout=2)
    pickle_thread.start()
    assert pickle_started.wait(timeout=2)
    finish_delete.set()
    drop_thread.join(timeout=2)
    pickle_thread.join(timeout=2)

    assert not drop_thread.is_alive()
    assert not pickle_thread.is_alive()
    pickle.loads(serialized[0]).drop()
    assert len(delete_calls) == 1


def test_overlapping_lifetimes_pickle_one_atomic_snapshot(monkeypatch):
    delete_calls = []
    first_state_snapshotted = threading.Event()
    continue_pickle = threading.Event()
    remote = _column_remote({"tokens": {"ref": "tokens", "kind": "batch"}}, 1)
    alias = remote.clone()
    remote.union(_column_remote({"rewards": {"ref": "rewards", "kind": "batch"}}, 1))
    state_type = type(remote._states[0])
    first_state_id = remote._states[0].state_id
    original_snapshot = state_type._snapshot

    def snapshot(state):
        result = original_snapshot(state)
        if state.state_id == first_state_id:
            first_state_snapshotted.set()
            assert continue_pickle.wait(timeout=2)
        return result

    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    monkeypatch.setattr(state_type, "_snapshot", snapshot)
    serialized = []
    pickle_thread = threading.Thread(target=lambda: serialized.append(pickle.dumps((alias, remote))))
    drop_thread = threading.Thread(target=remote.drop)

    pickle_thread.start()
    assert first_state_snapshotted.wait(timeout=2)
    drop_thread.start()
    assert delete_calls == []
    continue_pickle.set()
    pickle_thread.join(timeout=2)
    drop_thread.join(timeout=2)

    assert not pickle_thread.is_alive()
    assert not drop_thread.is_alive()
    restored_alias, restored = pickle.loads(serialized[0])
    assert all(state.active and state.delete_pending for state in restored._states)
    assert restored_alias._states[0] is restored._states[0]
    restored.drop()
    restored_alias.drop()
    assert [call["keys"] for call in delete_calls] == [
        ["tokens"],
        ["rewards"],
        ["tokens"],
        ["rewards"],
    ]


def test_union_after_rhs_alias_snapshot_preserves_cleanup_identity(monkeypatch):
    delete_calls = []
    left = _column_remote({"tokens": {"ref": "tokens", "kind": "batch"}}, 1)
    right = _column_remote({"rewards": {"ref": "rewards", "kind": "batch"}}, 1)
    right_alias = right.clone()
    domain_type = type(right._lifetime._domain)
    original_getstate = domain_type.__getstate__
    snapshotted = threading.Event()
    continue_pickle = threading.Event()

    def getstate(domain):
        result = original_getstate(domain)
        snapshotted.set()
        assert continue_pickle.wait(timeout=2)
        return result

    monkeypatch.setattr(domain_type, "__getstate__", getstate)
    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    serialized = []
    pickle_thread = threading.Thread(target=lambda: serialized.append(pickle.dumps((right_alias, left))))

    pickle_thread.start()
    assert snapshotted.wait(timeout=2)
    left.union(right)
    continue_pickle.set()
    pickle_thread.join(timeout=2)

    assert not pickle_thread.is_alive()
    restored_alias, restored = pickle.loads(serialized[0])
    rewards_state = next(state for state in restored._states if state.fields[0]["ref"] == "rewards")
    assert restored_alias._states[0] is rewards_state
    assert restored_alias._lifetime._domain.root() is restored._lifetime._domain.root()

    snapshotted = threading.Event()
    continue_pickle = threading.Event()
    serialized_again = []
    pickle_thread = threading.Thread(target=lambda: serialized_again.append(pickle.dumps((restored_alias, restored))))

    pickle_thread.start()
    assert snapshotted.wait(timeout=2)
    restored.drop()
    continue_pickle.set()
    pickle_thread.join(timeout=2)

    assert not pickle_thread.is_alive()
    second_alias, second_owner = pickle.loads(serialized_again[0])
    assert second_alias._lifetime._domain.root() is second_owner._lifetime._domain.root()
    assert all(state.active and state.delete_pending for state in second_owner._states)
    second_owner.drop()
    second_alias.drop()
    assert [call["fields"][0]["ref"] for call in delete_calls] == [
        "tokens",
        "rewards",
        "tokens",
        "rewards",
    ]


def test_pickle_waits_for_atomic_union_metadata(monkeypatch):
    delete_calls = []
    union_paused = threading.Event()
    continue_union = threading.Event()
    pickle_started = threading.Event()
    pickle_finished = threading.Event()
    remote = _column_remote({"tokens": {"ref": "tokens", "kind": "batch"}}, 1)
    rhs = _column_remote({"rewards": {"ref": "rewards", "kind": "batch"}}, 1)
    original_states_for_fields = rhs._states_for_fields

    def states_for_fields(self, fields):
        union_paused.set()
        assert continue_union.wait(timeout=2)
        return original_states_for_fields(fields)

    def serialize():
        pickle_started.set()
        serialized.append(pickle.dumps(remote))
        pickle_finished.set()

    monkeypatch.setattr(transfer_backend, "delete", lambda **kwargs: delete_calls.append(kwargs))
    rhs._states_for_fields = types.MethodType(states_for_fields, rhs)
    serialized = []
    union_thread = threading.Thread(target=remote.union, args=(rhs,))
    pickle_thread = threading.Thread(target=serialize)

    union_thread.start()
    assert union_paused.wait(timeout=2)
    pickle_thread.start()
    assert pickle_started.wait(timeout=2)
    assert not pickle_finished.wait(timeout=0.1)
    continue_union.set()
    union_thread.join(timeout=2)
    pickle_thread.join(timeout=2)

    assert not union_thread.is_alive()
    assert not pickle_thread.is_alive()
    restored = pickle.loads(serialized[0])
    assert set(restored.fields) == {"tokens", "rewards"}
    restored.drop()
    assert [call["fields"][0]["ref"] for call in delete_calls] == ["tokens", "rewards"]


def test_mooncake_node_actor_serializes_before_release(monkeypatch):
    actor_cls = MooncakeNodeTransferActor.__ray_metadata__.modified_class
    actor = actor_cls.__new__(actor_cls)
    data = TensorDict({"tokens": torch.tensor([[1]])}, batch_size=[1])
    calls = []
    failures = {"put": False, "release": False}

    class Client:
        def get(self, partition, keys, fields):
            calls.append("get")
            return data

        def put(self, partition, row_ids, fields, batch_size):
            calls.append("put")
            return "remote"

        def release(self, value):
            calls.append(("release", value))
            if failures["release"]:
                raise RuntimeError("release failed")

    def ray_put(value):
        calls.append(("ray.put", value))
        if failures["put"]:
            raise RuntimeError("serialization failed")
        return "object-ref"

    actor.client = Client()
    monkeypatch.setattr(transfer_backend.ray, "put", ray_put)

    assert actor.get("rollout", ["tokens"], ["ref"]) == "object-ref"
    assert [call if isinstance(call, str) else call[0] for call in calls] == ["get", "ray.put", "release"]

    calls.clear()
    failures["put"] = True
    with pytest.raises(RuntimeError, match="serialization failed"):
        actor.get("rollout", ["tokens"], ["ref"])
    assert [call if isinstance(call, str) else call[0] for call in calls] == ["get", "ray.put", "release"]

    calls.clear()
    failures.update(put=False, release=True)
    with pytest.raises(RuntimeError, match="release failed"):
        actor.get("rollout", ["tokens"], ["ref"])
    assert [call if isinstance(call, str) else call[0] for call in calls] == ["get", "ray.put", "release"]

    calls.clear()
    failures["put"] = True
    with pytest.raises(RuntimeError, match="serialization failed"):
        actor.get("rollout", ["tokens"], ["ref"])
    assert [call if isinstance(call, str) else call[0] for call in calls] == ["get", "ray.put", "release"]


def test_mooncake_client_real_rdma_local_round_trip(mooncake_master):
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "")
    if protocol != "rdma":
        pytest.skip("Set MOONCAKE_PROTOCOL=rdma to run the Mooncake RDMA backend test")

    local_hostname = os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "")
    rdma_devices = os.environ.get("MOONCAKE_DEVICE") or os.environ.get("MOONCAKE_DEVICE_NAME", "")
    if not local_hostname or not rdma_devices:
        pytest.skip("Set MOONCAKE_LOCAL_HOSTNAME and MOONCAKE_DEVICE for RDMA testing")

    local_buffer_size = min(
        int(os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 128 * 1024 * 1024)),
        128 * 1024 * 1024,
    )
    config = {
        "local_hostname": local_hostname,
        "metadata_server": os.environ.get("MOONCAKE_METADATA_SERVER", "P2PHANDSHAKE"),
        "global_segment_size": int(os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1024 * 1024 * 1024)),
        "local_buffer_size": local_buffer_size,
        "protocol": protocol,
        "rdma_devices": rdma_devices,
        "master_server_addr": mooncake_master,
        "transfer_policy": {"copy_mode": "auto"},
        "field_schemas": {
            "token_rows": {
                "codec": "typed_ragged",
                "nullable": False,
                "metadata": {"section": "non_tensor_batch", "dtype": "int64"},
            },
            "nullable_rows": {
                "codec": "typed_ragged",
                "nullable": True,
                "metadata": {"section": "non_tensor_batch", "dtype": "int64"},
            },
        },
    }
    started_ray = False
    if not ray.is_initialized():
        ray.init(include_dashboard=False, num_cpus=2)
        started_ray = True
    config["node_actor_session_id"] = uuid.uuid4().hex
    client = MooncakeNodeClientProxy(config)

    row_length = 1024 * 1024
    rows = np.empty(2, dtype=object)
    rows[0] = np.arange(row_length, dtype=np.int64)
    rows[1] = np.arange(row_length, dtype=np.int64) + row_length
    nullable_rows = np.array([None, None], dtype=object)
    fields = {
        "tokens": torch.tensor([[1, 2], [3, 4]]),
        "prompt": np.array(["a", "b"], dtype=object),
        "token_rows": rows,
        "nullable_rows": nullable_rows,
    }

    original_client = transfer_backend._client
    held_results = []
    try:
        transfer_backend._client = client
        for _ in range(10):
            remote = client.put("rollout", ["0", "1"], fields, batch_size=2)
            remote.cache = None
            materialized = remote.materialize(list(fields))
            assert torch.equal(materialized["tokens"], fields["tokens"])
            assert list(materialized["prompt"]) == ["a", "b"]
            assert materialized["token_rows"][0].shape == (row_length,)
            assert materialized["token_rows"][1][-1] == 2 * row_length - 1
            assert list(materialized["nullable_rows"]) == [None, None]
            held_results.append(materialized)
            remote.drop()
    finally:
        transfer_backend._client = original_client
        ray.kill(client.actor)
        if started_ray:
            ray.shutdown()
