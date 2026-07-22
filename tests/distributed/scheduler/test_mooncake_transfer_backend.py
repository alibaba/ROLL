import os
import shutil
import socket
import subprocess
import sys
import time
import types

import numpy as np
import pytest
import torch

from roll.configs.base_config import TransferBackendArguments
from roll.distributed.scheduler.transfer_backend import (
    MOONCAKE_CLIENT_SCOPE_NODE,
    MOONCAKE_CLIENT_SCOPE_PROCESS,
    MooncakeClient,
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
    if shutil.which("mooncake_master") is None:
        pytest.skip("mooncake_master is not available in PATH")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, port)) == 0:
            yield f"{host}:{port}"
            return

    process = subprocess.Popen(
        [
            "mooncake_master",
            f"--rpc_address={host}",
            f"--rpc_port={port}",
            "--logtostderr=true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_port(host, port)
        yield f"{host}:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


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


def test_mooncake_client_uses_unified_dataproto_api(monkeypatch):
    calls = []

    class FakeStore:
        def setup(self, *args, **kwargs):
            return 0

    class FakePolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTransfer:
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

    mooncake_module = types.ModuleType("mooncake")
    store_module = types.ModuleType("mooncake.store")
    structured_module = types.ModuleType("mooncake.structured_object_store")
    store_module.MooncakeDistributedStore = FakeStore
    structured_module.BundleTransferPolicy = FakePolicy
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
        }
    )

    remote = client.put(
        "rollout",
        ["0", "1"],
        {
            "tokens": torch.tensor([[1], [2]]),
            "prompt": np.array(["a", "b"], dtype=object),
        },
        batch_size=2,
    )
    materialized = client.get("rollout", ["tokens", "prompt"], [remote.fields["tokens"], remote.fields["prompt"]])

    put_name, put_kwargs, put_data = calls[0]
    get_name, get_kwargs, get_ref = calls[1]
    assert put_name == "put"
    assert put_kwargs["type"] == "dataproto"
    assert put_kwargs["partition"] == "rollout"
    assert put_data.meta_info["roll_row_ids"] == ["0", "1"]
    assert get_name == "get"
    assert get_ref == "ref"
    assert get_kwargs == {
        "type": "dataproto",
        "batch_fields": ["tokens"],
        "non_tensor_fields": ["prompt"],
        "data_cls": dict,
    }
    assert torch.equal(materialized["tokens"], torch.tensor([[1], [2]]))
    assert list(materialized["prompt"]) == ["a", "b"]


def test_mooncake_client_real_rdma_round_trip(mooncake_master):
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "")
    if protocol != "rdma":
        pytest.skip("Set MOONCAKE_PROTOCOL=rdma to run the Mooncake RDMA backend test")

    local_hostname = os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "")
    rdma_devices = os.environ.get("MOONCAKE_DEVICE_NAME", "")
    if not local_hostname or not rdma_devices:
        pytest.skip("Set MOONCAKE_LOCAL_HOSTNAME and MOONCAKE_DEVICE_NAME for RDMA testing")

    client = MooncakeClient(
        {
            "local_hostname": local_hostname,
            "metadata_server": os.environ.get("MOONCAKE_METADATA_SERVER", "P2PHANDSHAKE"),
            "global_segment_size": int(os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1024 * 1024 * 1024)),
            "local_buffer_size": int(os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1024 * 1024 * 1024)),
            "protocol": protocol,
            "rdma_devices": rdma_devices,
            "master_server_addr": mooncake_master,
            "transfer_policy": {"copy_mode": "auto"},
        }
    )
    fields = {
        "tokens": torch.tensor([[1, 2], [3, 4]]),
        "prompt": np.array(["a", "b"], dtype=object),
    }

    remote = client.put("rollout", ["0", "1"], fields, batch_size=2)
    materialized = client.get("rollout", ["tokens", "prompt"], [remote.fields["tokens"], remote.fields["prompt"]])

    assert torch.equal(materialized["tokens"], fields["tokens"])
    assert list(materialized["prompt"]) == ["a", "b"]

    client.delete("rollout", list(remote.fields.keys()), list(remote.fields.values()))
