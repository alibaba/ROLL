"""Run against an isolated Ray cluster to cover outbound-network independence."""
import os

import pytest


@pytest.fixture(scope="module")
def ray_cluster():
    if os.environ.get("RUN_QWEN38_RAY_NETWORK_TESTS") != "1":
        pytest.skip("requires an isolated Ray test cluster")
    import ray
    ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False)
    yield ray
    ray.shutdown()


def node_address_without_external_socket():
    from unittest.mock import patch
    import ray
    from roll.utils.network_utils import get_node_ip

    node_id = ray.get_runtime_context().get_node_id()
    expected = next(node["NodeManagerAddress"] for node in ray.nodes() if node["NodeID"] == node_id)
    # Ray's connected node metadata supplies the IP. No external UDP socket
    # may be constructed, even on a host which happens to have internet access.
    with patch("roll.utils.network_utils.socket.socket", side_effect=AssertionError("external socket")):
        observed = get_node_ip()
    assert observed == expected
    return observed


def test_driver_uses_connected_ray_node_address(ray_cluster):
    node_address_without_external_socket()


def test_worker_uses_connected_ray_node_address(ray_cluster):
    remote = ray_cluster.remote(node_address_without_external_socket)
    result = ray_cluster.get(remote.remote())
    assert result in {node["NodeManagerAddress"] for node in ray_cluster.nodes()}
