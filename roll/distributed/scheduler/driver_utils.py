import os
import subprocess
import time
import asyncio

import ray
from ray import WORKER_MODE

from roll.utils.logging import get_logger

logger = get_logger()


def is_driver():
    return ray.get_runtime_context().worker.mode != WORKER_MODE if ray.is_initialized() else True


def get_driver_rank():
    assert is_driver(), "this function should only be run on a driver"
    return int(os.getenv("RANK", "0"))


def get_driver_world_size():
    assert is_driver(), "this function should only be run on a driver"
    return int(os.getenv("WORLD_SIZE", "1"))


def get_driver_master_addr():
    assert is_driver(), "this function should only be run on a driver"
    return os.getenv("MASTER_ADDR", "127.0.0.1")


def get_driver_master_port():
    assert is_driver(), "this function should only be run on a driver"
    return os.getenv("MASTER_PORT", "6379")


def get_driver_node_name():
    assert is_driver(), "this function should only be run on a driver"
    return os.getenv("WORKER_ID", f"{get_driver_master_addr()}:{get_driver_rank()}")


def get_driver_dashboard_port():
    assert is_driver(), "this function should only be run on a driver"
    return os.getenv("DASHBOARD_PORT", "8265")


def is_multi_tenant():
    return os.getenv("MULTI_TENANT", "0") == "1"

def execute(cmd, check=False, retry=1):
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
    state = ret.returncode == 0
    msg = ret.stdout if state else ret.stderr
    if not state:
        logger.warning(f"execute {cmd} got error {msg}")
        if retry > 1:
            logger.warning(f"retry {cmd} ...")
            time.sleep(1)
            return execute(cmd, check, retry - 1)
    return state, msg


def is_connection_refused(msg):
    keywords = ["StatusCode.UNAVAILABLE", "Connection refused", "failed to connect to all addresses"]
    return any(keyword in msg for keyword in keywords)


def get_ray_status(retry=3):
    cluster_state, msg = execute("ray status", retry=retry)
    if cluster_state:
        return True, None
    elif is_connection_refused(msg):
        return False, msg
    if not cluster_state:
        return False, msg
    return True, msg


def filter_known_msg(msg):
    if "StatusCode.DEADLINE_EXCEEDED" in msg:
        return True
    return False


def is_ray_cluster_running():
    if is_multi_tenant():
        ret = subprocess.run(f"ray status --address {get_driver_master_addr()}:{get_driver_master_port()}", shell=True, capture_output=True)
    else:
        ret = subprocess.run(f"ray status", shell=True, capture_output=True)
    if ret.returncode != 0:
        return False
    return True


def wait_for_nodes(expected):
    # Wait for all nodes to join the cluster.
    while True:
        nodes_info = ray.nodes()
        active_nodes = [node for node in nodes_info if node["Alive"]]
        num_nodes = len(active_nodes)
        if num_nodes != expected:
            logger.info(f"{num_nodes} nodes have joined so far, waiting for {expected - num_nodes}.")
            time.sleep(1)
        else:
            break

@ray.remote(num_cpus=0)
class Barrier:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.arrived = 0
        self.event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            self.arrived += 1
            if self.arrived == self.num_workers:
                self.event.set()
                self.arrived = 0
                self.event.clear()
                return
        await self.event.wait()