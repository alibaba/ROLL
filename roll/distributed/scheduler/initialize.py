import os
import socket
import subprocess
import sys
import time
import atexit

import ray

from roll.distributed.scheduler.driver_utils import (
    get_driver_rank,
    get_driver_master_addr,
    get_driver_node_name,
    get_driver_master_port,
    get_driver_world_size,
    get_driver_dashboard_port,
    get_ray_status,
    is_ray_cluster_running,
    wait_for_nodes,
)
from roll.distributed.scheduler.log_monitor import LogMonitorListener
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.platforms import current_platform

logger = get_logger()
log_monitor_listener = None

def wait_for_head_node_ready(master_addr: str, master_port: str, timeout: int = 600, check_interval: int = 2):
    """Wait for Ray head node GCS to become available.

    Args:
        master_addr: Head node address
        master_port: Head node GCS port
        timeout: Maximum time to wait in seconds (default: 10 minutes)
        check_interval: Interval between connection attempts in seconds (default: 2s)

    Raises:
        RuntimeError: If head node doesn't become available within timeout
    """
    start_time = time.time()
    elapsed = 0

    logger.info(f"Waiting for Ray head node at {master_addr}:{master_port} to become available...")

    while elapsed < timeout:
        try:
            # Try to connect to the GCS port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(check_interval)
            result = sock.connect_ex((master_addr, int(master_port)))
            sock.close()

            if result == 0:
                logger.info(f"Ray head node at {master_addr}:{master_port} is ready (took {elapsed:.1f}s)")
                return
        except (socket.timeout, socket.error, OSError) as e:
            pass

        elapsed = time.time() - start_time
        if elapsed < timeout:
            time.sleep(check_interval)

    raise RuntimeError(
        f"Ray head node at {master_addr}:{master_port} did not become available within {timeout}s. "
        f"Please check if the head node is starting correctly."
    )


def start_ray_cluster():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()
    node_name = get_driver_node_name()
    dashboard_port = get_driver_dashboard_port()

    if is_ray_cluster_running():
        logger.info("Ray cluster already initialized")
        return False

    if rank == 0:
        cmd = f"ray start --head --port={master_port} --node-name={node_name} --dashboard-port={dashboard_port}"
    else:
        # Wait for head node to be ready before starting worker
        wait_for_head_node_ready(master_addr, master_port)
        cmd = f"ray start --address={master_addr}:{master_port} --node-name={node_name} --dashboard-port={dashboard_port}"

    logger.info(f"Starting ray cluster: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Failed to start ray cluster: {cmd}")
        logger.error(f"ret.stdout: {ret.stdout}")
        logger.error(f"ret.stderr: {ret.stderr}")
        sys.exit(1)
    return True


def stop_handler():
    global log_monitor_listener
    if log_monitor_listener is not None:
        log_monitor_listener.stop()


def init():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()

    manual_start = start_ray_cluster()

    runtime_env = {
        "env_vars": current_platform.get_custom_env_vars(),
    }

    if not ray.is_initialized():
        ray.init(
            address=f"{master_addr}:{master_port}" if manual_start else None,
            namespace=RAY_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=not manual_start,
            runtime_env=runtime_env,
        )
        logger.info("Ray cluster initialized")

    if manual_start:
        wait_for_nodes(expected=world_size)
        atexit.register(stop_handler)
        global log_monitor_listener
        log_monitor_listener = LogMonitorListener()
        log_monitor_listener.start()

    logger.info(f"Current ray cluster resources: {ray.available_resources()}")

    if manual_start and rank > 0:
        sys.exit(0)
