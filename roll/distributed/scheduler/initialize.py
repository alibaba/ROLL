import os
import subprocess
import sys
import time

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
        # fix: 处理大规模下可能会出现的head/worker node创建顺序不一致问题
        time.sleep(5)
        cmd = f"ray start --address={master_addr}:{master_port} --node-name={node_name} --dashboard-port={dashboard_port}"

    logger.info(f"Starting ray cluster: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Failed to start ray cluster: {cmd}")
        logger.error(f"ret.stdout: {ret.stdout}")
        logger.error(f"ret.stderr: {ret.stderr}")
        sys.exit(1)
    return True


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
        listener = LogMonitorListener()
        listener.start()

    logger.info(f"Current ray cluster resources: {ray.available_resources()}")

    if manual_start and rank > 0:
        sys.exit(0)
