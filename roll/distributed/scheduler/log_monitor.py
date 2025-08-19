"""
reference: https://github.com/alibaba/ChatLearn/blob/main/chatlearn/utils/log_monitor.py
"""

import atexit
import glob
import inspect
import logging
import os
import subprocess
import threading
import time
from collections import defaultdict
from typing import Dict

import ray
from packaging.version import Version
from ray._private.log_monitor import (
    LogMonitor as RayLogMonitor,
    is_proc_alive,
    RAY_RUNTIME_ENV_LOG_TO_DRIVER_ENABLED,
    WORKER_LOG_PATTERN,
    RUNTIME_ENV_SETUP_PATTERN,
    LogFileInfo,
)
from ray._private.worker import print_to_stdstream, logger as monitor_logger, print_worker_logs

from roll.distributed.scheduler.driver_utils import get_driver_rank, wait_for_nodes, get_driver_world_size
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger

logger = get_logger()

EXCEPTION_MONITOR_ACTOR_NAME = "ExceptionMonitor"


class StdPublisher:

    file_handlers = {}

    @staticmethod
    def publish_logs(data: Dict):
        print_signature = inspect.signature(print_to_stdstream)
        if "ignore_prefix" in print_signature.parameters:
            print_to_stdstream(data, ignore_prefix=False)
        else:
            print_to_stdstream(data)

        StdPublisher.publish_to_logfile(data)

    @staticmethod
    def publish_to_logfile(data: Dict):
        pid = data["pid"]
        role_tag = None
        if data.get("actor_name"):
            role_tag = data["actor_name"]
        elif data.get("task_name"):
            role_tag = data["task_name"]
        if role_tag is None:
            return

        log_dir = "./output/logs"
        os.makedirs(log_dir, exist_ok=True)
        file_name = f"{role_tag}.log"
        sink = StdPublisher.file_handlers.get(file_name, None)

        if sink is None:
            try:
                print(f"log redirect to filename: {os.path.join(log_dir, file_name)}")
                sink = open(os.path.join(log_dir, file_name), "w")
                StdPublisher.file_handlers[file_name] = sink
            except IOError as e:
                print(f"Failed to open log file {file_name}: {e}")
                return
        try:
            print_worker_logs(data, sink)
            sink.flush()
        except Exception as e:
            pass
        finally:
            pass

    @classmethod
    def close_file_handlers(cls):
        for file_name, handler in cls.file_handlers.items():
            try:
                handler.close()
            except Exception as e:
                print(f"Error closing log file {file_name}: {e}")


class LogMonitor(RayLogMonitor):

    def update_log_filenames(self):
        """
        Update the list of log files to monitor.
        overwrite 控制哪些日志文件该被追踪
        """
        monitor_log_paths = []
        # output of user code is written here
        monitor_log_paths += glob.glob(f"{self.logs_dir}/worker*[.out|.err]") + glob.glob(
            f"{self.logs_dir}/java-worker*.log"
        )
        # segfaults and other serious errors are logged here
        monitor_log_paths += glob.glob(f"{self.logs_dir}/raylet*.err")
        # monitor logs are needed to report autoscaler events
        if not self.is_autoscaler_v2:
            # We publish monitor logs in autoscaler v1
            monitor_log_paths += glob.glob(f"{self.logs_dir}/monitor.log")
        else:
            # We publish autoscaler events directly in autoscaler v2
            monitor_log_paths += glob.glob(f"{self.logs_dir}/events/event_AUTOSCALER.log")

        # If gcs server restarts, there can be multiple log files.
        monitor_log_paths += glob.glob(f"{self.logs_dir}/gcs_server*.err")

        # runtime_env setup process is logged here
        if RAY_RUNTIME_ENV_LOG_TO_DRIVER_ENABLED:
            monitor_log_paths += glob.glob(f"{self.logs_dir}/runtime_env*.log")
        for file_path in monitor_log_paths:
            if os.path.isfile(file_path) and file_path not in self.log_filenames:
                worker_match = WORKER_LOG_PATTERN.match(file_path)
                if worker_match:
                    worker_pid = int(worker_match.group(2))
                else:
                    worker_pid = None
                job_id = None

                # Perform existence check first because most file will not be
                # including runtime_env. This saves some cpu cycle.
                if "runtime_env" in file_path:
                    runtime_env_job_match = RUNTIME_ENV_SETUP_PATTERN.match(file_path)
                    if runtime_env_job_match:
                        job_id = runtime_env_job_match.group(1)

                is_err_file = file_path.endswith("err")

                self.log_filenames.add(file_path)
                self.closed_file_infos.append(
                    LogFileInfo(
                        filename=file_path,
                        size_when_last_opened=0,
                        file_position=0,
                        file_handle=None,
                        is_err_file=is_err_file,
                        job_id=job_id,
                        worker_pid=worker_pid,
                    )
                )


@ray.remote
class ExceptionMonitor:
    """
    监控ray集群node上执行存在异常，及时透出异常日志
    """

    def __init__(self):
        self._node_and_err_msg = defaultdict(list)
        self.running = True
        self.stop_count = 0

    def add_error_node_and_msg(self, ip, msg):
        self._node_and_err_msg[ip].append(msg)

    def get_error_node_and_msg(self):
        return self._node_and_err_msg

    def get_error_msg(self, ip):
        return self._node_and_err_msg[ip]

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False
        self.stop_count += 1

    def get_stop_count(self):
        return self.stop_count


class LogMonitorListener:

    def __init__(self):
        log_dir = os.path.dirname(os.path.dirname(ray.nodes()[0]["ObjectStoreSocketName"]))
        self.log_dir = os.path.join(log_dir, "logs")
        self.node_ip_address = ray.util.get_node_ip_address()
        self.rank = get_driver_rank()
        self.world_size = get_driver_world_size()
        if Version(ray.__version__) >= Version("2.47.0"):
            self.log_monitor = LogMonitor(
                node_ip_address=self.node_ip_address,
                logs_dir=self.log_dir,
                gcs_client=StdPublisher(),
                is_proc_alive_fn=is_proc_alive,
            )
        else:
            self.log_monitor = LogMonitor(
                node_ip_address=self.node_ip_address,
                logs_dir=self.log_dir,
                gcs_publisher=StdPublisher(),
                is_proc_alive_fn=is_proc_alive,
            )
        monitor_logger.setLevel(logging.CRITICAL)

        self.exception_monitor = None
        self.log_monitor_thread = threading.Thread(target=self.log_monitor.run)
        self.log_monitor_thread.daemon = True
        self.log_monitor_thread.start()

    def wait_for_grace_stop(self):
        if self.exception_monitor is None:
            return
        for i in range(50):
            if ray.get(self.exception_monitor.get_stop_count.remote()) >= self.world_size:
                return
            time.sleep(0.1)

    def stop(self):
        StdPublisher.close_file_handlers()
        time.sleep(5)
        self.log_monitor_thread.join(2)
        if self.exception_monitor is not None:
            try:
                ray.get(self.exception_monitor.stop.remote())
            except Exception as e:
                logger.info(f"{e}")
                logger.info("ExceptionMonitor has been killed when stopping")
        if self.rank == 0:
            self.wait_for_grace_stop()
        ray.shutdown()
        logger.info("Execute ray.shutdown before the program exits...")
        cmd = f"ray stop --force"
        subprocess.run(cmd, shell=True, capture_output=True)

    def start(self):
        atexit.register(self.stop)

        if self.rank == 0:
            self.exception_monitor = ExceptionMonitor.options(
                name=EXCEPTION_MONITOR_ACTOR_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
            ).remote()
        else:
            while True:
                if self.exception_monitor is None:
                    try:
                        self.exception_monitor = ExceptionMonitor.options(
                            name=EXCEPTION_MONITOR_ACTOR_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
                        ).remote()
                    except Exception as e:
                        self.exception_monitor = None
                else:
                    try:
                        if ray.get(self.exception_monitor.is_running.remote()):
                            error_msg_list = ray.get(
                                self.exception_monitor.get_error_msg.remote(ip=self.node_ip_address)
                            )
                            if error_msg_list:
                                msg = "\n".join(error_msg_list)
                                raise Exception(msg)
                        else:
                            self.exception_monitor = None
                            logger.info("ExceptionMonitor has been stopped")
                            break
                    except Exception as e:
                        logger.info(f"{e}")
                        logger.info("ExceptionMonitor has been killed")
                        break
                time.sleep(1)
            logger.info(f"driver_rank {self.rank} worker exit...")
