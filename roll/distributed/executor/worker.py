import logging
import os
import socket
from concurrent import futures
from dataclasses import dataclass
from typing import Dict, Optional, List

import ray

from roll.configs.worker_config import WorkerConfig
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import RAY_NAMESPACE, STORAGE_NAME
from roll.utils.context_managers import state_offload_manger
from roll.utils.logging import get_logger
from roll.utils.network_utils import collect_free_port, get_node_ip
from roll.utils.offload_states import OffloadStateType
from roll.utils.offload_nccl import monkey_patch_torch_dist
from roll.platforms import current_platform


@dataclass
class RankInfo:
    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1

    rank: int = 0
    tp_rank: int = 0
    dp_rank: int = 0
    pp_rank: int = 0
    cp_rank: int = 0

    @property
    def is_pipeline_last_stage(self):
        return self.pp_rank == (self.pp_size - 1)


class Worker:

    def __init__(self, worker_config: WorkerConfig):
        if worker_config.offload_nccl:
            monkey_patch_torch_dist()
        self.worker_config = worker_config
        self.pipeline_config = None
        self.worker_name = os.environ.get("WORKER_NAME", None)
        self.cluster_name = os.environ.get("CLUSTER_NAME", None)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.shared_storage = SharedStorage.options(
            name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()

        if self.rank == 0:
            master_addr = self.get_node_ip()
            master_port = str(self.get_free_port())
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = int(os.environ["MASTER_PORT"])
        self.shared_storage.put.remote(
            self.cluster_name, {"MASTER_ADDR": self.master_addr, "MASTER_PORT": self.master_port}
        )
        # NOTE: 自定义Worker时根据需要配置rank_info
        self.rank_info = RankInfo(
            world_size=self.world_size,
            rank=self.rank,
            dp_rank=self.rank,
            dp_size=self.world_size,
        )
        self.thread_executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        self._logger = None

    def __repr__(self):
        return f"{type(self).__name__}({self.worker_name})"

    @property
    def logger(self) -> logging.Logger:
        """
        在ray.Actor内要使用自定义的logger, 避免ray context造成的logger不一致
        """
        self._logger = get_logger()
        return self._logger

    @staticmethod
    def get_node_ip():
        return get_node_ip()

    @staticmethod
    def get_free_port():
        shared_storage = SharedStorage.options(
            name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()
        master_addr = Worker.get_node_ip()
        max_retry_count = int(os.environ.get("MAX_PORT_RETRY_COUNT", 1000))
        retry_count = 0
        master_port = collect_free_port()
        while retry_count < max_retry_count:
            master_addr_port_key = f"MASTER_ADDR_PORT:{master_addr}:{master_port}"
            if ray.get(shared_storage.get.remote(master_addr_port_key)) is None:
                ray.get(shared_storage.put.remote(master_addr_port_key, True))
                break
            master_port = collect_free_port()
            retry_count += 1
        if retry_count >= max_retry_count:
            raise RuntimeError(f"Can not allocate unique MASTER_PORT on {master_addr}.")
        return master_port

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    @staticmethod
    def get_visible_gpus():
        return current_platform.get_visible_gpus()

    def get_devices_info(self):
        devices_info = [
            dict(rank=rank, node_rank=pg["node_rank"], gpu_rank=pg["gpu_rank"])
            for rank, pg in enumerate(self.worker_config.resource_placement_groups)
        ]
        return devices_info

    def get_rank_info(self):
        return self.rank_info

    def initialize(self, pipeline_config, *args, **kwargs):
        self.pipeline_config = pipeline_config

        model_name = self.worker_config.model_args.model_name_or_path
        if model_name:
            self.worker_config.model_args.model_name_or_path = download_model(model_name)

        if self.pipeline_config.resume_from_checkpoint:
            self.logger.info(f"resume_from_checkpoint: {self.pipeline_config.resume_from_checkpoint}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_states(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.load_states()
        else:
            self.logger.warning("worker has not strategy")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_states(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.offload_states()
        else:
            self.logger.warning("worker has not strategy")

    def broadcast_parameter(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.broadcast_parameter(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def broadcast_bucket(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.broadcast_bucket(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def setup_collective_group(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.setup_collective_group(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def start_model_update(self, *args, **kwargs):
        metrics = {}
        if getattr(self, "strategy", None) is not None:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/model_update",
                load_kwargs={"include": [OffloadStateType.model_params]},
            ):
                exec_metrics: Dict = self.strategy.model_update(*args, **kwargs)
            metric_prefix = f"time/{self.cluster_name}/model_update"
            metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        else:
            self.logger.warning("worker has not strategy")

        output = DataProto(meta_info={"metrics": metrics})
        return output

    def update_parameter(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.update_parameter(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def update_parameter_in_bucket(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.update_parameter_in_bucket(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def add_lora(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.add_lora(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def download_models(self, model_name_or_paths: set[str]):
        futures.wait([self.thread_executor.submit(download_model, model_name_or_path)
                      for model_name_or_path in model_name_or_paths])

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def get_metrics(self, metric_names: Optional[List[str]] = None) -> DataProto:
        """
        Get performance metrics from the strategy layer.

        Args:
            metric_names: Optional list of specific metric names to filter

        Returns:
            Dictionary of metric names to aggregated values
        """
        if getattr(self, "strategy", None) is not None:
            metrics = self.strategy.get_metrics(metric_names=metric_names)
        else:
            metrics = {}
        return DataProto(meta_info={"metrics": metrics})
