# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import socket
from concurrent import futures
from dataclasses import dataclass
from typing import Dict

from roll.configs.worker_config import WorkerConfig
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import RAY_NAMESPACE, STORAGE_NAME
from roll.utils.context_managers import state_offload_manger
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType


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

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            master_addr = instance.get_node_ip()
            master_port = str(instance.get_free_port())
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

        return instance

    def __init__(self, worker_config: WorkerConfig):
        self.worker_config = worker_config
        self.pipeline_config = None
        self.worker_name = os.environ.get("WORKER_NAME", None)
        self.cluster_name = os.environ.get("CLUSTER_NAME", None)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = int(os.environ["MASTER_PORT"])

        # NOTE: 自定义Worker时根据需要配置rank_info
        self.rank_info = RankInfo(
            world_size=self.world_size,
            rank=self.rank,
            dp_rank=self.rank,
            dp_size=self.world_size,
        )

        self.shared_storage = SharedStorage.options(
            name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()
        self.shared_storage.put.remote(
            self.cluster_name, {"MASTER_ADDR": self.master_addr, "MASTER_PORT": self.master_port}
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
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

    @staticmethod
    def get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    @staticmethod
    def get_visible_gpus():
        return os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

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
