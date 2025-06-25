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
import json
import random
import time
from collections import defaultdict
from threading import Thread, Event
from typing import Any, Dict, List, Optional, Union

import ray
from ray.util.queue import Queue
from tqdm import tqdm

from roll.agentic.rollout.es_manager import EnvironmentManager
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import append_to_dict, GenerateRequestType
from roll.utils.logging import get_logger

logger = get_logger()


class RolloutScheduler:
    def __init__(self, config, env_manager_config, resource_manager, infer_cluster, mode):
        self.config = config
        self.env_manager_config = env_manager_config
        self.resource_manager = resource_manager
        self.infer_cluster = infer_cluster
        self.mode = mode
        self.env_input_queue: Queue = Queue()
        self.env_output_queue: Queue = Queue()
        self.es_manager: Any = EnvironmentManager(
            name=self.env_manager_config.name,
            worker_cls=self.env_manager_config.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.env_manager_config,
        )
        self.es_manager.initialize(
            pipeline_config=self.config,
            infer_cluster=self.infer_cluster,
            input_queue=self.env_input_queue,
            output_queue=self.env_output_queue,
            mode=self.mode,
        )
        self.group_size = self.es_manager.worker_config.group_size
        self.rollout_buffers: Dict[str, List] = defaultdict(list)
        self.completed_rollouts: Dict[str, List] = defaultdict(list)
        self.thread_collect: Optional[Thread] = None
        self.thread_alive_check: Optional[Thread] = None
        self.progress_bar: Optional[tqdm] = None
        self.running = False
        self.alive_check_interval = self.config.alive_check_interval
        self.last_alive_check = time.time()
        self.event = Event()

    def get_batch(self, data: DataProto, batch_size):
        self.start_collect_rollouts()
        self.progress_bar = tqdm(
            total=batch_size, desc=f"{self.mode} rollout progress(trajectory)", mininterval=int(batch_size * 0.1) + 1
        )
        if self.mode == "train":
            data.meta_info["seed"] = random.randint(0, 1000000)
        else:
            data.meta_info["seed"] = self.config.seed
        self.infer_cluster.start_server(data=DataProto(meta_info=data.meta_info))
        self.reset_status()
        self.running = True

        rollout_refs: List[ray.ObjectRef] = self.es_manager.run_rollout_loop(data=data, blocking=False)

        while True:
            self.event.wait()
            self.check_worker_alive(self.infer_cluster)
            if len(self.completed_rollouts) * self.group_size >= batch_size:
                self.running = False
                self.stop_es_manager()
                break

            # TODO: dynamic filter

        ray.get(rollout_refs)
        gen_metrics = self.infer_cluster.stop_server(data=DataProto())

        completed_buffers = {k: v for k, v in self.completed_rollouts.items() if len(v) > 0}
        collect_data = [item for sublist in list(completed_buffers.values())[:] for item in sublist]
        data_batch: List[DataProto] = collect_data[:batch_size]
        metrics = {}
        [append_to_dict(metrics, meta_info.meta_info["metrics"]) for meta_info in data_batch]
        batch = DataProto.concat(data_batch)
        metrics.update(gen_metrics.meta_info.pop("metrics", {}))
        batch.meta_info["metrics"] = metrics
        self.reset_status()

        return batch

    def start_collect_rollouts(self):
        def collect_rollout():
            while True:
                rollout: Union[DataProto, str] = self.env_output_queue.get()

                if rollout == "stop" or not self.running:
                    self.clear_queue(queue=self.env_output_queue)
                    break

                group_id = rollout.non_tensor_batch["traj_group_id"][0]
                self.rollout_buffers[group_id].append(rollout)
                if len(self.rollout_buffers[group_id]) >= self.group_size:
                    self.completed_rollouts[group_id].extend(self.rollout_buffers[group_id])
                    self.progress_bar.update(self.group_size)
                    self.event.set()

        def alive_check():
            while self.running:
                time.sleep(self.alive_check_interval)
                self.event.set()

        self.clear_queue(queue=self.env_output_queue)
        self.thread_collect = Thread(target=collect_rollout)
        self.thread_collect.start()
        self.thread_alive_check = Thread(target=alive_check)
        self.thread_alive_check.start()

    def stop_es_manager(self):
        for _ in range(self.es_manager.world_size):
            self.env_input_queue.put("stop")
        self.env_output_queue.put("stop")
        self.thread_collect.join()

    def reset_status(self):
        self.rollout_buffers = defaultdict(list)
        self.completed_rollouts = defaultdict(list)

    def check_worker_alive(self, cluster):
        current_time = time.time()
        if current_time - self.last_alive_check >= self.alive_check_interval:
            outputs: List[DataProto] = cluster.add_request(command=GenerateRequestType.ALIVE_CHECK, data=DataProto())
            request_counts = {key: output.meta_info["request_counts"] for key, output in enumerate(outputs)}
            metrics = {"time": time.time(), "metrics": request_counts}
            logger.debug(f"metrics: {json.dumps(metrics)}")
            self.last_alive_check = current_time

    def clear_queue(self, queue):
        while not queue.empty():
            queue.get_nowait()
