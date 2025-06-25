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
import itertools

import ray

from roll.distributed.executor.cluster import Cluster


class EnvironmentManager(Cluster):

    def initialize(self, pipeline_config, infer_cluster: Cluster, input_queue, output_queue, mode: str = "train"):
        infer_worker_iter = itertools.cycle(infer_cluster.workers)
        refs = []
        for worker in self.workers:
            infer_worker = next(infer_worker_iter)
            refs.append(
                worker.initialize.remote(
                    pipeline_config=pipeline_config,
                    infer_worker=infer_worker,
                    input_queue=input_queue,
                    output_queue=output_queue,
                    mode=mode,
                )
            )
        ray.get(refs)
