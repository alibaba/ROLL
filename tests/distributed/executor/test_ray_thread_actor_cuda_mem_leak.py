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
import gc
import json

import ray
import torch

from roll.distributed.scheduler.initialize import init


def log_gpu_memory_usage(head: str):
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    message = f"{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}"
    print(message)
    return memory_allocated, memory_reserved


@ray.remote(num_gpus=1)
class ComputeTensorActor:
    def __init__(self, name):
        self.name = name

    def compute_tensor(self, num=100):
        torch.manual_seed(0)

        tensor_size = (1024, 1024)
        tensor = torch.randn(tensor_size, device="cuda")
        tensor_list = [torch.randn(tensor_size, device="cuda") for _ in range(num * 100)]

        for _ in range(num):
            other_tensor = torch.randn(tensor_size, device="cuda")
            tensor = torch.mm(tensor, other_tensor)
            tensor = torch.relu(tensor)
            tensor += 0.1 * torch.randn(tensor_size, device="cuda")

        metrics = {}
        memory_allocated, memory_reserved = log_gpu_memory_usage(head=f"{self.name} before empty cache")
        metrics["onload/memory_allocated"] = memory_allocated
        metrics["onload/memory_reserved"] = memory_reserved

        del tensor_list, tensor

        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()

        memory_allocated, memory_reserved = log_gpu_memory_usage(head=f"{self.name} after empty cache")
        metrics["offload/memory_allocated"] = memory_allocated
        metrics["offload/memory_reserved"] = memory_reserved

        return metrics


def test_thread_actor():
    ray.init(num_gpus=1, ignore_reinit_error=True)
    cp_actor = ComputeTensorActor.options(num_gpus=1, max_concurrency=1000).remote("thread actor")
    num = 100
    metric_list = [ray.get(cp_actor.compute_tensor.remote(num=num)) for _ in range(num)]

    print(metric_list)

    with open("thread_actor_metrics.json", "w") as f:
        json.dump(metric_list, f)


def test_common_actor():
    ray.init(num_gpus=1, ignore_reinit_error=True)
    cp_actor = ComputeTensorActor.options(num_gpus=1).remote("thread actor")
    num = 100
    metric_list = [ray.get(cp_actor.compute_tensor.remote(num=num)) for _ in range(num)]

    print(metric_list[-1])
    with open("common_actor_metrics.json", "w") as f:
        json.dump(metric_list, f)


if __name__ == "__main__":
    test_thread_actor()
    test_common_actor()
