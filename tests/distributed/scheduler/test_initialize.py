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
import ray

from roll.distributed.scheduler.initialize import init


@ray.remote
class MyActor:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        msg = f"Hello from {self.name}! current node: {ray.get_runtime_context().get_node_id()}"
        print(msg)
        return msg


def test_ray_cluster_func():
    init()
    placement_group = ray.util.placement_group(bundles=[{"CPU": 1}, {"CPU": 1}], strategy="STRICT_SPREAD")
    ray.get(placement_group.ready())

    actor1 = MyActor.options(placement_group=placement_group, placement_group_bundle_index=0, num_cpus=1).remote(
        "Actor on Node 1"
    )
    actor2 = MyActor.options(placement_group=placement_group, placement_group_bundle_index=1, num_cpus=1).remote(
        "Actor on Node 2"
    )

    hello_msg1 = ray.get(actor1.say_hello.remote())
    hello_msg2 = ray.get(actor2.say_hello.remote())

    print(hello_msg1)
    print(hello_msg2)


if __name__ == "__main__":
    """
    RANK=0 WORLD_SIZE=2 MASTER_ADDR='33.197.137.224' MASTER_PORT=54893 python tests/distributed/scheduler/test_initialize.py
    RANK=1 WORLD_SIZE=2 MASTER_ADDR='33.197.137.224' MASTER_PORT=54893 python tests/distributed/scheduler/test_initialize.py
    """
    test_ray_cluster_func()
