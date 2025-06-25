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
from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor

from roll.third_party.vllm.vllm_0_8_4.ray_distributed_executor import ( 
    CustomRayDistributedExecutor as CustomRayDistributedExecutorV0)

# Force RayDistributedExecutor to come before CustomRayDistributedExecutorV0
# to ensure correct method resolution order (MRO) and override behavior.
class CustomRayDistributedExecutor(RayDistributedExecutor, CustomRayDistributedExecutorV0):
    pass
