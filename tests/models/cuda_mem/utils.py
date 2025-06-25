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
import torch


def log_gpu_memory_usage(head: str):
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**2
    memory_reserved_max = torch.cuda.max_memory_reserved() / 1024**3
    message = (
        f"{head}, memory allocated (GB): {memory_allocated}, memory reserved (MB): {memory_reserved}, "
        f"memory max reserved (GB): {memory_reserved_max}"
    )
    print(message)
