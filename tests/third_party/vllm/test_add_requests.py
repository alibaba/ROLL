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
import os
import time
from typing import Optional

import ray
import torch
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def test():
    model_path = "Qwen/Qwen2.5-7B-Instruct"

    prompts = [[1, 2, 3]]
    TOTAL = 3

    ray.init()
    resource_manager = ResourceManager()
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])
    sampling_params = SamplingParams(temperature=0.1, top_p=0.99, top_k=100, max_tokens=512, n=TOTAL)

    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT, stacks="python")

    model = LLM(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
        distributed_executor_backend="ray",
        disable_custom_all_reduce=True,
        enforce_eager=True,
        enable_sleep_mode=True,
    )

    model.add_requests(request_ids=[12345], sampling_params=sampling_params, prompt_token_ids=prompts)

    vllm_outputs = []
    count = 0
    while count < TOTAL:
        assert model.llm_engine.has_unfinished_requests()
        vllm_outputs = model.fetch_output()
        if len(vllm_outputs) > 0:
            assert len(vllm_outputs) == 1
            count += len(vllm_outputs[0].outputs)
    assert not model.llm_engine.has_unfinished_requests()


if __name__ == "__main__":
    test()
