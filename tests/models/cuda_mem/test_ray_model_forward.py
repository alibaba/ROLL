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
import time

import ray

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cuda_mem.utils import log_gpu_memory_usage


@ray.remote(num_gpus=1)
def ray_forward():
    print("ray_forward")
    device = torch.device("cuda:0")

    path = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.eval()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=True, split_special_tokens=True, padding_side="left", padding=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "Compared with Google, Microsoft",
        "据悉，美国总统",
    ]
    batch = tokenizer(prompts, return_tensors="pt", padding=True)

    model = model.to("cpu")
    torch.cuda.empty_cache()
    log_gpu_memory_usage(head="initialize offload")

    with torch.no_grad():
        # model prepare done
        model = model.to(device)
        input_ids = batch["input_ids"].to(device)
        outputs = model.forward(input_ids)

    del outputs, input_ids, model
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory_usage(head="forward offload")
    time.sleep(600)


if __name__ == "__main__":
    ray.init(log_to_driver=True)
    ray.get(ray_forward.remote())
