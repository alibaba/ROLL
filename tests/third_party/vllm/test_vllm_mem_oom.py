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
import os
import time

import ray
from transformers import AutoTokenizer
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM
from roll.utils.context_managers import cpu_memory_info
from roll.utils.logging import get_logger

logger = get_logger()


model_path = "Qwen/Qwen2.5-7B-Instruct"

prompts = [
    "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
    "根据关键词描述生成女装/女士精品行业连衣裙品类的发在淘宝的小红书风格的推送配文，包括标题和内容。关键词：pe。要求:1. 推送标题要体现关键词和品类特点，语言通顺，有吸引力，约10个字；2. 推送内容要语言通顺，突出关键词和品类特点，对目标受众有吸引力，长度约30字。标题:",
    "100.25和90.75谁更大？",
]


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

chat_prompts = []
for prompt in prompts:
    chat_prompts.append(chat_format(prompt))

# os.environ["RAY_DEBUG"] = "legacy"

ray.init()
resource_manager = ResourceManager()
placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=list(range(1)))
sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=1024)

model = LLM(
    resource_placement_groups=placement_groups[0],
    model=model_path,
    block_size=16,
    dtype="bfloat16",
    gpu_memory_utilization=0.8,
    tensor_parallel_size=1,
    trust_remote_code=True,
    enforce_eager=True,
    load_format="dummy",
)


from memory_profiler import profile
import tracemalloc

# tracemalloc.start()

snapshot_1 = None
snapshot_last = None


# @profile
def generate_memory():
    global snapshot_1, snapshot_last
    for _ in range(20):
        model.load_states()
        model.generate(
            sampling_params=sampling_params,
            prompts=chat_prompts,
            use_tqdm=False,
        )
        model.offload_states()
        rss = cpu_memory_info().rss / 1024**2
        logger.info(f"rss: {rss}")
        # snapshot_last = tracemalloc.take_snapshot()
        # if snapshot_1 is None:
        #     snapshot_1 = snapshot_last


generate_memory()

# tracemalloc.stop()

# snapshot.dump(f"mem_dump.pickle")
ray.shutdown()

# https://www.datacamp.com/tutorial/memory-profiling-python
#
# stats_1 = snapshot_1.compare_to(snapshot_last, 'lineno')
#
# with open('memory_leak_analysis.txt', 'w') as f:
#     f.write("[ Memory usage increase from snapshot 1 to snapshot 2 ]\n")
#     for stat in stats_1[:10]:
#         f.write(f"{stat}\n")
#
#     # Detailed traceback for the top memory consumers
#     f.write("\n[ Detailed traceback for the top memory consumers ]\n")
#     for stat in stats_1[:-1]:
#         f.write('\n'.join(stat.traceback.format()) + '\n\n\n')
