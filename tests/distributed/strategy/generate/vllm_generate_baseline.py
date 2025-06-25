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
import os

from roll.configs.worker_config import StrategyArguments
from roll.distributed.scheduler.initialize import init
from tests.distributed.strategy.generate.generate_pipeline import GeneratePipeline
from tests.distributed.strategy.make_baseline_config import make_baseline_config
from roll.utils.logging import get_logger

logger = get_logger()


def vllm_generate_baseline():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./generate", config_name="generate_baseline_config")

    vllm_strategy_args = StrategyArguments(
        strategy_name="vllm",
        strategy_config={
            "gpu_memory_utilization": 0.8,
            "block_size": 16,
        },
    )
    ppo_config.generate_opt_level = 0
    ppo_config.actor_infer.strategy_args = vllm_strategy_args

    pipeline = GeneratePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()

    output_file = "generate_baseline.json"
    with open(output_file, "w") as f:
        json.dump(metric_list, f, ensure_ascii=False)

    generate_times = [metric["time/generate"] for metric in metric_list]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")

    import ray

    ray.timeline(filename="timeline.json")
    ray._private.state.object_transfer_timeline(filename="object_timeline.json")


def vllm_generate_baseline_tp_2():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./generate", config_name="generate_baseline_config")

    vllm_strategy_args = StrategyArguments(
        strategy_name="vllm",
        strategy_config={
            "gpu_memory_utilization": 0.8,
            "block_size": 16,
            "tensor_parallel_size": 2,
        },
    )

    ppo_config.actor_infer.strategy_args = vllm_strategy_args
    ppo_config.generate_opt_level = 0

    pipeline = GeneratePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()

    output_file = "generate_baseline.json"
    with open(output_file, "w") as f:
        json.dump(metric_list, f, ensure_ascii=False)

    generate_times = [metric["time/generate"] for metric in metric_list]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")

    import ray

    ray.timeline(filename="timeline.json")
    ray._private.state.object_transfer_timeline(filename="object_timeline.json")


def vllm_async_generate_baseline():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./generate", config_name="generate_baseline_config")

    vllm_strategy_args = StrategyArguments(
        strategy_name="vllm",
        strategy_config={
            "gpu_memory_utilization": 0.8,
            "block_size": 16,
            "tensor_parallel_size": 2,
        },
    )
    ppo_config.generate_opt_level = 1
    ppo_config.is_num_return_sequences_expand = True
    ppo_config.actor_infer.strategy_args = vllm_strategy_args

    pipeline = GeneratePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()

    output_file = "generate_baseline.json"
    with open(output_file, "w") as f:
        json.dump(metric_list, f, ensure_ascii=False)

    generate_times = [metric["time/generate"] for metric in metric_list]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")

    import ray

    ray.timeline(filename="timeline.json")
    ray._private.state.object_transfer_timeline(filename="object_timeline.json")


if __name__ == "__main__":
    # vllm_generate_baseline()
    # vllm_generate_baseline_tp_2()
    vllm_async_generate_baseline()
