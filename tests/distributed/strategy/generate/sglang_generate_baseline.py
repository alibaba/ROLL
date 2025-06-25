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


def sglang_generate_baseline():
    init()

    ppo_config = make_baseline_config(config_path="./generate", config_name="generate_baseline_config")

    sglang_strategy_args = StrategyArguments(
        strategy_name="sglang",
        strategy_config={
            "mem_fraction_static": 0.8,
            #    "tensor_parallel_size": 8,
        },
    )
    ppo_config.actor_infer.strategy_args = sglang_strategy_args
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


if __name__ == "__main__":
    sglang_generate_baseline()
