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

from roll.distributed.scheduler.initialize import init
from tests.distributed.strategy.generate.generate_pipeline import GeneratePipeline
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config

logger = get_logger()


def hf_generate_baseline():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./generate", config_name="generate_baseline_config")
    ppo_config.generate_opt_level = 0

    pipeline = GeneratePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()

    generate_times = [metric["time/generate"] for metric in metric_list]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")

    import ray

    ray.timeline(filename="timeline.json")
    ray._private.state.object_transfer_timeline(filename="object_timeline.json")


if __name__ == "__main__":
    hf_generate_baseline()
