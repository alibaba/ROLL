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
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config
from tests.distributed.strategy.model_update.model_update_pipeline import ModelUpdatePipeline

logger = get_logger()


def vllm_model_update_baseline():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./model_update", config_name="model_update_baseline_config")
    # vllm_strategy_args = StrategyArguments(strategy_name="vllm",
    #                                        strategy_config={
    #                                            "gpu_memory_utilization": 0.8,
    #                                            "block_size": 16,
    #                                        })
    #
    # ppo_config.actor_infer.strategy_args = vllm_strategy_args

    pipeline = ModelUpdatePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()
    generate_times = [metric["time/model_update"] for metric in metric_list[:-2]]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")

    output_file = "model_update_baseline.json"
    with open(output_file, "w") as f:
        json.dump(metric_list, f, ensure_ascii=False)


def ds_2_hf_model_update_baseline():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./model_update", config_name="model_update_baseline_config")

    pipeline = ModelUpdatePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()
    generate_times = [metric["time/model_update"] for metric in metric_list]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")


if __name__ == "__main__":
    vllm_model_update_baseline()
    # ds_2_hf_model_update_baseline()
