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
from typing import Union

from roll.distributed.executor.worker import Worker
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy


def create_strategy(worker: Worker) -> Union[InferenceStrategy, TrainStrategy]:
    strategy_name = worker.worker_config.strategy_args.strategy_name

    # Lazy import strategy to avoid cuda initialized
    if strategy_name == "deepspeed_infer":
        from roll.distributed.strategy.deepspeed_strategy import DeepSpeedInferStrategy as strategy_cls
    elif strategy_name == "deepspeed_train":
        from roll.distributed.strategy.deepspeed_strategy import DeepSpeedTrainStrategy as strategy_cls
    elif strategy_name == "hf_infer":
        from roll.distributed.strategy.hf_strategy import HfInferStrategy as strategy_cls
    elif strategy_name == "vllm":
        from roll.distributed.strategy.vllm_strategy import VllmStrategy as strategy_cls
    elif strategy_name == "sglang":
        from roll.distributed.strategy.sglang_strategy import SgLangStrategy as strategy_cls
    elif strategy_name == "megatron_infer":
        from roll.distributed.strategy.megatron_strategy import MegatronInferStrategy as strategy_cls
    elif strategy_name == "megatron_train":
        from roll.distributed.strategy.megatron_strategy import MegatronTrainStrategy as strategy_cls
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

    return strategy_cls(worker)
