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
from roll.agentic.env.base import BaseEnvConfig
from dataclasses import dataclass


@dataclass
class CountdownEnvConfig:
    train_path: str = "data/countdown/train.parquet"
    max_instances: int = 20000
    render_mode: str = "text"
    score = 1
    format_score = 0.1
