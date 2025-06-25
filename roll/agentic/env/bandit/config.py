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
from dataclasses import dataclass
from typing import Dict


@dataclass
class BanditEnvConfig:
    lo_arm_name: str = "phoenix"
    hi_arm_name: str = "dragon"
    action_space_start: int = 1
    lo_arm_score: float = 0.1
    hi_arm_loscore: float = 0.0
    hi_arm_hiscore: float = 1.0
    hi_arm_hiscore_prob: float = 0.25
    render_mode: str = "text"
    action_lookup: Dict[int, str] = None  # defined in env.py
