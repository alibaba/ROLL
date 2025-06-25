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
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AlfredEnvConfig(BaseEnvConfig):
    """configuration for text world AlfredEnv"""

    config_file: str = "./ragen/env/alfworld/alfworld_config.yaml"
    action_lookup: Dict[int, str] = field(
        default_factory=lambda: {
            1: "look",
            2: "inventory",
            3: "go to <receptacle>",
            4: "open <receptacle>",
            5: "close <receptacle>",
            6: "take <object> from <receptacle>",
            7: "move <object> to <receptacle>",
            8: "examine <something>",
            9: "use <object>",
            10: "heat <object> with <receptacle>",
            11: "clean <object> with <receptacle>",
            12: "cool <object> with <receptacle>",
            13: "slice <object> with <object>",
        }
    )
    format_score: float = 0.1
    score: float = 1.0
    render_mode: str = "text"
