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
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class StaticEnvConfig:
    """Configuration for StaticEnv environment"""

    # Dataset config
    dataset_name: str = field(default="metamathqa")  # metamathqa, gsm8k,theoremqa,mmlu
    cache_dir: str = field(default="./data")
    split: Optional[str] = field(default=None)
