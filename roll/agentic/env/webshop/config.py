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
from dataclasses import dataclass, field
from typing import Any

from webshop_minimal.utils import (
    DEFAULT_FILE_PATH,
)


@dataclass
class WebShopEnvConfig:
    """Configuration for WebAgentText environment"""

    observation_mode: str = field(default="text", metadata={"choices": ["html", "text"]})
    file_path: str = field(
        default=DEFAULT_FILE_PATH, metadata={"description": "File path for SimServer"}
    )  # TODO: Remove hardcoded file path
    server: Any = field(default=None, metadata={"description": "If None, use SimServer"})
    filter_goals: Any = field(
        default=None,
        metadata={"description": "SimServer arg: Custom function to filter specific goals for consideration"},
    )
    limit_goals: int = field(
        default=-1, metadata={"description": "SimServer arg: Limit the number of goals available"}
    )
    num_products: int = field(
        default=None, metadata={"description": "SimServer arg: Number of products to search across"}
    )
    human_goals: bool = field(
        default=False, metadata={"description": "SimServer arg: Load human goals if True, otherwise synthetic goals"}
    )
    show_attrs: bool = field(
        default=False, metadata={"description": "SimServer arg: Whether to show additional attributes"}
    )
