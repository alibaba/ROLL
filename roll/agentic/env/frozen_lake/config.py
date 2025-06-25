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
class FrozenLakeEnvConfig:
    """Configuration for FrozenLake environment"""

    # Map config
    size: int = 4
    p: float = 0.8
    is_slippery: bool = True
    map_seed: Optional[int] = None
    render_mode: str = "text"

    # Mappings
    action_map: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 1, 3: 2, 4: 3})
    map_lookup: Dict[bytes, int] = field(
        default_factory=lambda: {b"P": 0, b"F": 1, b"H": 2, b"G": 3}
    )  # b'' string is used for vectorization in numpy
    # P: Player; F: Frozen; H: Hole; G: Goal
    grid_lookup: Dict[int, str] = field(default_factory=lambda: {0: "P", 1: "_", 2: "O", 3: "G", 4: "X", 5: "√"})
    grid_vocab: Dict[str, str] = field(
        default_factory=lambda: {
            "P": "player",
            "_": "empty",
            "O": "hole",
            "G": "goal",
            "X": "player in hole",
            "√": "player on goal",
        }
    )
    action_lookup: Dict[int, str] = field(default_factory=lambda: {1: "Left", 2: "Down", 3: "Right", 4: "Up"})
