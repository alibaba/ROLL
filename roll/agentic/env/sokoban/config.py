from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List

from roll.agentic.env.base import BaseEnvConfig


@dataclass
class SokobanEnvConfig(BaseEnvConfig):
    dim_room: Tuple[int, int] = (6, 6)
    max_steps: int = 100
    num_boxes: int = 3
    search_depth: int = 300
    grid_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"}
    )
    grid_vocab: Optional[Dict[str, str]] = field(
        default_factory=lambda: {
            "#": "wall",
            "_": "empty",
            "O": "target",
            "√": "box on target",
            "X": "box",
            "P": "player",
            "S": "player on target",
        }
    )
    action_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
    )
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"

    env_instruction: str = "You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer must be one of action in a turn, format is <answer>Right</answer>"
    action_pattern: str = r"<answer>(.*?)</answer>"
    max_tokens_per_step: int = 128
    special_token_list: Optional[List[str]] = field(default_factory=lambda: ["<think>", "</think>", "<answer>",
                                                                             "</answer>", "<|im_start|>", "<|im_end|>"])

    def __post_init__(self):
        if self.dim_x is not None and self.dim_y is not None:
            self.dim_room = (self.dim_x, self.dim_y)
            delattr(self, "dim_x")
            delattr(self, "dim_y")

        grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
            [f"{k}: {v}" for k, v in self.grid_vocab.items()])
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{v}" for k, v in self.action_lookup.items()])
        self.env_instruction = self.env_instruction + grid_vocab_str + action_lookup_str
