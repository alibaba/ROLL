from typing import Optional, List, Dict
from dataclasses import dataclass, field

from roll.agentic.env.base import BaseEnvConfig


@dataclass
class FrozenLakeEnvConfig(BaseEnvConfig):
    """Configuration for FrozenLake environment"""

    # Map config
    size: int = 4
    p: float = 0.8
    is_slippery: bool = True
    map_seed: Optional[int] = None
    render_mode: str = "text"

    # Mappings
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
    action_lookup: Dict[int, str] = field(default_factory=lambda: {0: "Left", 1: "Down", 2: "Right", 3: "Up"})

    max_steps: int = 100
    env_instruction: str = "You are solving the FrozenLake puzzle. Forbid the whole and go to the target. You may move to the unintended direction due to the slippery ice. The answer must be one of action in a turn, format is <answer>Right</answer>"
    action_pattern: str = r"<answer>(.*?)</answer>"
    special_token_list: Optional[List[str]] = field(default_factory=lambda: ["<think>", "</think>", "<answer>",
                                                                             "</answer>", "<|im_start|>", "<|im_end|>"])

    def __post_init__(self):
        grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
            [f"{k}: {v}" for k, v in self.grid_vocab.items()])
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{v}" for k, v in self.action_lookup.items()])
        self.env_instruction = self.env_instruction + grid_vocab_str + action_lookup_str