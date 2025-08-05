from dataclasses import dataclass, field
from typing import Any

import spacy  # temporary fix of segmentation fault when importing pyserini.search.lucene before spacy
from webshop_minimal import init_basedir
from webshop_minimal.utils import DEFAULT_FILE_PATH

from roll.agentic.env.base import BaseEnvConfig

init_basedir()  # init DEFAULT_FILE_PATH, hardcoded dataset to small


@dataclass
class WebShopEnvConfig(BaseEnvConfig):
    """Configuration for WebAgentText environment"""

    # dataset: str = field(default="small", metadata={"description": "Small or full dataset"})
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

    max_steps: int = 10
    env_instruction: str = ("You are web shopping. I will give you instructions about what to do. "
                            "You have to follow the instructions. Every round I will give you an observation and "
                            "a list of available actions, you have to respond an action based on the state and instruction. "
                            "You can use search action if search is available. You can click one of the buttons in clickables. "
                            "An action should be of the following structure: search[keywords] click[value] If the action is not valid, perform nothing. "
                            "Keywords in search are up to you, but the value in click must be a value in the list of available actions. "
                            "Remember that your keywords in search should be carefully designed. "
                            "Your response should use the following format Thought: I think ... Action: click[something]")
    action_pattern: str = r"<answer>(.*?)</answer>"
