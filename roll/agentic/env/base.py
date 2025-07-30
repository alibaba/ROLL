from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Optional


class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    The class needs to handle text-based input, input may be invalid
        - Environment will track the total reward for the trajectory

    """

    def __init__(self, config):
        self.config: BaseEnvConfig = config

    @abstractmethod
    def reset(self, seed=None, **kwargs) -> Tuple[Any, dict]:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed, IMPORTANT,IMPORTANT,IMPORTANT
        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: llm response, parser_action by self.parser_action
        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically, a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
        """
        pass

    # below are optional methods
    def parse_action(self, text):
        pass

    def render(self, mode: str = "text") -> Any:
        """Render the environment. Optional method."""
        pass

    def close(self):
        """Close the environment."""
        pass

    def get_all_actions(self) -> List[str]:
        """Get list of all valid actions."""
        return []


@dataclass
class BaseEnvConfig(ABC):
    """
    Abstract base class for environment configurations.
    """
    max_steps: int = 10

    env_instruction: str = ""
    action_pattern: str = r"<answer>(.*?)</answer>"

    # used for partition datasets
    # TODO: We need to consider the pressure caused by multiple environments (envs) reading the dataset concurrently.
    group_id: int = 0
    group_size: int = 1
