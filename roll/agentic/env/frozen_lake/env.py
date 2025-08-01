import re

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
import numpy as np

from roll.agentic.env.base import BaseEnv
from roll.agentic.env.parse_action_utils import default_parser_action_func
from .config import FrozenLakeEnvConfig
from .utils import generate_random_map
from roll.agentic.utils import all_seed


class FrozenLakeEnv(BaseEnv, GymFrozenLakeEnv):
    def __init__(self, config: FrozenLakeEnvConfig = FrozenLakeEnvConfig()):
        BaseEnv.__init__(self, config)
        self.config = config
        # Using mappings directly from config
        self.GRID_LOOKUP = config.grid_lookup
        self.ACTION_LOOKUP = config.action_lookup
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = config.render_mode
        self.MAP_LOOKUP = config.map_lookup
        random_map = generate_random_map(size=config.size, p=config.p, seed=config.map_seed)
        GymFrozenLakeEnv.__init__(
            self, desc=random_map, is_slippery=config.is_slippery, render_mode=config.render_mode
        )
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        try:
            with all_seed(seed):
                self.config.map_seed = seed
                self.__init__(self.config)
                GymFrozenLakeEnv.reset(self, seed=seed)
                return self.render(), {}
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed)

    def step(self, action: str):
        action_info = self.parse_action(action)
        if action_info["action"] is None:
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": self.desc[self.player_pos] == b"G",
            }
            info = {
                "metrics": metrics,
            }
            info.update(action_info)
            self.step_count += 1
            return self.render(), 0, False, False, info


        prev_pos = int(self.s)
        _, reward, terminated, truncated, _ = GymFrozenLakeEnv.step(self, action_info["action"])
        self.step_count += 1
        next_obs = self.render()
        metrics = {
            "action_is_effective": prev_pos != int(self.s),
            "action_is_valid": True,
            "success": self.desc[self.player_pos] == b"G",
        }
        info = {
            "metrics": metrics,
        }
        info.update(action_info)
        if terminated:
            if not metrics["success"] and self.step_count >= self.config.max_steps:
                truncated = True
        return next_obs, reward, terminated, truncated, info

    def parse_action(self, text):
        return default_parser_action_func(text, self.config.action_pattern, self.config.action_lookup, self.config.special_token_list)

    def render(self, mode=None):
        if not mode:
            mode = self.render_mode
        if mode == "text":
            room = self.desc.copy()
            # replace the position of start 'S' with 'F', mark the position of the player as 'p'.
            room = np.where(room == b"S", b"F", room)
            room[self.player_pos] = b"P"
            room = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room)
            # add player in hole or player on goal
            room[self.player_pos] = (
                4 if self.desc[self.player_pos] == b"H" else 5 if self.desc[self.player_pos] == b"G" else 0
            )
            return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room)
        elif mode == "rgb_array":
            return self._render_gui("rgb_array")
        else:
            raise ValueError(f"Invalid mode: {self.render_mode}")

    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.values()])

    @property
    def player_pos(self):
        return (self.s // self.ncol, self.s % self.ncol)  # (row, col)

    def close(self):
        self.render_cache = None
        super(FrozenLakeEnv, self).close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = FrozenLakeEnvConfig(size=4, p=0.8, is_slippery=False, map_seed=42)
    env = FrozenLakeEnv(config)
    obs, _ = env.reset(seed=42)
    print(obs)
    while True:
        keyboard = input("Enter action: ")
        if keyboard == "q":
            break
        action = int(keyboard)
        assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        action_text = f"<answer>{env.ACTION_LOOKUP[action]}</answer>"
        obs, reward, terminate, truncated, info = env.step(action_text)
        print(obs, reward, terminate, info)
        if terminate:
            break
    np_img = env.render("rgb_array")
    # save the image
    plt.imsave("frozen_lake.png", np_img)
