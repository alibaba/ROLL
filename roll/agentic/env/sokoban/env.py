import re

import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np

from roll.agentic.env.base import BaseEnv
from roll.agentic.env.parse_action_utils import default_parser_action_func
from .utils import generate_room

from roll.agentic.env.sokoban.config import SokobanEnvConfig
from roll.agentic.utils import all_seed


class SokobanEnv(BaseEnv, GymSokobanEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SokobanEnvConfig()
        BaseEnv.__init__(self, config=self.config)
        self.GRID_LOOKUP = self.config.grid_lookup
        self.ACTION_LOOKUP = self.config.action_lookup
        self.search_depth = self.config.search_depth
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = self.config.render_mode

        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.dim_room,
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
            **kwargs,
        )

    def reset(self, seed=None):
        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth,
                )
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            self.player_position = np.argwhere(self.room_state == 5)[0]
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
                "success": self.boxes_on_target == self.num_boxes,
            }
            info = {
                "metrics": metrics,
            }
            info.update(action_info)
            return self.render(), 0, False, False, info

        previous_pos = self.player_position
        _, reward, terminated, _ = GymSokobanEnv.step(self, action_info["action"])
        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)

        metrics = {
            "action_is_effective": action_effective,
            "action_is_valid": True,
            "success": self.boxes_on_target == self.num_boxes,
        }
        info = {
            "metrics": metrics,
        }
        info.update(action_info)
        truncated = False
        if terminated:
            truncated = not self._check_if_all_boxes_on_target()

        return next_obs, reward, terminated, truncated, info

    def parse_action(self, text):
        return default_parser_action_func(text, self.config.action_pattern, self.config.action_lookup, self.config.special_token_list)

    def render(self, mode=None):
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == "text":
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        elif render_mode == "rgb_array":
            return self.get_image(mode="rgb_array", scale=1)
        else:
            raise ValueError(f"Invalid mode: {render_mode}")

    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.values()])

    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
    env = SokobanEnv(config)
    for i in range(10):
        obs, _ = env.reset(seed=1010 + i)
        print(obs)
        print()
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
    np_img = env.get_image("rgb_array")
    # save the image
    plt.imsave("sokoban1.png", np_img)
