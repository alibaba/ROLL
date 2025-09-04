import gem
import random

from typing import Any

from gem import Env
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np

from roll.agentic.env.parse_action_utils import default_parser_action_func
from .utils import generate_room

from roll.agentic.utils import all_seed


class SokobanEnv(Env, GymSokobanEnv):
    def __init__(self,
                 render_mode="text",
                 dim_room=(10, 10),
                 max_steps=20,
                 num_boxes=4,
                 search_depth=300,
                 grid_lookup=None,
                 grid_vocab=None,
                 action_lookup=None,
                 env_instruction=None,
                 format_penalty=0.0,
                 action_pattern="^<answer>(.*?)</answer>$",
                 special_token_list=("<|im_start|>", "<|im_end|>"),
                 **kwargs):
        self.GRID_VOCAB = {"#": "wall", "_": "empty", "O": "target", "√": "box on target", "X": "box", "P": "player", "S": "player on target"}
        self.GRID_LOOKUP = {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"}
        self.ACTION_LOOKUP = {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
        self.env_instruction = (
            "You are solving the Sokoban puzzle. "
            "You are the player and you need to push all boxes to targets. "
            "When you are right next to a box, you can push it by moving in the same direction. "
            "You cannot push a box through a wall, and you cannot pull a box. "
            f"The answer must be one of action in a turn, format is <answer>Right</answer>."
        )
        if grid_lookup is not None:
            self.GRID_LOOKUP = grid_lookup
        if grid_vocab is not None:
            self.GRID_VOCAB = grid_vocab
        if action_lookup is not None:
            self.ACTION_LOOKUP = action_lookup
        if env_instruction is not None:
            self.env_instruction = env_instruction
        self.search_depth = search_depth
        self.render_mode = render_mode

        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list

        GymSokobanEnv.__init__(
            self,
            dim_room=dim_room,
            max_steps=max_steps,
            num_boxes=num_boxes,
            **kwargs,
        )

    def get_instructions(self) -> str:
        grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
            [f"{k}: {v}" for k, v in self.GRID_VOCAB.items()])
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{v}" for k, v in self.ACTION_LOOKUP.items()])
        return self.env_instruction + grid_vocab_str + action_lookup_str

    def get_task_suffix(self) -> Any:
        if self.render_mode == "text":
            return (
                f"Here is the current state of the Sokoban puzzle:\n{self.render(mode='text')}\n"
            )
        else:
            return self.render(mode=self.render_mode)

    def reset(self, seed=None):
        Env.reset(self, seed)
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

            # TODO: `env.reset()` does not return the raw state; how should we describe the image-based state?
            #       Currently returning the image via suffix instead.
            return self.get_instructions(), {"suffix": self.get_task_suffix()}
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed)

    def step(self, action: str):
        action_info = self.parse_action(action)

        if action_info["action"] is None:
            _, reward, terminated, _ = GymSokobanEnv.step(self, 0)
            reward += self.format_penalty

            terminate_obs = f"At turn {self.num_env_steps}, You did not provide a valid action."
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": self.boxes_on_target == self.num_boxes,
                "format_penalty": self.format_penalty
            }
            info = {
                "suffix": self.get_task_suffix(),
                "metrics": metrics,
            }
            info.update(action_info)
            return terminate_obs, reward, False, False, info

        previous_pos = self.player_position
        _, reward, terminated, _ = GymSokobanEnv.step(self, action_info["action"])

        action_effective = not np.array_equal(previous_pos, self.player_position)
        if not action_effective:
            next_obs = f"At turn {self.num_env_steps}, you tried to move {action_info['action_content']}, which is not effective yet."
        else:
            next_obs = f"At turn {self.num_env_steps}, you moved {action_info['action_content']}, which is effective."

        metrics = {
            "action_is_effective": action_effective,
            "action_is_valid": True,
            "success": self.boxes_on_target == self.num_boxes,
            "format_penalty": 0,
        }
        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }
        info = {
            "suffix": self.get_task_suffix(),
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode
        }
        info.update(action_info)
        truncated = False
        if terminated:
            truncated = not self._check_if_all_boxes_on_target()

        return next_obs, reward, terminated, truncated, info

    def parse_action(self, text):
        return default_parser_action_func(text, self.action_pattern, self.ACTION_LOOKUP, self.special_token_list)

    def render(self, mode=None):
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == "text":
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        elif render_mode == "rgb_array":
            return self.get_image(mode="rgb_array", scale=1)
        else:
            raise ValueError(f"Invalid mode: {render_mode}")

    def sample_random_action(self):
        return random.choice(list([k for k in self.ACTION_LOOKUP.values()]))

    def close(self):
        super(SokobanEnv, self).close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env: SokobanEnv = gem.make(env_id="sokoban", dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
    for i in range(10):
        obs, info = env.reset(seed=1010 + i)
        print(obs, info["suffix"])
        print()
    while True:
        keyboard = input("Enter action: ")
        if keyboard == "q":
            break
        action = int(keyboard)
        assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        action_text = f"<answer>{env.ACTION_LOOKUP[action]}</answer><|im_end|>"
        import pdb; pdb.set_trace()
        obs, reward, terminate, truncated, info = env.step(action_text)
        print(obs, reward, terminate, info["suffix"])
        if terminate:
            break
    np_img = env.get_image("rgb_array")
    # save the image
    plt.imsave("sokoban1.png", np_img)
