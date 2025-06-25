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
import gymnasium as gym
from roll.agentic.env.base import BaseLanguageBasedEnv
import datasets
import re
import itertools
from .config import CountdownEnvConfig


def check_format(equation, nums):
    try:
        nums_in_eq = [int(n) for n in re.findall(r"\d+", equation)]
        return sorted(nums_in_eq) == sorted(nums)
    except:
        return False


def check_correctness(equation_str, target):
    try:
        result = eval(equation_str, {"__builtins__": None}, {})
        return abs(result - target) < 1e-5
    except:
        return False


def has_solution(nums, target):
    """Check if there is a valid equation using each number exactly once."""
    # pad nums all to 4 numbers
    length = 4
    nums = nums + [0] * (length - len(nums))
    # +- num1 +- num2 +- num3 +- num4 = target, try all
    combinations = list(itertools.product([1, -1], repeat=length))
    for combination in combinations:
        if sum(combination[i] * nums[i] for i in range(length)) == target:
            return True
    return False


class CountdownEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else CountdownEnvConfig()
        self.data = self._get_data_from_parquet(self.config.train_path)
        self.index = None
        self.render_cache = None
        self.render_mode = self.config.render_mode
        assert self.render_mode == "text"

    def _get_data_from_parquet(self, path):
        df = datasets.load_dataset("parquet", data_files=path)["train"].select(range(self.config.max_instances))
        df = df.filter(lambda x: has_solution(x["nums"], x["target"]))
        return df

    def reset(self, seed=None):
        gym.Env.reset(self, seed=seed)
        self.index = seed % len(self.data)
        data = self.data[self.index]
        self.render_cache = f"Target: {data['target']}, nums: {data['nums']}"
        return self.render_cache

    def step(self, action):
        reward = self.compute_reward(action, self.data[self.index])
        next_obs, done, info = (
            f"Your answer get {reward} points.",
            True,
            {"action_is_effective": reward > 0, "action_is_valid": True, "success": reward == self.config.score},
        )
        self.render_cache = next_obs
        return next_obs, reward, done, info

    def render(self, mode: str = "text"):
        return self.render_cache

    def compute_reward(self, action, ground_truth):
        """Score the countdown task solution."""
        target = ground_truth["target"]
        nums = ground_truth["nums"]
        if not check_format(action, nums):
            return 0
        if not check_correctness(action, target):
            return self.config.format_score
        else:
            return self.config.score

    def close(self):
        pass


if __name__ == "__main__":

    def test(path, seed=43):
        config = CountdownEnvConfig(train_path=path)
        env = CountdownEnv(config)
        obs = env.reset(seed=seed)
        problem = env.data[env.index]
        solution = f"- {problem['nums'][0]} + {problem['nums'][1]} + {problem['nums'][2]}"
        _, reward, _, _ = env.step(solution)
        print(f"{obs}\nSolution: {solution}, Reward: {reward}")

    test("data/countdown/train.parquet")
