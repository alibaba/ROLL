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
import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import re
import random
from roll.agentic.env.base import BaseLanguageBasedEnv
from roll.agentic.utils import all_seed
from .config import MetaMathQAEnvConfig


class MetaMathQAEnv(BaseLanguageBasedEnv):
    def __init__(self, config: MetaMathQAEnvConfig):
        super(MetaMathQAEnv, self).__init__()

        self.config = config
        self.dataset = load_dataset(path=self.config.dataset_path, cache_dir=self.config.cache_dir)
        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.render_cache = None

    def _extract_answer(self, response):
        match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
        print(response)
        if match:
            return match.group(1).strip()
        return None

    def reset(self, seed=None):
        dataset = self.dataset[self.config.split]
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        question_data = dataset[self.current_question_idx]
        self.current_question = question_data["query"]
        self.correct_answer = self._extract_answer(question_data["response"])
        self.step_num = 0
        self.render_cache = self.current_question
        return self.render_cache

    def step(self, action):
        is_correct, is_valid = self._check_answer(action)
        reward = 1.0 / (2**self.step_num) if is_correct else 0.0
        if is_correct:
            observation = "Correct!"
            done = True
        else:
            observation = "Incorrect. Please think again."
            done = False
        self.step_num += 1
        info = {"action_is_valid": is_valid, "success": is_correct}
        self.render_cache = observation
        return self.render_cache, reward, done, info

    def _check_answer(self, user_answer):
        """Check if the user's answer matches the correct answer."""
        user_answer = user_answer.strip()
        normalized_answer = re.sub(r"\s+", "", user_answer.lower())
        if self.correct_answer:
            normalized_label = re.sub(r"\s+", "", self.correct_answer.lower())
            is_correct = normalized_answer == normalized_label
        is_valid = normalized_answer != ""
        return is_correct, is_valid

    def render(self, mode: str = "text"):
        return self.render_cache


if __name__ == "__main__":
    # Create the environment configuration
    config = MetaMathQAEnvConfig(dataset_path="meta-math/MetaMathQA", cache_dir="./data", split="train")

    # Initialize the environment
    env = MetaMathQAEnv(config)

    # Reset the environment to get the first question
    print("Question:")
    question = env.reset(seed=42)
    print(question)
    print("\nCorrect answer (for testing purposes):")
    print(env.correct_answer)

    # Interactive loop for testing
    while True:
        user_answer = input("\nEnter your answer (or 'q' to quit): ")
        if user_answer.lower() == "q":
            break

        # Take a step in the environment with the user's answer
        # breakpoint()
        obs, reward, done, info = env.step(user_answer)

        # Print the results
        print("\nFeedback:", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)

        # If the episode is done, reset the environment for a new question
        if done:
            print("\n--- New Question ---")
            question = env.reset()
            print(question)
            print("\nCorrect answer (for testing purposes):")
            print(env.correct_answer)
