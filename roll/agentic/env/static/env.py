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
import numpy as np
from datasets import load_dataset
import re
import random
from typing import Dict, Any, Optional, List, Tuple, Callable
from roll.agentic.env.base import BaseLanguageBasedEnv
from roll.agentic.utils import all_seed
from .config import StaticEnvConfig
from .utils import REGISTERD_STATIC_ENV


class StaticEnv(BaseLanguageBasedEnv):
    """
    A general environment for evaluating language models on Hugging Face datasets.
    Supports multiple datasets: MetaMathQA, TheoremQA, MATH, MMLU-STEM, GSM8K, etc.
    """

    def __init__(self, config: StaticEnvConfig):
        super(StaticEnv, self).__init__()

        self.config = config
        dataset_config = getattr(config, "dataset_config", None)
        if dataset_config is None:
            dataset_config = REGISTERD_STATIC_ENV[self.config.dataset_name]["config"]
        self.dataset = load_dataset(**dataset_config, cache_dir=self.config.cache_dir)

        if self.config.split is None:
            self.split = list(self.dataset.keys())[0]
        else:
            self.split = self.config.split

        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None

        self.processor = REGISTERD_STATIC_ENV[self.config.dataset_name]["processor"]
        self.compute_score = REGISTERD_STATIC_ENV[self.config.dataset_name]["compute_score"]

    def reset(self, seed=None):
        """Reset the environment and get a new question."""
        dataset_split = self.dataset[self.split]
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset_split) - 1)
        question_data = dataset_split[self.current_question_idx]
        self.current_question, self.correct_answer = self.processor(question_data)
        self.step_num = 0

        return self.current_question

    def step(self, action):
        """Take a step in the environment with the given action (answer)."""
        score_result = self.compute_score(action, self.correct_answer)
        is_correct = score_result["is_correct"]
        is_valid = score_result["is_valid"]
        reward = 1.0 / (2**self.step_num) if is_correct else 0.0
        if is_correct:
            observation = "Correct!"
            done = True
        else:
            observation = "Incorrect. Please think again."
            done = False

        self.step_num += 1
        info = {
            "success": is_correct,
            "is_valid": is_valid,
        }

        return observation, reward, done, info


if __name__ == "__main__":
    # Example usage

    for dataset_name in REGISTERD_STATIC_ENV.keys():
        config = StaticEnvConfig(
            dataset_name=dataset_name,
            cache_dir="./data",
        )

        # Initialize the environment
        env = StaticEnv(config)

        # Reset the environment to get the first question
        print("\n--- New Question ---")
        obs = env.reset(seed=42)
        print(obs)

        print("\n--- Correct Answer ---")
        print(env.correct_answer)

        # Interactive loop for testing
        while True:
            user_answer = input("\nEnter your answer (or 'q' to quit): ")
            if user_answer.lower() == "q":
                break

            # Take a step in the environment with the user's answer
            obs, reward, done, info = env.step(user_answer)

            # Print the results
            print(f"\n{obs}")

            # If the episode is done, reset the environment for a new question
            if done:
                print(f"\ntotal step: {env.step_num}, reward: {reward}")
                print("\n--- New Question ---")
                question = env.reset()
                print(question)
                print("\n--- Correct Answer ---")
                print(env.correct_answer)
