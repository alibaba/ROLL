import multiprocessing
import random
from typing import Optional, Tuple, Any, SupportsFloat

from gem import Env
from gem.envs.math_env import MathEnv as GEMMathEnv
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer


class MathEnv(GEMMathEnv):

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        Env.reset(self, seed)
        if seed is not None:
            self.idx = random.randint(0, len(self.dataset) - 1)
        else:
            if self.idx == len(self.dataset):
                self.epoch += 1
                self.dataset = self.dataset.shuffle(seed=self.seed + self.epoch)
                self.idx = 0

        data = self.dataset[self.idx]
        self.first_obs = data[self.question_key]
        self.answer = data[self.answer_key]
        self.idx += 1
        return self.first_obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = extract_last_boxed_answer(action)
        action_is_valid = True
        if model_answer is None:
            reward = 0
            action_is_valid = False
        else:
            res = self.mp_pool.apply_async(
                self.check_correct, (model_answer, self.answer)
            )
            try:
                is_correct = res.get(timeout=1)
            except (multiprocessing.context.TimeoutError, Exception):
                is_correct = False
            reward = 1.0 if is_correct else 0

        metrics = {
            "action_is_valid": action_is_valid,
            "success": reward > 0,
            "raw_reward": reward,
        }
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "raw_reward": "last",
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode
        }
        return TERMINAL_STATE, reward, True, True, info