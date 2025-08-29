import random
from typing import Tuple, Any, SupportsFloat, Optional
from gem.envs.qa_env import QaEnv as GEMQaEnv
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE

class QaEnv(GEMQaEnv):

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = self.extractor(action)
        action_is_valid = True
        if model_answer is None:
            reward = 0.0
            action_is_valid = False
        else:
            is_correct = self.check_correct(model_answer, self.answer)
            reward = 1.0 if is_correct else 0.0
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