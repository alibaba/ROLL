import random
from typing import Tuple, Any, SupportsFloat, Optional

from datasets import Dataset
from gem.envs.qa_env import QaEnv as GEMQaEnv
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE

class QaEnv(GEMQaEnv):
    def __init__(
        self,
        dataset_name: Optional[str] = "",
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        question_key: str = "question",
        answer_key: str = "answer",
        seed: int = 0,
        extract_boxed: bool = False,
        load_from_cache_file: bool = True,  # False to force re-run the apply_prompt_func, useful when apply_prompt is changed
        **_,
    ):
        from datasets import tqdm
        tqdm.set_lock(tqdm.get_lock())
        super().__init__(dataset_name=dataset_name,
                         split=split,
                         dataset=dataset,
                         question_key=question_key,
                         answer_key=answer_key,
                         seed=seed,
                         extract_boxed=extract_boxed,
                         load_from_cache_file=load_from_cache_file, **_)

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