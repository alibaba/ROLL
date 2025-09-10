from typing import Tuple, Any, SupportsFloat, Optional

from datasets import Dataset
from gem.envs.code_env import CodeEnv as GEMCodeEnv
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_code_from_model


class CodeEnv(GEMCodeEnv):
    def __init__(
            self,
            dataset_name: Optional[str] = "",
            split: Optional[str] = None,
            dataset: Optional[Dataset] = None,
            question_key: str = "problem",
            test_key: str = "tests",
            seed: int = 0,
            max_workers: int = 5,
            max_tests: int = 12,
            verbose: bool = False,
            sandbox_type: str = "none",
            **_,
    ):
        from datasets import tqdm
        tqdm.set_lock(tqdm.get_lock())
        super().__init__(dataset_name=dataset_name,
                         split=split,
                         dataset=dataset,
                         question_key=question_key,
                         test_key=test_key,
                         seed=seed,
                         max_workers=max_workers,
                         max_tests=max_tests,
                         verbose=verbose,
                         sandbox_type=sandbox_type, **_)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:

        model_code = extract_code_from_model(action)
        action_is_valid = True
        if model_code is None:
            action_is_valid = False
            reward = 0.0
        else:
            is_correct = self._check_correct(model_code)
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