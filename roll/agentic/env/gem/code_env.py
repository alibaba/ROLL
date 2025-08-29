from typing import Tuple, Any, SupportsFloat

from gem.envs.code_env import CodeEnv as GEMCodeEnv
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_code_from_model


class CodeEnv(GEMCodeEnv):

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