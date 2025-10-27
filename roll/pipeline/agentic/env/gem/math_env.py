import logging
import multiprocessing
import random
from typing import Optional, Tuple, Any, SupportsFloat, Dict

from datasets import load_dataset, Dataset, DatasetDict
from gem import Env
from gem.envs.math_env import MathEnv as GEMMathEnv
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer
import ray

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE

logger = logging.getLogger(__name__)

class MathEnv(GEMMathEnv):

    def __init__(
            self,
            dataset_name: Optional[str] = "",
            split: Optional[str] = None,
            dataset: Optional[Dataset] = None,
            question_key: str = "problem",
            answer_key: str = "answer",
            seed: int = 0,
            mode: str = "train",
            **_,
    ):
        Env.__init__(self)
        self.seed = seed
        self.question_key = question_key
        self.answer_key = answer_key
        self.mode = mode

        # Convert train/val mode to sample/traversal for GlobalDataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(name=f"{self.mode}_{dataset_name}",
                                             get_if_exists=True,
                                             namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name,
                                                                             split=split,
                                                                             mode=global_dataset_mode)
        self.dataset_manager = GlobalDatasetManager.options(name=f"{self.mode}_dataset_manager",
                                                            get_if_exists=True,
                                                            namespace=RAY_NAMESPACE).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))
        self.idx = 0
        self.epoch = 0
        # Process pool is used to enable the timeout mechanism for answer grading in a potential distributed training setup
        self.mp_pool = multiprocessing.Pool(1)

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        Env.reset(self, seed)
        data: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None
        self.first_obs = data[self.question_key]
        self.answer = data[self.answer_key]
        self.idx += 1
        return self.first_obs, {"env_instruction": ""}

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