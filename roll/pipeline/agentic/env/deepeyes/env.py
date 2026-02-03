import os
import random
import requests
import hashlib
import json
import PIL.Image as Image
from io import BytesIO
from typing import Optional, Dict, List, Tuple

import datasets
import ray
import numpy as np
from dacite import from_dict
from gem import Env
from transformers.image_utils import load_image

from roll.configs.data_args import DataArguments
from roll.distributed.scheduler.protocol import DataProto
from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.pipeline.rlvr.rlvr_config import RewardConfig
from roll.pipeline.agentic.llm_proxy.proxy_utils import generate_by_proxy
from roll.utils.checkpoint_manager import file_lock_context
from roll.utils.constants import RAY_NAMESPACE, EpisodeStopReason
from roll.utils.random_utils import all_seed
from roll.utils.logging import get_logger

from .utils import VisualToolBoxV2, get_prompt


logger = get_logger()


def load_images(images, timeout=None):
    out_images = []
    for image in images:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        image = load_image(image, timeout)
        out_images.append(image)
    return out_images


def encode_function(
    data,
    prompt_getter,
    ground_truth_getter,
    image_getter,
    env_getter,
    data_source_getter,
    question_getter,
):
    image_list = []
    for idx, image in enumerate(image_getter(data)):
        try:
            image_out = load_images(image if isinstance(image, (list, tuple)) else [image], timeout=None)
        except Exception as e:
            image_num = len(image) if isinstance(image, (list, tuple)) else 1
            image_out = [Image.new("RGB", (224, 224), (255, 255, 255))] * image_num
        image_list.append(image_out)
    encodings = {
        "data_source": data_source_getter(data),
        "images": image_list,
        "prompt": prompt_getter(data),
        "env_name": env_getter(data),
        "ground_truth": ground_truth_getter(data),
        "question": question_getter(data),
    }
    return encodings


def encode_dataset(dataset, num_proc, encode_function, new_fingerprint=None):
    # regularized data filed
    features = datasets.Features(
        {
            "data_source": datasets.Value(dtype="string"),
            "images": datasets.Sequence(feature=datasets.Image(mode=None, decode=True)),
            "prompt": dataset.features["prompt"],
            "env_name": datasets.Value(dtype="string"),
            "ground_truth": datasets.Value(dtype="string"),
            "question": datasets.Value(dtype="string"),
            # use index to match dataset item with rollout item
            # "index": datasets.Value(dtype="int"),
        }
    )
    remove_columns = list(dataset.features.keys() - features.keys())
    prompt_getter = lambda data: data["prompt"]
    ground_truth_getter = lambda data: [x["ground_truth"] for x in data["reward_model"]]
    image_getter = lambda data: data["images"]
    env_getter = lambda data: data["env_name"]
    data_source_getter = lambda data: data["data_source"]
    question_getter = lambda data: [x["question"] for x in data["extra_info"]]
    logger.info(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(
            data,
            prompt_getter,
            ground_truth_getter,
            image_getter,
            env_getter,
            data_source_getter,
            question_getter,
        ),
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        features=features,
        remove_columns=remove_columns,
        new_fingerprint=new_fingerprint,
        desc="Encoding dataset",
    )
    logger.info(f"Encoding: {dataset}")
    return dataset


@ray.remote
class DeepEyesDataset(GlobalDataset.__ray_actor_class__):
    def __init__(
        self,
        dataset_name,
        split: str = "train",
        mode="sample",
        dataset_kwargs: Dict = None,
        seed: Optional[int] = None,
        epoch: Optional[int] = 0,
        idx: Optional[int] = 0,
    ):
        num_proc = dataset_kwargs.pop("num_proc", 1)
        logger.info("load dataset")
        super().__init__(dataset_name, split, mode, dataset_kwargs)
        # use seed/epoch/idx to resume
        self.seed = seed
        self.epoch = epoch
        self.idx = idx
        logger.info("encode dataset")
        self.dataset = encode_dataset(dataset=self.dataset, num_proc=num_proc, encode_function=encode_function)
        if self.seed is not None and self.mode != "traversal":
            self.dataset = self.dataset.shuffle(seed=self.seed + self.epoch)

    async def get_data_item(self, seed: int, **kwargs):
        if self.idx == len(self.dataset):
            self.epoch += 1
            self.idx = 0
            if self.mode != "traversal":
                self.dataset = self.dataset.shuffle(seed=self.seed + self.epoch)
        data = None
        if seed not in self.seed_to_idx:
            self.seed_to_idx[seed] = self.idx
            if self.idx < len(self.dataset):
                data = self.dataset[self.idx]
                self.idx += 1
        else:
            stored_idx = self.seed_to_idx[seed]
            if stored_idx < len(self.dataset):
                data = self.dataset[stored_idx]
        return data



class DeepEyesEnv(Env):
    image_placeholder: str = "<image>"

    def __init__(
        self,
        data_args,
        mode: str = "train",
        seed: Optional[int] = None,
        epoch: Optional[int] = 0,
        idx: Optional[int] = 0,
        max_steps: int = 10,
        acc_weight: float = 0.8,
        format_weight: float = 0.2,
        tool_weight: float = 1.2,
        reward_tokenizer=None,
        reward_proxy=None,
        enable_thinking: bool = False,
        reward_generating_args: Optional[Dict] = None,
        current_env_id: Optional[int] = None,
    ):
        data_args: DataArguments = from_dict(data_class=DataArguments, data=data_args)
        self.mode = mode
        self.visual_toolbox = VisualToolBoxV2()
        self.max_steps = max_steps

        # Reward weights
        self.acc_weight = acc_weight
        self.format_weight = format_weight
        self.tool_weight = tool_weight

        # Reward inference components
        self.reward_tokenizer = reward_tokenizer
        self.reward_proxy = reward_proxy
        self.enable_thinking = enable_thinking
        # Default generation config for reward model if not provided
        self.reward_generating_args = reward_generating_args or {
            "temperature": 0.2,
            "max_new_tokens": 2048,
            "top_p": 0.95,
        }

        # Store current_env_id for src_rank tracking in reward inference
        self.current_env_id = current_env_id if current_env_id is not None else 0

        # Episode tracking
        self.step_count = 0
        self.has_tool_call_failure = False

        # Convert train/val mode to sample/traversal for GlobalDataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = DeepEyesDataset.options(
            name=f"{self.mode}_deepeyes", get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote(
            dataset_name=data_args.file_name,
            split="train",
            dataset_kwargs={"num_proc": data_args.preprocessing_num_workers},
            mode=global_dataset_mode,
            seed=seed,
            epoch=epoch,
            idx=idx,
        )
        self.dataset_manager = GlobalDatasetManager.options(
            name=f"{self.mode}_dataset_manager", get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name="deepeyes", dataset_ref=self.dataset))

    def reset(self, seed=None):
        data: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        self._data_item = data
        first_obs = {"prompt": self._data_item["prompt"], "image": [self._data_item["images"][0]]}
        self.visual_toolbox.reset(first_obs["image"])

        # Reset episode tracking
        self.step_count = 0
        self.has_tool_call_failure = False

        return first_obs, {}

    def step(self, action: str):
        self.step_count += 1

        # Handle control-type actions (EpisodeStopReason)
        # Similar to terminal_native_env.py:281-286
        if isinstance(action, EpisodeStopReason) and action == EpisodeStopReason.MAX_LENGTH:
            # Force termination and compute reward
            logger.info(f"[MAX_LENGTH] Episode terminated due to MAX_LENGTH, step_count={self.step_count}")
            reward, reward_info = self.obtain_outcome_reward("")
            info = {"metrics": {}, "metrics_agg_mode": self.visual_toolbox.metrics_agg_mode}
            if reward_info:
                info.update(reward_info)
            return "", reward, True, True, info

        result, _, done, exe_info = self.visual_toolbox.execute(action)
        info = {"metrics": exe_info, "metrics_agg_mode": self.visual_toolbox.metrics_agg_mode}

        # Track tool call failures: if a tool call was attempted but failed or was invalid
        # success_tool_call is 1 when tool call succeeds, 0 otherwise
        if exe_info.get("tool_call", 0) == 1 and exe_info.get("success_tool_call", 0) == 0:
            self.has_tool_call_failure = True

        # Check if max_steps is reached
        step_limit_reached = self.step_count >= self.max_steps
        truncated = False

        # If step limit is reached, force episode termination
        if step_limit_reached and not done:
            done = True
            truncated = True
            logger.info(f"[MAX_STEPS] Reached maximum steps ({self.max_steps}), truncating episode")

        # Compute reward on the last step (when done=True)
        # Pass the action (final model response) to obtain_outcome_reward
        reward = 0.0
        if done:
            reward, reward_info = self.obtain_outcome_reward(action)
            if reward_info:
                info.update(reward_info)

        return result, reward, done, truncated, info

    def obtain_outcome_reward(self, response: str) -> Tuple[float, Dict]:
        """
        Compute the final reward for the episode using LLM-as-judge.

        This method is called in step() when the episode terminates (done=True).
        It extracts the answer from the model response, validates the format,
        calls the reward model (LLM judge) to evaluate accuracy, and computes
        the final weighted reward.

        Args:
            response: The final model response (action from the last step)

        Returns:
            Tuple[float, Dict]: (final_reward, reward_info)
                - final_reward: weighted combination of acc, format, and tool rewards
                - reward_info: dict with detailed reward breakdown and metadata
        """
        # Extract answer and validate format from the response
        # Following DeepEyesRewardWorker._get_llm_judgment logic
        answer_text, is_format_error = self._extract_answer(response)

        # Get LLM judgment for accuracy if reward proxy is available
        # Following the exact logic from DeepEyesRewardWorker._get_llm_judgment
        acc_reward = 0.0
        llm_response = None

        if self.reward_proxy is not None and self.reward_tokenizer is not None:
            question = self._data_item["question"]
            ground_truth = self._data_item["ground_truth"]

            # yali: 与使用prompt作为question有diff, prompt里包含了system/user, question只包含问题
            judge_prompt_text = get_prompt(answer_text, ground_truth, question)
            judge_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": judge_prompt_text},
            ]

            # Call reward model through proxy
            llm_response = generate_by_proxy(
                messages=judge_messages,
                tokenizer=self.reward_tokenizer,
                proxy=self.reward_proxy,
                enable_thinking=self.enable_thinking,
                generation_config=self.reward_generating_args,
                src_rank=self.current_env_id,
            )

            if llm_response is not None:
                acc_reward = self._extract_score(llm_response)
            else:
                # LLM judgment failed, return -999.0 (invalid sample)
                logger.warning("LLM judgment failed (returned None), marking sample as invalid")
                return -999.0, {
                    "reward_info": {
                        "final_reward": -999.0,
                        "acc_reward": 0.0,
                        "format_reward": 0.0,
                        "tool_reward": 0.0,
                        "llm_judgment_failed": True,
                        "response": response,
                        "answer": answer_text,
                    }
                }

        # Penalize for model trying to predict longer answer to hack llm-as-judge
        if len(answer_text) >= 1000:
            acc_reward = 0.0
            is_format_error = True

        # Compute component rewards
        # tool_reward is based on whether vision tools were used successfully
        # - step_count > 1 means tools were called
        # - acc_reward > 0.5 means the answer is correct
        # - has_tool_call_failure=False means all tool calls were successful
        format_reward = -1.0 if is_format_error else 0.0
        tool_reward = 1.0 if self.step_count > 1 and acc_reward > 0.5 and not self.has_tool_call_failure else 0.0

        # Compute final weighted reward
        final_reward = (
            self.acc_weight * acc_reward +
            self.format_weight * format_reward +
            self.tool_weight * tool_reward
        )

        # Build detailed reward info
        reward_info = {
            "reward_info": {
                "final_reward": final_reward,
                "acc_reward": acc_reward,
                "format_reward": format_reward,
                "tool_reward": tool_reward,
                "is_format_error": is_format_error,
                "step_count": self.step_count,
                "has_tool_call_failure": self.has_tool_call_failure,
                "question": self._data_item.get("question"),
                "ground_truth": self._data_item.get("ground_truth"),
                "response": response,
                "answer": answer_text,
                "llm_response": llm_response,
            }
        }

        # logger.info(json.dumps(reward_info, ensure_ascii=False))
        return final_reward, reward_info

    def _extract_answer(self, predict_str: str) -> Tuple[str, bool]:
        """
        Extract answer from model response and validate format.

        Args:
            predict_str: The model's response string

        Returns:
            Tuple[str, bool]: (answer_text, is_format_error)
        """
        is_format_error = False

        # Check think tags
        count_think_1 = predict_str.count("<think>")
        count_think_2 = predict_str.count("</think>")
        if count_think_1 != count_think_2:
            is_format_error = True

        # Extract content after last </think>
        predict_no_think = predict_str.split("</think>")[-1].strip()

        # Check answer tags
        count_answer_1 = predict_no_think.count("<answer>")
        count_answer_2 = predict_no_think.count("</answer>")
        if count_answer_1 != count_answer_2:
            is_format_error = True

        # Extract answer text
        answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

        return answer_text, is_format_error

    def _extract_score(self, response: str) -> float:
        """
        Extract accuracy score from LLM judge response.

        Args:
            response: The LLM judge's response string

        Returns:
            float: Accuracy reward (1.0 or 0.0)
        """
        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
            if "1" in response:
                return 1.0
            elif "0" in response:
                return 0.0
            else:
                logger.warning(f"[WARNING] Response format error: {response}")
                return 0.0
        else:
            if response == "1":
                return 1.0
            elif response == "0":
                return 0.0
            else:
                logger.warning(f"[WARNING] Response format error: {response}")
                return 0.0

    def add_extra_data(self, data: DataProto, messages: List[Dict]):
        data.non_tensor_batch.update(
            {
                "question": np.array([self._data_item["question"]], dtype=object),
                "ground_truth": np.array([self._data_item["ground_truth"]], dtype=object),
                "message": np.array([messages], dtype=object),
            }
        )
