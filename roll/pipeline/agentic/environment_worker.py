import copy
import re
from dataclasses import dataclass, field, asdict
from itertools import zip_longest
from threading import Thread
from typing import Dict, List, Optional, Union, Tuple

import PIL
import numpy as np
import ray
import torch
from ray.util.queue import Queue
from tensordict import TensorDict
from transformers import AutoTokenizer

from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.generate_scheduler import OneRequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.functionals import pad_to_length


"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/llm_agent/es_manager.py
"""

@dataclass
class EnvStatus:
    """Status of an environment"""

    truncated: bool = False  # done but not success
    terminated: bool = False  # done and success
    num_actions: int = 0  # current action step (single action)
    rewards: List[float] = field(default_factory=list)  # rewards for each turn
    seed: Optional[int] = None  # what seed is used to reset this environment
    step: int = 0  # current step (single step)

    @property
    def done(self):
        return self.truncated or self.terminated


def get_masks_and_scores(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    all_scores: List[List[float]] = None,
    use_turn_scores: bool = False,
):
    """
    input_ids: shape (bsz, seq_len)
    all_scores: list[list[float], 存储每个env每轮的reward
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    NOTE: important! input_ids is left pad
    """
    assistant_turn_start_tokens = tokenizer.encode("<|im_start|>assistant\n")
    turn_start_token = assistant_turn_start_tokens[0]
    turn_starts = torch.where(input_ids == turn_start_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)

    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)  # only learns all assistant turns
    non_prompt_mask = turn_indicators > 2  # learns everything after system prompt + user prompts

    # turn text: '<|im_start|>assistant\n<answer>Right</answer><|im_end|>'
    # <|im_start|>assistant\n 应该mask掉才对，保留<|im_end|>
    for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
        turn_indicator = idx * 2 + 3  # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
        turn_start_position = (input_ids == turn_start_token) & (turn_indicators == turn_indicator)
        batch_size, seq_len = input_ids.shape
        num_tokens = len(assistant_turn_start_tokens)
        turn_start_indices = turn_start_position.nonzero(as_tuple=True)
        mask_matrix = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
        for batch_idx, start_idx in zip(turn_start_indices[0], turn_start_indices[1]):
            end_idx = start_idx + num_tokens
            if end_idx <= seq_len:
                mask_matrix[batch_idx, start_idx:end_idx] = True
        response_mask[mask_matrix] = False
        if idx == 0:
            non_prompt_mask[mask_matrix] = False

    reward_token = tokenizer.encode("<|im_end|>")[0]
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3  # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)

    return non_prompt_mask, score_tensor, response_mask


def left_pad_2_right(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    non_prompt_mask: torch.Tensor,
    pad_token_id: int,
    score_tensor: torch.Tensor,
):
    """
    Convert left-padded tensors to right-padded tensors.
    """
    batch_size = input_ids.size(0)
    first_one = attention_mask.float().argmax(dim=1)

    for i in range(batch_size):
        shift = first_one[i].item()
        if shift > 0:
            input_ids[i, :-shift] = input_ids[i, shift:].clone()
            input_ids[i, -shift:] = pad_token_id
            attention_mask[i, :-shift] = attention_mask[i, shift:].clone()
            attention_mask[i, -shift:] = False
            response_mask[i, :-shift] = response_mask[i, shift:].clone()
            response_mask[i, -shift:] = False
            non_prompt_mask[i, :-shift] = non_prompt_mask[i, shift:].clone()
            non_prompt_mask[i, -shift:] = False
            score_tensor[i, :-shift] = score_tensor[i, shift:].clone()
            score_tensor[i, -shift:] = 0


class EnvironmentWorker(Worker):
    """
    1. 一个EnvironmentWorker(进程)持有一个env实例: 执行env.reset, env.step, 管理rollout的状态
        group trajectory表达: group内的init state一致，依赖env_config 中的seed来控制, 一个group内env 对应episode的seed一致
        不采用持有envs的原因是，envs需要管理一组env的交互，增加描述的复杂性
    2. 持有infer_cluster ref, 用于async generate
    3. run_rollout_loop, 持续rollout trajectory, 将done的trajectory回传到output_queue

    承担EnvStateManager的history收集功能
    一个group内的env reset进度应该一致

    TODO: env并行方式后续改成进程+线程并行：目的解决一个env占用一个进程对系统资源的开销
          - 一个EnvironmentWorker持有n个EnvStateManager
          - EnvStateManager管理一个env的rollout loop
          - EnvStateManager.run_rollout_loop,运行在n个线程里
    TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    def __init__(self, worker_config: EnvManagerConfig):
        super().__init__(worker_config)
        self.worker_config: EnvManagerConfig = worker_config
        self.env_config: Dict = worker_config.env_configs[self.rank]
        self.env_entry = None
        self.output_queue = None
        self.input_queue = None
        self.infer_worker = None
        self.rollout_cache = None
        self.mode = "train"
        self.group_seed = None
        self.episode_id = 0
        self.process_input_queue_thread = None
        self.running = False
        self.generate_scheduler = None

        self.prefix_lookup = None
        self.env_config_lookup = None
        self.tokenizer = None

    def initialize(self, pipeline_config, infer_worker, input_queue: Queue, output_queue: Queue, mode: str = "train"):
        super().initialize(pipeline_config)
        self.output_queue = output_queue
        self.input_queue = input_queue
        self.infer_worker = infer_worker
        self.rollout_cache = None
        self.mode = mode

        self.env_entry = copy.deepcopy(self.env_config)
        self.env_entry["env"] = REGISTERED_ENVS[self.env_entry["env_class"]](self.env_entry["config"])
        self.env_entry["status"] = EnvStatus()

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)

        self._init_prefix_lookup()
        self.generate_scheduler = OneRequestScheduler.remote(
            infer_worker=self.infer_worker, pipeline_config=self.pipeline_config
        )

    def reset(self):
        entry = self.env_entry
        self.rollout_cache = {
            "env_id": entry["env_id"],
            "history": [],
            "group_id": entry["group_id"],
            "tag": entry["tag"],
            "penalty": 0,
            "frames": [],
        }

        seed = self.group_seed + self.episode_id
        entry["env"].reset(seed=seed)
        entry["status"] = EnvStatus(seed=seed)
        next_state = self._handle_mm_state(entry["env"].render())

        # update rollout cache
        self.rollout_cache["history"] = self._update_cache_history(
            self.rollout_cache["history"],
            next_state=next_state,
            actions_left=entry["max_actions_per_traj"],
            num_actions_info=None,
        )
        self.episode_id += 1
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        env_input: Dict = self.get_env_input(llm_output)

        entry = self.env_entry
        actions_left_before = entry["max_actions_per_traj"] - entry["status"].num_actions

        # execute actions in env
        valid_actions = self._extract_map_valid_actions(entry, env_input["actions"])

        acc_reward, turn_info, turn_done, executed_actions = self._execute_actions(
            entry["env"], valid_actions[:actions_left_before]
        )

        if len(valid_actions) != len(env_input["actions"]) or not valid_actions:
            self.rollout_cache["penalty"] += self.worker_config.format_penalty

        status, history = self._log_env_state(
            entry["status"],
            self.rollout_cache["history"],
            entry["env"].render(),
            entry["max_actions_per_traj"],
            executed_actions,
            valid_actions,
            acc_reward,
            turn_done,
            turn_info,
            env_input,
        )
        status.step += 1
        entry["status"] = status

        max_steps_per_traj = entry.get("max_steps_per_traj", entry["max_actions_per_traj"])
        if status.step >= max_steps_per_traj and not turn_done:
            entry["status"].truncated = True
            entry["status"].terminated = True

        self.rollout_cache["history"] = history

        frame = entry["env"].render(mode="rgb_array")
        if isinstance(frame, np.ndarray):
            self.rollout_cache["frames"].append(frame)

        return status

    def generate(self, env_output: Dict):
        lm_input: DataProto = self.get_lm_input(env_output, prepare_for_update=False)
        lm_input.meta_info = env_output["meta_info"]

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(
            generation_config["max_new_tokens"],
            max(self.pipeline_config.sequence_length - lm_input.batch["input_ids"].shape[1] - 1, 1),
        )
        if generation_config["max_new_tokens"] <= 1:
            self.logger.warning(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {lm_input.batch['input_ids'].shape[1]},"
                f"maybe you should increase the response_length"
            )
            return None

        gen_batch = lm_input.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        gen_batch.meta_info["generation_config"] = generation_config
        gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
        lm_output: DataProto = ray.get(self.generate_scheduler.generate_one_request.remote(data=gen_batch))

        if lm_output is not None:
            # 未被abort
            gen_batch.meta_info.pop("generation_config")
            gen_batch.meta_info.pop("response_callback_fn")
            lm_input = lm_input.repeat(repeat_times=generation_config["num_return_sequences"])
            lm_output.union(lm_input)
        return lm_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def run_rollout_loop(self, data: DataProto):
        """
        1. 每次调用run_rollout_loop,
            会持续的play episode, 直到收到采集完成的command
            需要重置seed, 确保每个group的seed一致
            episode_id 置0
        seed更新逻辑:
            group_seed = seed + group_seed
            episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """

        self.start_input_queue_process()
        self.running = True
        self.episode_id = 0

        self.group_seed = data.meta_info["seed"] + self.env_entry["group_seed"]
        env_output = self.reset()
        env_output["meta_info"] = data.meta_info

        while self.running:
            lm_output: DataProto = self.generate(env_output)

            status = EnvStatus(truncated=True, terminated=True)
            if lm_output is not None:
                status: EnvStatus = self.step(lm_output)

            if status.done and self.running:
                rollout: DataProto = self.formulate_rollouts()
                traj_group_id = f"{self.env_entry['group_id']}_{self.episode_id}_{self.group_seed}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id], dtype=object)
                self.output_queue.put(rollout)
                self.rollout_cache = None
                if self.episode_id >= self.worker_config.max_traj_per_env:
                    self.logger.debug(
                        f"max_traj_per_env {self.worker_config.max_traj_per_env} reached, stopping rollout loop"
                    )
                    break
                self.reset()

        self.process_input_queue_thread.join()

    def get_lm_input(self, env_output, prepare_for_update: bool) -> DataProto:
        """"""
        llm_input_texts, messages_list = self._format_messages(
            env_output=env_output, prepare_for_update=prepare_for_update, use_raw_llm_response=False
        )
        inputs = self.tokenizer(
            llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False
        )
        # (bsz, seq_len), bsz=1
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"]], dtype=object),
            "group_ids": np.array([env_output["group_id"]], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
            "tags": np.array([env_output["tag"]], dtype=object),
        }
        return llm_inputs

    def get_env_input(self, lm_output: DataProto) -> Dict:
        if lm_output.batch is not None and "responses" in lm_output.batch.keys():
            responses = self.tokenizer.batch_decode(lm_output.batch["responses"], skip_special_tokens=True)
        else:  # dataproto has textual responses
            responses = lm_output.non_tensor_batch["response_texts"]
        responses = [
            "<think>" + response if self.pipeline_config.enable_think else "<answer>" + response
            for response in responses
        ]  # The LLM generation does not include <think> tags. Add them back here.

        env_ids = lm_output.non_tensor_batch["env_ids"]
        env_id = env_ids[0]
        response = responses[0]
        llm_response, actions = self._parse_response(response)
        env_input = {
            "env_id": env_id,
            "llm_raw_response": response,
            "llm_response": llm_response,
            "actions": actions,
        }
        return env_input

    def formulate_rollouts(self):
        """
        1. 每个env的trajectory 是一个rollout
        2. 每个rollout 是一个List[Dict]
        3. 每个Dict 是一个step的信息
        """
        llm_input_texts, messages_list = self._format_messages(
            env_output=self.rollout_cache, prepare_for_update=True, use_raw_llm_response=False
        )
        inputs = self.tokenizer(
            llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        scores = [[i["reward"] for i in self.rollout_cache["history"]]]
        episode_scores = [sum(i) for i in scores]
        penalty = self.rollout_cache["penalty"]

        non_prompt_mask, score_tensor, response_mask = get_masks_and_scores(
            input_ids, self.tokenizer, scores, use_turn_scores=self.pipeline_config.use_turn_scores
        )
        non_prompt_mask = torch.logical_and(non_prompt_mask, attention_mask)
        response_mask = torch.logical_and(response_mask, attention_mask)

        left_pad_2_right(
            input_ids, attention_mask, response_mask, non_prompt_mask, self.tokenizer.pad_token_id, score_tensor
        )

        response_length = response_mask.sum(dim=-1).float().mean().item()
        input_ids = pad_to_length(
            input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        non_prompt_mask = pad_to_length(non_prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        position_ids = attention_mask.cumsum(dim=-1)
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "penalty": torch.Tensor([penalty]),
            },
            batch_size=input_ids.shape[0],
        )
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([self.rollout_cache["env_id"]], dtype=object),
            "group_ids": np.array([self.rollout_cache["group_id"]], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
            "tags": np.array([self.rollout_cache["tag"]], dtype=object),
            "frames": np.array([self.rollout_cache["frames"]], dtype=object),
        }
        # pad to response length
        llm_inputs.batch["llm_response_mask"] = response_mask
        llm_inputs.batch["non_prompt_mask"] = non_prompt_mask
        llm_inputs.batch["response_mask"] = non_prompt_mask
        if self.pipeline_config.enable_response_mask:
            # 只使用llm的response mask，不包含环境的state
            llm_inputs.batch["response_mask"] = response_mask
        first_true_indices = non_prompt_mask.int().argmax(dim=1)
        no_true_mask = ~non_prompt_mask.any(dim=1)
        first_true_indices[no_true_mask] = non_prompt_mask.size(1)
        batch_size, seq_len = non_prompt_mask.size()
        arange = torch.arange(seq_len, device=non_prompt_mask.device).unsqueeze(0).expand(batch_size, -1)
        prompt_mask = arange < first_true_indices.unsqueeze(1)
        llm_inputs.batch["prompt_mask"] = prompt_mask
        llm_inputs.batch["scores"] = score_tensor
        # for llm raw response
        llm_raw_text_list, _ = self._format_messages(
            env_output=self.rollout_cache, prepare_for_update=True, use_raw_llm_response=True
        )
        llm_inputs.non_tensor_batch["turn_scores"] = np.array(scores, dtype=object)
        llm_inputs.non_tensor_batch["episode_scores"] = np.array(episode_scores, dtype=object)
        llm_inputs.non_tensor_batch["llm_raw_text_list"] = np.array(llm_raw_text_list, dtype=object)

        entry = self.env_entry
        status = entry["status"]
        env_metric = {
            "success": float(status.terminated and (not status.truncated)),
            "num_actions": status.num_actions,
        }
        custom_metric = {}

        for turn in self.rollout_cache["history"]:
            for k, v in turn.get("info", {}).items():
                if k == "success":
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))

        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(self.rollout_cache["history"])

        self.rollout_cache["history"][-1]["metrics"] = custom_metric
        env_metric = {f"env/{entry['tag']}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        self.rollout_cache["metrics"] = env_metric
        llm_inputs.meta_info = {"metrics": env_metric}
        return llm_inputs

    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment"""
        if isinstance(state, str):  # text state
            return state
        elif isinstance(
            state, np.ndarray
        ):  # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode="RGB") for _state in state]
        return results

    def _update_cache_history(
        self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None
    ):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None:  # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)

        entry = {}  # append state to history
        if isinstance(next_state, str):  # text state
            entry["state"] = next_state
        else:  # multimodal state
            entry["state"] = "<images>" * len(next_state)
            entry["images"] = next_state
        entry["actions_left"] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry["env"].config, "action_lookup", None)
        if action_lookup is None:
            mapped_actions = actions
        else:  # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions

    def _execute_actions(self, env, actions):
        acc_reward, turn_info, turn_done = 0, {}, False
        executed_actions = []
        for action in actions:
            _, reward, done, info = env.step(action)
            acc_reward += reward
            turn_info.update(info)  # NOTE: currently use last info for multi-action
            executed_actions.append(action)
            if done:
                turn_done = True
                break
        return acc_reward, turn_info, turn_done, executed_actions

    def _log_env_state(
        self,
        status,
        history,
        cur_obs,
        max_actions_per_traj,
        executed_actions,
        all_actions,
        acc_reward,
        turn_done,
        turn_info,
        env_input,
    ) -> Tuple[EnvStatus, List[Dict]]:
        obs = self._handle_mm_state(cur_obs)
        status.num_actions += len(executed_actions)
        status.rewards.append(acc_reward)
        actions_left = max_actions_per_traj - status.num_actions
        if turn_done:
            status.terminated = True
            status.truncated = not turn_info.get("success", False)
        history = self._update_cache_history(
            history,
            next_state=obs,
            actions_left=actions_left,
            num_actions_info={
                "actions": executed_actions,
                "reward": acc_reward,
                "info": turn_info,
                "llm_response": env_input["llm_response"],
                "llm_raw_response": env_input["llm_raw_response"],
            },
        )
        return status, history

    def _format_messages(self, env_output: Dict, prepare_for_update: bool, use_raw_llm_response: bool):
        if "state" in env_output["history"][-1] and (not use_raw_llm_response and prepare_for_update):
            env_output["history"] = env_output["history"][
                :-1
            ]  # when prepare for update, we do not add the state from the n+1 turn to the trajectory

        messages = [
            {
                "role": "system",
                "content": f"You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game.",
            },
            {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]},
        ]

        for idx, content in enumerate(env_output["history"]):
            messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
            if "state" in content:
                FORMAT_PROMPT = (
                    "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
                    if self.pipeline_config.enable_think
                    else "<answer> [your answer] </answer>"
                )
                LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                messages[-1]["content"] += (
                    f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. "
                    f"Always output: {FORMAT_PROMPT} with no extra text."
                    f"Strictly follow this format, history response that do not follow the format will be set as 'INVALID'. {LENGTH_PROMPT}\n"
                    f"Decide the next action:\n"
                )
            if "llm_raw_response" in content:
                # yali: using the raw response will cause continuous crashes: https://aliyuque.antfin.com/mdl-team/traning/wmne4oyxg4dozwia
                #       改成actions合理吗？
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            content["llm_response"] if not use_raw_llm_response else content["llm_raw_response"]
                        ),
                    }
                )
            if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})

        # NOTE: this assertion is important for loss mask computation
        assert all(msg["role"] == "assistant" for msg in messages[2::2])
        if use_raw_llm_response:
            messages = messages[2:]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=(not prepare_for_update), tokenize=False
        )
        if not prepare_for_update:
            if self.pipeline_config.enable_think:
                text += "<think>"  # force the LLM to think before answering
            else:
                text += "<answer>"  # force the LLM to answer

        # TODO: 应该没有必要，注意处理mask
        text = text.replace("<|im_end|>\n", "<|im_end|>")
        return [text], [messages]

    def _init_prefix_lookup(self):
        # TODO: 这里并不合理
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.pipeline_config.custom_envs.items():
            if env_tag not in self.worker_config.tags:
                continue
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            env_config_new.update(env_config)
            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
                    [f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()]
                )
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join(
                    [f"{v}" for k, v in env_config_new["action_lookup"].items()]
                )
                # one action per step
                # action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {
                "max_tokens": env_config.get("max_tokens", self.pipeline_config.response_length)
            }

        tags = self.worker_config.tags
        n_groups = self.worker_config.n_groups
        group_size = self.worker_config.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group

        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response(self, response: str) -> List:
        pattern = (
            r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>$"
            if self.pipeline_config.enable_think
            else r"^<answer>(.*?)</answer>$"
        )
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            think_content, action_content, actions = (
                "INVALID",
                "INVALID",
                [],
            )  # do not remove this kind of invalid string
            # yali: this may be cause potential crash
            # llm_response, actions = response, []
        else:
            if self.pipeline_config.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

            for special_token in self.pipeline_config.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

            actions = [
                action.strip() for action in action_content.split(self.pipeline_config.action_sep) if action.strip()
            ]
            max_actions = 1

            if len(actions) > max_actions:
                actions = actions[:max_actions]  # Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.pipeline_config.action_sep + " ").join(actions)

        llm_response = (
            f"<think>{think_content}</think><answer>{action_content}</answer>"
            if self.pipeline_config.enable_think
            else f"<answer>{action_content}</answer>"
        )
        return llm_response, actions

    def start_input_queue_process(self):
        def process_input_queue(input_queue):
            while True:
                command = input_queue.get()
                if command == "stop":
                    self.logger.info(f"{self.worker_name} stopped, episode_id: {self.episode_id}")
                    self.running = False
                    ray.get(self.generate_scheduler.abort_request.remote(DataProto()))
                    break

        self.process_input_queue_thread = Thread(target=process_input_queue, args=(self.input_queue,))
        self.process_input_queue_thread.start()
