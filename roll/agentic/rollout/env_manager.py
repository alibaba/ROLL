import copy
import re
import time
import random
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from itertools import zip_longest
from threading import Lock
from typing import Dict, List, Optional, Union, Tuple
from codetiming import Timer

import PIL
import numpy as np
import ray
import torch

from tensordict import TensorDict
from transformers import AutoTokenizer, PreTrainedTokenizer, ProcessorMixin

from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger
from roll.agentic.rollout.env_action_limiter import get_global_limiter

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

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None,
                         use_turn_scores: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    all_scores: list[list[float], 存储每个env每轮的reward
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    NOTE: important! input_ids is left pad
    """
    # TODO: special tokens add to config
    assistant_turn_start_tokens = tokenizer.encode("<|im_start|>assistant\n")
    turn_start_token = assistant_turn_start_tokens[0]
    turn_starts = torch.where(input_ids == turn_start_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)

    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)  # only learns all assistant turns
    non_prompt_mask = (turn_indicators > 2)  # learns everything after system prompt + user prompts

    # turn text: '<|im_start|>assistant\n<answer>Right</answer><|im_end|>'
    # <|im_start|>assistant\n 应该mask掉才对，保留<|im_end|>
    for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
        """
        system, user, assistant, user, assistant, user, assistant
        1,2,3,3,4,5,6
        assistant位于第3,5,7...
        """
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

    # TODO: special tokens add to config
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


class EnvManager:
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: Dict,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue,
                 thread_lock: Lock,
                 processor: Optional[ProcessorMixin] = None,
                 collator: Optional[callable] = None,
                 mode='train'):
        """
        1. 一个EnvManager持有一个env实例: 执行env.reset, env.step, 管理rollout的状态
            group trajectory表达: group内的init state一致，依赖env_config 中的seed来控制, 一个group内env 对应episode的seed一致
            EnvWorker持有多个EnvManager，通过线程的方式实现EnvWorker内部并行
        2. run_rollout_loop, 持续rollout trajectory, 将done的trajectory回传到output_queue
        TODO:
            1. special tokens add to config
            2. ray max_concurrency 描述多线程是否会更好？
        """
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: Dict = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor: ProcessorMixin = processor
        self.collator = collator
        self.env_entry = None
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler
        self.rollout_cache = None
        self.group_seed = None
        self.episode_id = 0
        self.current_step = -1
        self.running = False
        self.use_thread_lock = self.env_config.get("use_thread_lock", True) # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        
        # 设置env step 并发限制
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = None
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        self.env_entry = copy.deepcopy(self.env_config)
        self.env_entry['env'] = REGISTERED_ENVS[self.env_entry['env_class']](self.env_entry['config'])
        self.env_entry['status'] = EnvStatus()

        self.prefix_lookup = None
        self.env_config_lookup = None
        self._init_prefix_lookup()

    def reset(self):
        entry = self.env_entry
        self.rollout_cache = {"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'],
                              "tag": entry['tag'], "penalty": 0, "frames": []}

        seed = self.group_seed + self.episode_id

        with self.thread_lock:
            entry['env'].reset(seed=seed)

        entry['status'] = EnvStatus(seed=seed)
        next_state = self._handle_mm_state(entry['env'].render())

        # update rollout cache
        self.rollout_cache['history'] = self._update_cache_history(self.rollout_cache['history'],
                                                                   next_state=next_state,
                                                                   actions_left=entry['max_actions_per_traj'],
                                                                   num_actions_info=None)
        self.episode_id += 1
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        env_input: Dict = self.get_env_input(llm_output)

        entry = self.env_entry
        actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

        # execute actions in env
        valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])

        acc_reward, turn_info, turn_done, executed_actions = self._execute_actions(entry['env'], valid_actions[:actions_left_before])

        if len(valid_actions) != len(env_input['actions']) or not valid_actions:
            self.rollout_cache["penalty"] += self.worker_config.format_penalty

        status, history = self._log_env_state(entry['status'], self.rollout_cache['history'],
                                              entry['env'].render(), entry['max_actions_per_traj'], executed_actions,
                                              valid_actions, acc_reward, turn_done, turn_info, env_input)
        status.step += 1
        entry['status'] = status

        max_steps_per_traj = entry.get("max_steps_per_traj", entry['max_actions_per_traj'])
        if status.step >= max_steps_per_traj and not turn_done:
            entry['status'].truncated = True
            entry['status'].terminated = True

        self.rollout_cache['history'] = history

        if self.mode == "val":
            frame = entry['env'].render(mode='rgb_array')
            if isinstance(frame, np.ndarray):
                self.rollout_cache['frames'].append(frame)

        return status

    def generate(self, env_output: Dict):
        lm_input: DataProto = self.get_lm_input(env_output, prepare_for_update=False)

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(generation_config["max_new_tokens"],
                                                  max(self.pipeline_config.sequence_length - lm_input.batch['input_ids'].shape[1] - generation_config["max_new_tokens"], 1))
        if generation_config["max_new_tokens"] <= 1:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {lm_input.batch['input_ids'].shape[1]},"
                        f"maybe you should increase the response_length")
            return None

        gen_batch = lm_input.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=(["multi_modal_data"] if "multi_modal_data"
                                   in lm_input.non_tensor_batch else []))
        gen_batch.meta_info["generation_config"] = generation_config
        gen_batch.meta_info['response_callback_fn'] = self.generate_scheduler.report_response.remote
        gen_batch.meta_info["src_rank"] = self.env_config["env_id"]
        lm_output: DataProto = ray.get(self.generate_scheduler.generate_one_request.remote(data=gen_batch))

        if lm_output is not None:
            # 未被abort
            gen_batch.meta_info.pop("generation_config")
            lm_input = lm_input.repeat(repeat_times=generation_config['num_return_sequences'])
            lm_output.union(lm_input)
        return lm_output

    def run_rollout_loop(self, current_step, seed):
        assert not self.running
        self.running = True
        is_sync_training: bool = current_step is not None
        if is_sync_training:
            self.current_step = current_step
        assert self.current_step >= 0
        self.episode_id = 0
        self.group_seed = seed + self.env_entry['group_seed']
        env_output = self.reset()
        start_step = self.current_step

        log_stats = {"generate_time": [], "step_time": [], "current_step": []}

        while self.running:
            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.generate(env_output)
            if lm_output is None:
                continue
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            with Timer(name="step", logger=None) as step_timer:
                status: EnvStatus = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            if status.done:
                self.logger.debug(f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}")
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}

                rollout: DataProto = self.formulate_rollouts()
                ray.get(self.output_queue.put.remote(self.env_entry['group_id'], self.episode_id, start_step, rollout))
                self.rollout_cache = None
                if not self.running or (is_sync_training and self.episode_id >= self.worker_config.max_traj_per_env):
                    self.logger.debug(
                        f"env_id: {self.env_config['env_id']} max_traj_per_env {self.worker_config.max_traj_per_env} reached, stopping rollout loop")
                    break
                env_output = self.reset()
                start_step = self.current_step

    def update_step(self, global_step):
        # assume assignment of primitive type in python is atomic
        self.current_step = global_step

    def stop(self):
        # assume assignment of primitive type in python is atomic
        self.running = False

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
            # TODO: special tokens add to config
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
                    [f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join(
                    [f"{v}" for k, v in env_config_new["action_lookup"].items()])
                # one action per step
                # action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {
                'max_tokens': env_config.get("max_tokens", self.pipeline_config.response_length)}

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

    def get_lm_input(self, env_output, prepare_for_update: bool) -> DataProto:
        """"""
        llm_input_texts, messages_list, llm_input_images = self._format_messages(
            env_output=env_output,
            prepare_for_update=prepare_for_update,
            use_raw_llm_response=False)
        if llm_input_images and any(env_images for env_images in llm_input_images):
            assert self.collator, "requires collator for multi-modal data"
        if self.collator:
            # TODO: collator and image are coupled, fix it
            # assume the collator has these attributes
            prompt_key, image_key, image_flag_key = self.collator.prompt_key, self.collator.image_key, self.collator.image_flag_key
            features = [{
                prompt_key: text,
                image_key: image,
                image_flag_key: True if image else False
            } for text, image in zip(llm_input_texts, llm_input_images)]
            inputs = self.collator(features)
            llm_inputs: DataProto = DataProto.from_single_dict(inputs)
            input_ids, attention_mask, position_ids = llm_inputs.batch[
                "input_ids"], llm_inputs.batch[
                    "attention_mask"], llm_inputs.batch["position_ids"]
        else:
            inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left",
                                    truncation=False)  # do not truncate here. Process later at TODO
            input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            position_ids = attention_mask.cumsum(dim=-1)
            llm_inputs = DataProto()
            llm_inputs.batch = TensorDict({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }, batch_size=input_ids.shape[0])
        llm_inputs.non_tensor_batch.update({
            "env_ids": np.array([env_output["env_id"]], dtype=object),
            "group_ids": np.array([env_output["group_id"]], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
            "tags": np.array([env_output["tag"]], dtype=object),
        })
        return llm_inputs

    def get_env_input(self, lm_output: DataProto) -> Dict:
        if lm_output.batch is not None and 'responses' in lm_output.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_output.batch['responses'],
                skip_special_tokens=True
            )
        else:  # dataproto has textual responses
            responses = lm_output.non_tensor_batch['response_texts']
        responses = ["<think>" + response if self.pipeline_config.enable_think else "<answer>" + response for
                     response in responses]  # The LLM generation does not include <think> tags. Add them back here.

        env_ids = lm_output.non_tensor_batch['env_ids']
        env_id = env_ids[0]
        response = responses[0]
        llm_response, actions = self._parse_response(response)
        env_input = {"env_id": env_id,
                     "llm_raw_response": response,
                     "llm_response": llm_response,
                     "actions": actions,
                     }
        return env_input

    def formulate_rollouts(self):
        """
        1. 每个env的trajectory 应该是一个rollout
        2. 每个rollout 应该是一个List[Dict]
        3. 每个Dict 应该是一个step的信息
        """
        llm_input_texts, messages_list, llm_input_images = self._format_messages(
            env_output=self.rollout_cache,
            prepare_for_update=True,
            use_raw_llm_response=False)
        has_images = False
        if any(env_images for env_images in llm_input_images):
            has_images = True
            assert self.collator, "requires collator for multi-modal data"
        if self.collator:
            # assume the collator has these attributes
            prompt_key, image_key, image_flag_key = self.collator.prompt_key, self.collator.image_key, self.collator.image_flag_key
            features = [{
                prompt_key: text,
                image_key: image,
                image_flag_key: True if image else False
            } for text, image in zip(llm_input_texts, llm_input_images)]
            inputs = self.collator(features)
            llm_inputs: DataProto = DataProto.from_single_dict(inputs)
            input_ids, attention_mask, position_ids = llm_inputs.batch[
                "input_ids"], llm_inputs.batch[
                    "attention_mask"], llm_inputs.batch["position_ids"]
        else:
            inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left",
                                    truncation=False)
            input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            position_ids = attention_mask.cumsum(dim=-1)
            llm_inputs = DataProto()
            llm_inputs.batch = TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=input_ids.shape[0])
        scores = [[i['reward'] for i in self.rollout_cache['history']]]
        episode_scores = [sum(i) for i in scores]
        penalty = self.rollout_cache['penalty']

        non_prompt_mask, score_tensor, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores,
                                                                            use_turn_scores=self.pipeline_config.use_turn_scores)
        non_prompt_mask = torch.logical_and(non_prompt_mask, attention_mask)
        response_mask = torch.logical_and(response_mask, attention_mask)

        response_length = response_mask.sum(dim=-1).float().mean().item()
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        non_prompt_mask = pad_to_length(non_prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        llm_inputs.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "penalty": torch.Tensor([penalty]),
        })
        llm_inputs.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache["env_id"]], dtype=object),
            "group_ids": np.array([self.rollout_cache["group_id"]], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
            "tags": np.array([self.rollout_cache["tag"]], dtype=object),
            "frames": np.array([self.rollout_cache["frames"]], dtype=object),
        })
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
        llm_raw_text_list, _, _ = self._format_messages(env_output=self.rollout_cache, prepare_for_update=True, use_raw_llm_response=True)
        llm_inputs.non_tensor_batch['turn_scores'] = np.array(scores, dtype=object)
        llm_inputs.non_tensor_batch['episode_scores'] = np.array(episode_scores, dtype=object)
        llm_inputs.non_tensor_batch['llm_raw_text_list'] = np.array(llm_raw_text_list, dtype=object)

        entry = self.env_entry
        status = entry['status']
        env_metric = {
            'success': float(status.terminated and (not status.truncated)),
            'num_actions': status.num_actions,
        }
        custom_metric = {}

        for turn in self.rollout_cache['history']:
            for k, v in turn.get('info', {}).items():
                if k == 'success':
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))

        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(self.rollout_cache['history'])

        self.rollout_cache['history'][-1]['metrics'] = custom_metric
        env_metric = {f"env/{entry['tag']}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        self.rollout_cache['metrics'] = env_metric
        llm_inputs.meta_info = {"metrics": env_metric}
        if has_images:
            # TODO: maybe make this field be list of str for specific features
            llm_inputs.meta_info["_broadcast_non_tensor_batch"] = True
        return llm_inputs


    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str):  # text state
            return state
        elif isinstance(state,
                        np.ndarray):  # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results

    def _update_cache_history(self, history: List[Dict], next_state, actions_left,
                              num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None:  # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)

        entry = {}  # append state to history
        if isinstance(next_state, str):  # text state
            entry['state'] = next_state
        else:  # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
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
            if self.max_env_step_concurrent > 0:
                acquire_id = self.env_step_limiter.acquire()
                _, reward, done, info = env.step(action)
                self.env_step_limiter.release(acquire_id)
            else:
                with self.thread_lock:
                    _, reward, done, info = env.step(action)
            acc_reward += reward
            turn_info.update(info)  # NOTE: currently use last info for multi-action
            executed_actions.append(action)
            if done:
                turn_done = True
                break
        return acc_reward, turn_info, turn_done, executed_actions

    def _log_env_state(self, status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward,
                       turn_done, turn_info, env_input) -> Tuple[EnvStatus, List[Dict]]:
        obs = self._handle_mm_state(cur_obs)
        status.num_actions += len(executed_actions)
        status.rewards.append(acc_reward)
        actions_left = max_actions_per_traj - status.num_actions
        if turn_done:
            status.terminated = True
            status.truncated = not turn_info.get('success', False)
        history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
            'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
            'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
        })
        return status, history

    def _format_messages(self, env_output: Dict, prepare_for_update: bool, use_raw_llm_response: bool):
        if 'state' in env_output['history'][-1] and (not use_raw_llm_response and prepare_for_update):
            env_output['history'] = env_output['history'][
                                    :-1]  # when prepare for update, we do not add the state from the n+1 turn to the trajectory

        # TODO: allow window_size, allow env specific system_prompt,
        # allow specific prompt_format
        # TODO: maybe use image placeholder from result of env.step to
        # unify with and without image, while this requires to simulate
        # processor manually
        env_images = [] if "state" in env_output["history"][
            0] and "images" in env_output["history"][0] else None
        # for multi-modal, content is list of dict
        first_user_content = [{
            "type": "text",
            "text": self.prefix_lookup[env_output["env_id"]],
        }] if env_images is not None else self.prefix_lookup[
            env_output["env_id"]]
        messages = [
            {"role": "system", "content": f"You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game."},
            {"role": "user", "content": first_user_content}
        ]

        for idx, content in enumerate(env_output["history"]):
            turn_idx_content = [{
                "type": "text",
                "text": f"\nTurn {idx + 1}:\n"
            }] if env_images is not None else f"\nTurn {idx + 1}:\n"
            # assume the role of messages[-1] be user, this may not be
            # right if reward not in history content
            # TODO: support if messages[-1] is not user
            messages[-1]["content"] += turn_idx_content
            if "state" in content:
                FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.pipeline_config.enable_think else "<answer> [your answer] </answer>"
                LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                if env_images is not None:
                    messages[-1]["content"] += [{
                        "type": "text",
                        "text": "State:\n"
                    }, {
                        "type": "image",
                    }, {
                        "type":
                        "text",
                        "text":
                        (f"\nYou have {content['actions_left']} actions left. "
                         f"Always output: {FORMAT_PROMPT} with no extra text."
                         f"Strictly follow this format, history response that do not follow the format will be set as 'INVALID'. {LENGTH_PROMPT}\n"
                         f"Decide the next action:\n")
                    }]
                    # ex_manager returns list of images for mm
                    env_images.extend(content["images"])
                else:
                    messages[-1]["content"] += (
                        f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. "
                        f"Always output: {FORMAT_PROMPT} with no extra text."
                        f"Strictly follow this format, history response that do not follow the format will be set as 'INVALID'. {LENGTH_PROMPT}\n"
                        f"Decide the next action:\n")
            if "llm_raw_response" in content:
                #       改成actions合理吗？
                messages.append({"role": "assistant",
                                 "content": content["llm_response"] if not use_raw_llm_response else content[
                                     "llm_raw_response"]})
            if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                reward_content = [
                    {
                        "type": "text",
                        "text": f"Reward:\n{content['reward']}\n"
                    }
                ] if env_images is not None else f"Reward:\n{content['reward']}\n"
                messages.append({"role": "user", "content": reward_content})

        # NOTE: this assertion is important for loss mask computation
        assert all(msg["role"] == "assistant" for msg in messages[2::2])

        if self.processor:
            # processor.chat_template might be different with tokenizer
            # can also set tokenizer.chat_template to processor.chat_template
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=(not prepare_for_update),
                tokenize=False)
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=(not prepare_for_update),
                tokenize=False)
        if use_raw_llm_response:
            prompt_messages = messages[:2]
            if self.processor:
                prompt_text = self.processor.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=False,
                    tokenize=False)
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=False,
                    tokenize=False)
            text = text[len(prompt_text):]
        if not prepare_for_update:
            if self.pipeline_config.enable_think:
                text += "<think>"  # force the LLM to think before answering
            else:
                text += "<answer>"  # force the LLM to answer

        # TODO: 应该没有必要，注意处理mask
        # TODO: special tokens add to config
        text = text.replace("<|im_end|>\n", "<|im_end|>")
        return [text], [messages], [env_images]


    def _parse_response(self, response: str) -> List:
        pattern = (
            r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>$"
            if self.pipeline_config.enable_think
            else r"^<answer>(.*?)</answer>$"
        )
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            think_content, action_content, actions = "INVALID", "INVALID", [] # 如何更好的处理invalid response?
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

            actions = [action.strip() for action in action_content.split(self.pipeline_config.action_sep) if action.strip()]
            max_actions = 1

            if len(actions) > max_actions:
                actions = actions[:max_actions]  # Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.pipeline_config.action_sep + " ").join(actions)

        llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.pipeline_config.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions
