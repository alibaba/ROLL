from omegaconf import DictConfig

import gem
from contextlib import nullcontext
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.agentic.rollout.base_env_manager import RolloutCache, BaseEnvManager
from roll.agentic.rollout.env_action_limiter import get_global_limiter
from roll.agentic.rollout.rollout_scheduler import GroupQueueManager
from roll.agentic.rollout.token_mask_utils import split_by_token, token_ids_to_assistant_mask, \
    custom_apply_chat_template
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.utils.constants import GenerateStopReason
from roll.utils.functionals import pad_to_length
from roll.utils.hash_utils import compute_object_hash
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field


class StepEnvManager(TrajEnvManager):

    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: DictConfig,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 *args, **kwargs):
        BaseEnvManager().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = 0
        self.current_step = -1
        self.running = False
        self.use_thread_lock = self.env_config.get("use_thread_lock", False) # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        with self.thread_lock:
            self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config['config'])

        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = None
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        self.cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = self.cfg_template["agent_system_template"]
        self.agent_template = self.cfg_template["agent_template"]

        if self.env_config["env_id"] == 0:
            self.logger.info(f"agent_system_template: {self.agent_system_template}")
            self.logger.info(f"agent_template: {self.agent_template}")

        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )

    def reset(self) -> RolloutCache:
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        seed = self.group_seed + self.episode_id

        with self.thread_lock:
            observation, info = self.env.reset(seed=seed)

        self.rollout_cache.history.append({
            "observation": observation,    # env return
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None,     # agent input messages
            **info
        })
        self.episode_id += 1
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        responses = self.tokenizer.batch_decode(
            llm_output.batch['responses'],
            skip_special_tokens=True
        )

        observation, reward, terminated, truncated, info = self.env.step(action=responses[0])
        suffix = info.pop("suffix", None)

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['penalty'] = 0

        metrics = info.get("metrics", {})
        if not metrics.get("action_is_valid", True):
            self.rollout_cache.history[-1]['penalty'] = self.worker_config.format_penalty
        self.rollout_cache.history[-1]['llm_response'] = responses[0]
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None
        })
        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix

        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        memory_history = []
        if "history_length" in self.cfg_template:
            memory_history = rollout_cache.history[-self.cfg_template["history_length"]:-1]
        env_instruction = rollout_cache.history[0]["observation"]

        def get_observation(inner_entry):
            if env_instruction == inner_entry["observation"]:
                obs = inner_entry['suffix']
            else:
                obs = f"{inner_entry['observation']} {inner_entry['suffix']}"
            return obs

        sar_history = []
        for history_step, entry in enumerate(memory_history):
            observation = get_observation(entry)
            sar_history.append(f"(step:{self.rollout_cache.step - len(memory_history) + history_step + 1}, observation: {observation}, response: {entry.get('llm_response')})")

        observation = get_observation(rollout_cache.history[-1])

        render_dict = {"env_instruction": env_instruction, "history": ", ".join(sar_history)}
        if contains_renderable_field(self.agent_template, "step_count"):
            render_dict["step_count"] = self.rollout_cache.step
        if contains_renderable_field(self.agent_template, "history_length"):
            render_dict["history_length"] = len(memory_history)
        if contains_renderable_field(self.agent_template, "current_step"):
            render_dict["current_step"] = self.rollout_cache.step + 1
        if contains_renderable_field(self.agent_template, "current_observation"):
            render_dict["current_observation"] = observation
        if contains_renderable_field(self.agent_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
        messages = [
            {"role": "system", "content": self.agent_system_template},
            {"role": "user", "content": self.agent_template.format(**render_dict)}
        ]
        rollout_cache.history[-1]['messages'] = messages

        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True)
        if len(prompt_ids) >= self.pipeline_config.sequence_length:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {len(prompt_ids)},"
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1)
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])

        max_new_tokens = min(self.env_config["max_tokens_per_step"], self.worker_config.generating_args.max_new_tokens)
        generation_config = self.worker_config.generating_args.to_dict()

        generation_config["max_new_tokens"] = min(max_new_tokens,
                                                  max(self.pipeline_config.sequence_length - lm_input.batch['input_ids'].shape[1] - max_new_tokens, 1))
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        lm_output: DataProto = self.llm_proxy.generate(messages=messages,
                                                       lm_input=lm_input,
                                                       generation_config=generation_config)

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})
        response_ids = lm_output.batch['responses'][0]
        response_ids = response_ids.tolist()
        rollout_cache.history[-1]["prompt_ids"] = prompt_ids
        rollout_cache.history[-1]["response_ids"] = response_ids
        rollout_cache.history[-1]["messages"].append({"role": "assistant", "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)})
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """
        Construct step-wise training samples from the collected trajectory.
        """
        if 'observation' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)

        samples: List[DataProto] = []
        episode_score = sum([i['reward'] for i in self.rollout_cache.history])
        for step, history in enumerate(rollout_cache.history):
            token_ids = history["prompt_ids"] + history["response_ids"]
            response_masks = [0] * len(history["prompt_ids"]) + [1] * len(history["response_ids"])
            input_ids =torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
            response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)

            first_response_idx = response_masks.index(1)
            prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
            prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
            score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
            score_tensor[0][-1] = history['reward']
            position_ids = attention_mask.cumsum(dim=-1)

            input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
            attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
            response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

            samples.append(DataProto(
                batch=TensorDict(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "response_mask": response_mask,
                        "prompt_mask": prompt_mask,
                        "scores": score_tensor,
                    },
                    batch_size=input_ids.shape[0]),
                non_tensor_batch={
                    "episode_scores": np.array([episode_score], dtype=object),
                    "step_scores": np.array([history["reward"]], dtype=object), # step-level reward, return by env
                    "tags": np.array([self.rollout_cache.tag], dtype=object),
                    "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
                    "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
                    "state_hash": np.array([compute_object_hash(history["suffix"])], dtype=object),
                    "step": np.array([step], dtype=object),
                }
            ))

        batch: DataProto = DataProto.concat(samples)

        response_length = batch.batch["response_mask"].sum().float().item()
        metrics = self.rollout_cache.history[-1].get('metrics', {})
        env_metric = {
            'success': float(metrics.get('success', episode_score > 0)),
            'num_actions': rollout_cache.step,
        }
        custom_metric = {}
        for turn in self.rollout_cache.history:
            for k, v in turn.get('metrics', {}).items():
                if k == 'success':
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))

        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(self.rollout_cache.history)

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        batch.meta_info = {"metrics": env_metric}
        return batch