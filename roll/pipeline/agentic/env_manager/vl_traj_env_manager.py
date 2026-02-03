from contextlib import nullcontext
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional, Tuple

import PIL
import gem
import numpy as np
import ray
import torch
from transformers import PreTrainedTokenizer, ProcessorMixin
from codetiming import Timer

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache, BaseEnvManager
from roll.pipeline.agentic.env_manager.token_mask_utils import split_by_token, \
    token_ids_to_assistant_mask
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.utils.constants import EpisodeStopReason, GenerateStopReason, RAY_NAMESPACE
from roll.utils.env_action_limiter import get_global_limiter
from roll.utils.functionals import pad_to_length, aggregate_metrics
from roll.utils.logging import get_logger


class VLTrajEnvManager(TrajEnvManager):
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: Dict,
                 tokenizer: PreTrainedTokenizer,
                 processor: ProcessorMixin,
                 generate_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 extra_data_provider=None,
                 *args, **kwargs):
        """
        """
        BaseEnvManager.__init__(self)
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: Dict = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor: ProcessorMixin = processor
        self.extra_data_provider = extra_data_provider
        # TODO: allow to specify image_token and other processor settings
        self.collator = DataCollatorWithPaddingForMM(
            tokenizer=self.tokenizer,
            processor=self.processor,
            answer_key=None,
            image_flag_key=None,
            video_flag_key=None,
            extra_data_provider=self.extra_data_provider,
        )
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.running = False
        self.use_thread_lock = self.env_config.get("use_thread_lock", False) # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = nullcontext()
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=f"{env_tag}_{self.mode}", max_concurrent_calls=self.max_env_step_concurrent)

        # Initialize reward scheduler and reward proxy BEFORE creating the environment
        # This allows passing reward components through env_config to the environment constructor
        self.reward_scheduler: Optional[RequestScheduler] = None
        self.reward_proxy: Optional[BaseLLMProxy] = None
        self.reward_tokenizer: Optional[PreTrainedTokenizer] = None

        # Create environment kwargs from config (convert OmegaConf to dict to avoid type errors)
        env_kwargs = dict(self.env_config['config'])

        # Try to get reward scheduler from Ray named actor
        if self.pipeline_config.reward:
            self.reward_scheduler = ray.get_actor(
                name=f"RewardScheduler-{pipeline_config.reward.name}",
                namespace=RAY_NAMESPACE
            )
            # Get reward tokenizer
            self.reward_tokenizer = default_tokenizer_provider(
                model_args=pipeline_config.reward.model_args
            )
            # Create reward proxy (without env reference since env doesn't exist yet)
            self.reward_proxy = create_llm_proxy(
                generate_scheduler=self.reward_scheduler,
                llm_proxy_config=pipeline_config.reward.llm_proxy,
                tokenizer=self.reward_tokenizer,
                env=None,
            )
            self.logger.info(f"Initialized reward proxy with scheduler: RewardScheduler-{pipeline_config.reward.name}")

            # Inject reward components into env_kwargs (not OmegaConf config)
            env_kwargs['current_env_id'] = self.env_config["env_id"]
            env_kwargs['reward_tokenizer'] = self.reward_tokenizer
            env_kwargs['reward_proxy'] = self.reward_proxy
            if self.pipeline_config.reward.generating_args:
                env_kwargs['reward_generating_args'] = self.pipeline_config.reward.generating_args.to_dict()

        with self.thread_lock, self.env_step_limiter:
            self.env = gem.make(env_id=self.env_config["env_type"], **env_kwargs)

        cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = cfg_template["agent_system_template"]

        """
        vl messages user content is List[Dict], like:
        [
                {
                    "type": "text",
                    "text":  "{observation}\nTurn {turn_idx}:\nCurrent state is:\n"
                },
                {
                    "type": "image",
                    "image": None
                },
                {
                    "type": "text",
                    "text": self.next_step_template

                }
            ]
        """
        self.pre_step_template = cfg_template["pre_step_template"]
        self.next_step_template = cfg_template["next_step_template"]
        if self.env_config["env_id"] == 0:
            self.logger.info(f"agent_system_template: {self.agent_system_template}")
            self.logger.info(f"pre_step_template: {self.pre_step_template}")
            self.logger.info(f"next_step_template: {self.next_step_template}")

        # Create LLM proxy for policy generation
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )
    
    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step

        log_stats = {"generate_time": [], "step_time": [], "current_step": []}
        self.stop_reason = EpisodeStopReason.FINISH

        while self.running and rollout_cache is not None:

            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                generation_stop_reason = lm_output.meta_info.pop("stop_reason")
                # Convert GenerateStopReason.MAX_LENGTH to EpisodeStopReason.MAX_LENGTH
                # Similar to agent_native_env_manager.py:74-77
                if generation_stop_reason == GenerateStopReason.MAX_LENGTH:
                    self.stop_reason = EpisodeStopReason.MAX_LENGTH
                elif generation_stop_reason == GenerateStopReason.ABORT:
                    self.stop_reason = EpisodeStopReason.ABORT
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            with Timer(name="step", logger=None) as step_timer:
                if generation_stop_reason in [GenerateStopReason.FINISH, GenerateStopReason.MAX_LENGTH]:
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            if self.running and (rollout_cache.terminated or generation_stop_reason == GenerateStopReason.MAX_LENGTH):
                self.logger.debug(f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}")
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}

                rollout: DataProto = self.formulate_rollouts(rollout_cache)
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout, self.env_config['env_id']))

                rollout_cache = self.reset()
                start_step = self.current_step
                self.stop_reason = EpisodeStopReason.FINISH

        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, None, self.env_config['env_id']))

    def step(self, llm_output: DataProto):
        # Similar to agent_native_env_manager.py:133-157
        # If stop_reason is MAX_LENGTH, pass it to env; otherwise decode response
        if llm_output.batch is not None:
            response = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=False)[0]
        else:
            # When MAX_LENGTH, batch may be None, pass stop_reason as action
            response = self.stop_reason if self.stop_reason else ""

        with self.thread_lock, self.env_step_limiter:
            observation, reward, terminated, truncated, info = self.env.step(action=response)
        suffix = info.pop("suffix", None)

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['llm_response'] = response
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
        lm_input, messages = self.format_messages(rollout_cache)
        # cache length of newly appended prompt to help to compute response_mask
        rollout_cache.history[-1]["input_ids_length"] = lm_input.batch["input_ids"].shape[1]
        rollout_cache.history[-1]["prompt_ids_length"] = rollout_cache.history[-1]["input_ids_length"] - (
            (rollout_cache.history[-2]["input_ids_length"] + rollout_cache.history[-2]["response_ids_length"])
            if len(rollout_cache.history) >= 2
            else 0
        )

        input_ids = lm_input.batch["input_ids"]
        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {input_ids.shape[1]},"
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        max_new_tokens = min(self.env_config["max_tokens_per_step"],
                             self.worker_config.generating_args.max_new_tokens,
                             self.pipeline_config.sequence_length-input_ids.shape[1])
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        lm_output: DataProto = self.llm_proxy.generate(messages=messages,
                                                       lm_input=lm_input,
                                                       generation_config=generation_config)

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        # cache length of response_ids to help to compute response_mask
        # eos_token should be taken into account
        rollout_cache.history[-1]["response_ids_length"] = len(lm_output.batch["responses"][0])
        self.logger.debug(
            f"env_id={self.env_config['env_id']}, global_step={self.current_step}, episode_id={self.episode_id}, turn_idx={rollout_cache.step}, "
            f"input_ids_length={rollout_cache.history[-1]['input_ids_length']}, prompt_ids_length={rollout_cache.history[-1]['prompt_ids_length']}, "
            f"response_ids_length={rollout_cache.history[-1]['response_ids_length']}"
        )
        return lm_output

    def format_messages(self, history: RolloutCache) -> Tuple[DataProto, List[Dict]]:
        messages = [{"role": "system", "content": self.agent_system_template}]
        mm_data = None

        for idx, content in enumerate(history.history):

            assert "observation" in content, ("The current EnvManager is specifically tailored for standard RL interaction "
                                        "sequences, following the format of (s, a, r, s, a, r...).")

            pre_step_content = self.pre_step_template.format(turn_idx=idx + 1)
            # cannot use `self.rollout_cache.step==0` which would add env_instruction only once for multi-turns
            if content["actions_left"] == self.env_config.max_steps:
                # add env_instruction in the first step
                pre_step_content = history.history[0].get("env_instruction", "") + pre_step_content
            next_step_content = self.next_step_template.format(actions_left=content["actions_left"],
                                                               max_response_length=self.env_config["max_tokens_per_step"])
            obs = content["observation"]
            obs_content = None
            mm_dict = defaultdict(list)
            # obs might be a str, a image (as ndarray), a dict with prompt/image/video as values,
            if isinstance(obs, str):
                obs_content = obs
            elif isinstance(obs, np.ndarray):
                obs_content = [{"type": "image"}]
                mm_dict = {"image": [PIL.Image.fromarray(obs, mode="RGB")]}
            else :
                assert isinstance(obs, dict), f"observation type {type(obs)} is not supported"
                obs_content = obs.get("prompt", "")
                # str or list of dict, and the dict is item of chat format or user content
                if isinstance(obs_content, list):
                    if "role" in obs_content[0]:
                        if obs_content[0].get("role", None) == "system":
                            messages[0]["content"] = obs_content[0]["content"]
                            obs_content = obs_content[1]["content"]
                        else:
                            obs_content = obs_content[0]["content"]
                mm_dict = dict((k, v) for k, v in obs.items() if k not in ["prompt"])

            # replace image placeholder included in env returned prompt
            def replace_placeholder(text):
                if "image" in mm_dict and getattr(self.env, "image_placeholder", None):
                    text = text.replace(self.env.image_placeholder, self.collator.image_token)
                if "video" in mm_dict and getattr(self.env, "video_placeholder", None):
                    text = text.replace(self.env.video_placeholder, self.collator.video_token)
                return text

            if not isinstance(obs_content, str):
                pre_step_content = [
                    {
                        "type": "text",
                        "text": pre_step_content,  # Reward:\n1.0\nTurn 1:\nState:
                    }
                ]
                next_step_content = [
                    {
                        "type": "text",
                        "text": next_step_content,  # You have 3 actions left. Always output: <answer> [your answer] </answer> with no extra text.Strictly follow this format. Max response length: 200 words (tokens).Decide the next action:
                    }
                ]
                for obs_item in obs_content:
                    if obs_item["type"] == "text":
                        obs_item["text"] = replace_placeholder(obs_item["text"])
            else:
                obs_content = replace_placeholder(obs_content)
            user_content_list_dict = pre_step_content + obs_content + next_step_content
            messages.append({"role": "user", "content": user_content_list_dict})
            if mm_dict:
                mm_data = defaultdict(list) if mm_data is None else mm_data
                for k, v in mm_dict.items():
                    mm_data[k].extend([v] if not isinstance(v, (list, tuple)) else v)

            if "llm_response" in content:
                # eos token is included in response, only need to process once actually
                llm_response = (
                    content["llm_response"][: -len(self.tokenizer.eos_token)]
                    if content["llm_response"].endswith(self.tokenizer.eos_token)
                    else content["llm_response"]
                )
                messages.append({"role": "assistant", "content": llm_response})

        if not messages[0]["content"]:
            messages = messages[1:]
        assert messages, f"empty messages with {history=}"
        add_generation_prompt = False if messages[-1]["role"] == "assistant" else True
        lm_input_texts = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        feature = {
            self.collator.prompt_key: lm_input_texts,
        }
        if mm_data:
            if "image" in mm_data:
                feature[self.collator.image_key] = mm_data["image"]
            if "video" in mm_data:
                feature[self.collator.video_key] = mm_data["video"]

        self.logger.debug(
            f"env_id={self.env_config['env_id']}, global_step={self.current_step}, episode_id={self.episode_id}, turn_idx={idx + 1}, {feature=}"
        )
        if not add_generation_prompt:  # the final multi-turn feature, no need for infer
            self.collator.return_infer_inputs = False
        inputs = self.collator([feature])
        self.collator.return_infer_inputs = True
        lm_input: DataProto = DataProto.from_single_dict(inputs)
        if not add_generation_prompt:
            # NOTE: apply_chat_template would append suffix in response such as "<|im_end|>\n",
            # while generated response often contains "<|im_end|>", and "\n" should not be
            # treated as response
            history.history[-1]["extra_suffix_length"] = lm_input.batch["input_ids"].shape[1] - (
                history.history[-1]["input_ids_length"] + history.history[-1]["response_ids_length"]
            )
            self.logger.debug(
                f"env_id={self.env_config['env_id']}, global_step={self.current_step}, episode_id={self.episode_id}, turn_idx={history.step}, "
                f"final input_ids_shape={lm_input.batch['input_ids'].shape}, last turn input_ids_length={history.history[-1]['input_ids_length']}/"
                f"prompt_ids_length={history.history[-1]['prompt_ids_length']}/response_ids_length={history.history[-1]['response_ids_length']}, "
                f"extra_suffix_length={history.history[-1]['extra_suffix_length']}"
            )

        return lm_input, messages

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        # TODO: check inconsistent tokenization between successive encode-decode operations
        #  can potentially lead to a training crash. check token in token out
        #  the same as TrajEnvManager.

        if 'observation' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)

        lm_input, messages = self.format_messages(rollout_cache)

        # can be used to trigger trajectory reward computation
        if callable(getattr(self.env, "normalize_reward", None)):
            self.env.normalize_reward(messages, rollout_cache, self.tokenizer)

        scores = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(scores)

        input_ids = lm_input.batch["input_ids"]
        attention_mask = lm_input.batch["attention_mask"]
        position_ids = lm_input.batch["position_ids"]

        token_ids = input_ids[0].tolist()
        # TODO: use length in cache to construct response_masks after conner case is fixed
        # response_masks = []
        # for item in rollout_cache.history:
        #     response_masks.extend([0] * item["prompt_ids_length"] + [1] * item["response_ids_length"])
        # response_masks.extend([0] * item["extra_suffix_length"])
        token_ids_split = split_by_token(token_ids, token_ids[0], messages=messages, tokenizer=self.tokenizer)
        response_masks_list = token_ids_to_assistant_mask(
            messages=messages, input_ids_list=token_ids_split, tokenizer=self.tokenizer
        )
        response_masks = [item for items in response_masks_list for item in items]

        assert len(response_masks) == len(token_ids), (
            f"response_masks length must be equal to token_ids length, {len(response_masks)=} != {len(token_ids)=}"
        )
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)

        first_response_idx = response_masks.index(1)
        last_response_idx = len(response_masks) - 1 - response_masks[::-1].index(1)
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][last_response_idx] = episode_score

        input_ids = input_ids[:, :last_response_idx+1]
        attention_mask = attention_mask[:, :last_response_idx+1]
        position_ids = (
            position_ids[:, :, : last_response_idx + 1]
            if position_ids.dim() == 3
            else position_ids[:, : last_response_idx + 1]
        )

        response_length = response_mask.sum(dim=-1).float().mean().item()
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        })
        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "messages_list": np.array([messages], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
        })

        metrics_agg_mode = self.rollout_cache.history[-1].get('metrics_agg_mode', {})
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        env_metric["num_actions"] = rollout_cache.step

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}

        if callable(getattr(self.env, "add_extra_data", None)):
            self.env.add_extra_data(lm_input, messages)

        return lm_input
