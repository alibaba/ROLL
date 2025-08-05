import base64
import copy
from contextlib import nullcontext
from threading import Lock
from typing import Dict, List, Optional

import PIL
import numpy as np
import torch
from transformers import PreTrainedTokenizer, ProcessorMixin

from roll.agentic.env import REGISTERED_ENVS
from roll.agentic.env.base import BaseEnv
from roll.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.agentic.rollout.base_env_manager import RolloutCache, BaseEnvManager
from roll.agentic.rollout.env_action_limiter import get_global_limiter
from roll.agentic.rollout.rollout_scheduler import GroupQueueManager
from roll.agentic.rollout.token_mask_utils import split_by_token, \
    token_ids_to_assistant_mask
from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import get_extra_data_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.utils.constants import GenerateStopReason
from roll.utils.functionals import pad_to_length
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
        self.collator = DataCollatorWithPaddingForMM(
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    answer_key=None,
                    extra_data_provider=get_extra_data_provider(
                        pipeline_config.actor_train.model_args.model_name_or_path,
                        processor=processor)
                )
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
            self.env: BaseEnv = REGISTERED_ENVS[self.env_config['env_class']](self.env_config['config'])

        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = None
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = cfg_template.get("agent_system_template", "You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game.")
        self.user_prompt_format = cfg_template.get("user_prompt_format", "<answer> [your answer] </answer>")

        """
        vl messages user content is List[Dict], like:
        [
                {
                    "type": "text",
                    "text": self.reward_template + self.pre_step_template
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
        self.pre_step_template = cfg_template.get("pre_step_template", "\nTurn {turn_idx}:\nState")
        self.next_step_template = cfg_template.get("next_step_template", "\nYou have {actions_left} actions left. "
                                                "Always output: {format_template} with no extra text. "
                                                "Strictly follow this format."
                                                "Max response length: {max_response_length} words (tokens).\nDecide the next action:\n")
        self.reward_template = cfg_template.get("reward_template", "Reward:\n{reward}\n")
        self.added_text = self.env_config["added_text"]
        if self.env_config["env_id"] == 0:
            self.logger.info(f"agent_system_template: {self.agent_system_template}")
            self.logger.info(f"user_prompt_format: {self.user_prompt_format}")
            self.logger.info(f"pre_step_template: {self.pre_step_template}")
            self.logger.info(f"next_step_template: {self.next_step_template}")
            self.logger.info(f"reward_template: {self.reward_template}")
            self.logger.info(f"added_text: {self.added_text}")

        # TODO: add rewards_scheduler for local ray reward workers
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            available_actions=self.env.get_all_actions()
        )


    def make_decision(self, rollout_cache: RolloutCache):
        messages = self.format_messages(rollout_cache.history)

        lm_input_texts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        lm_input_texts += self.added_text

        images = []
        for message in messages:
            if message["role"] == "user":
                content: List[Dict] = message["content"]
                images.extend([content[i].pop("image_PIL") for i in range(len(content)) if content[i]["type"] == "image"])

        features = [{
            self.collator.prompt_key: lm_input_texts,
            self.collator.image_key: images,
            self.collator.image_flag_key: True
        }]
        inputs = self.collator(features)
        lm_input: DataProto = DataProto.from_single_dict(inputs)

        max_new_tokens = min(self.env_config["max_tokens_per_step"], self.worker_config.generating_args.max_new_tokens)
        generation_config = self.worker_config.generating_args.to_dict()

        generation_config["max_new_tokens"] = min(max_new_tokens,
                                                  max(self.pipeline_config.sequence_length - lm_input.batch['input_ids'].shape[1] - max_new_tokens, 1))
        if generation_config["max_new_tokens"] <= 1:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {lm_input.batch['input_ids'].shape[1]},"
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        lm_output: DataProto = self.llm_proxy.generate(messages=messages,
                                                       lm_input=lm_input,
                                                       generation_config=generation_config)

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        lm_output.non_tensor_batch.update({
            "env_ids": np.array([rollout_cache.env_id], dtype=object),
            "group_ids": np.array([rollout_cache.group_id], dtype=object),
            "messages_list": np.array([messages], dtype=object),
            "tags": np.array([rollout_cache.tag], dtype=object),
        })
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, history: List[Dict]):
        messages = [
            {"role": "system", "content": self.agent_system_template},
        ]

        pre_step_content = ""
        for idx, content in enumerate(history):
            if idx == 0:
                pre_step_content = self.env.config.env_instruction

            assert "state" in content, ("The current EnvManager is specifically tailored for standard RL interaction "
                                        "sequences, following the format of (s, a, r, s, a, r...).")

            pre_step_content += self.pre_step_template.format(turn_idx=idx)
            next_step_content = self.next_step_template.format(actions_left=content["actions_left"],
                                                               format_template=self.user_prompt_format,
                                                               max_response_length=self.env_config["max_tokens_per_step"])
            base64_image = base64.b64encode(content["state"]).decode("utf-8")
            user_content_list_dict = [
                {
                    "type": "text",
                    "text": pre_step_content    # Reward:\n1.0\nTurn 1:\nState:
                },
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{base64_image}",
                    "image_PIL": PIL.Image.fromarray(content["state"], mode='RGB')
                },
                {
                    "type": "text",
                    "text": next_step_content     # You have 3 actions left. Always output: <answer> [your answer] </answer> with no extra text.Strictly follow this format. Max response length: 200 words (tokens).Decide the next action:
                }
            ]
            messages.append({"role": "user", "content": user_content_list_dict})

            if "llm_response" in content:
                messages.append({"role": "assistant", "content": content["llm_response"]})

            pre_step_content = ""
            if "reward" in content:
                pre_step_content = self.reward_template.format(reward=content['reward'])
        return messages

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        if 'state' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])
        last_cache.pop("reward", None)
        history.append(last_cache)

        scores = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(scores)
        penalty = [i['penalty'] for i in self.rollout_cache.history]
        episode_penalty = sum(penalty)

        messages = self.format_messages(history)

        messages_text = self.processor.apply_chat_template(messages)

        images = []
        for message in messages:
            if message["role"] == "user":
                content: List[Dict] = message["content"]
                images.extend([content[i].pop("image_PIL") for i in range(len(content)) if content[i]["type"] == "image"])

        features = [{
            self.collator.prompt_key: messages_text,
            self.collator.image_key: images,
            self.collator.image_flag_key: True
        }]

        inputs = self.collator(features)

        token_ids = inputs.input_ids[0].tolist()
        token_ids_split = split_by_token(token_ids, token_ids[0])
        response_masks_list = token_ids_to_assistant_mask(messages=messages, input_ids_list=token_ids_split, tokenizer=self.tokenizer)
        response_masks = [item for items in response_masks_list for item in items]

        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)

        first_response_idx = response_masks.index(1)
        last_response_idx = len(response_masks) - 1 - response_masks[::-1].index(1)
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][last_response_idx] = episode_score

        input_ids = inputs.input_ids[:, :last_response_idx+1]
        attention_mask = inputs.attention_mask[:, :last_response_idx+1]
        position_ids = inputs.position_ids[:, :, :last_response_idx+1]
        lm_input: DataProto = DataProto.from_single_dict(inputs)
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
            "penalty": torch.Tensor([episode_penalty]),
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        })
        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "messages_list": np.array([messages], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "frames": np.array([self.rollout_cache.frames], dtype=object),
            "turn_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
        })

        env_metric = {
            'success': float(self.rollout_cache.history[-1]['metrics'].get('success', episode_score > 0)),
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
        lm_input.meta_info = {"metrics": env_metric}
        return lm_input

