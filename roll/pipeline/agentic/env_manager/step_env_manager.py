from typing import List

import numpy as np
import torch
from tensordict import TensorDict

from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.utils.functionals import pad_to_length, aggregate_metrics
from roll.utils.hash_utils import compute_object_hash
from roll.utils.str_utils import contains_renderable_field


class StepEnvManager(TrajEnvManager):
    """
    Used for GiGPO like format.
    You can extend your format_messages as needed.
    """

    def format_messages(self, rollout_cache: RolloutCache) -> DataProto:
        current_cache = rollout_cache.history[-1]
        memory_history = []
        if "history_length" in self.cfg_template:
            memory_history = rollout_cache.history[-self.cfg_template["history_length"]:-1]
        env_instruction = rollout_cache.history[0]["env_instruction"]
        sar_history = []
        for history_step, entry in enumerate(memory_history):
            action = entry.get('action_content', entry.get('action_content', entry.get('llm_response')))
            action_is_valid = entry['metrics'].get("action_is_valid", True)
            if not action_is_valid:
                action += "(IMPORTANT TIPS: this action is not valid, your new response *must* strictly adhere to the format according to env instructions.)"
                sar_history.append(f"(step: {self.rollout_cache.step - len(memory_history) + history_step + 1}, observation: {entry['observation']}, action: {action}, reward: {entry['reward']})")

        current_observation = current_cache["observation"]
        render_dict = {"env_instruction": env_instruction, "history": ", ".join(sar_history)}
        if contains_renderable_field(self.agent_template, "step_count"):
            render_dict["step_count"] = self.rollout_cache.step
        if contains_renderable_field(self.agent_template, "history_length"):
            render_dict["history_length"] = len(memory_history)
        if contains_renderable_field(self.agent_template, "current_step"):
            render_dict["current_step"] = self.rollout_cache.step + 1
        if contains_renderable_field(self.agent_template, "current_observation"):
            render_dict["current_observation"] = current_observation
        if contains_renderable_field(self.agent_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
        messages = []
        if self.agent_system_template is not None:
            messages.append({"role": "system", "content": self.agent_system_template})
        messages.append({"role": "user", "content": self.agent_template.format(**render_dict)})
        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1)
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])
        current_cache["prompt_ids"] = prompt_ids
        current_cache['state_hash'] = compute_object_hash(current_observation)
        current_cache['messages'] = messages
        return lm_input

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
            infer_logprobs = []
            if "infer_logprobs" in history:
                infer_logprobs = [0] * len(history["prompt_ids"]) + history["infer_logprobs"]

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
            lm_input = DataProto(
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
                    "state_hash": np.array([history['state_hash']], dtype=object),
                    "step": np.array([step], dtype=object),
                }
            )
            if len(infer_logprobs):
                infer_logprobs = torch.tensor(infer_logprobs, dtype=torch.float).unsqueeze(0)
                infer_logprobs = pad_to_length(infer_logprobs, length=self.pipeline_config.sequence_length, pad_value=0)
                lm_input.batch["infer_logprobs"] = infer_logprobs[:, 1:]

            samples.append(lm_input)
        batch: DataProto = DataProto.concat(samples)

        response_length = batch.batch["response_mask"].float().sum(-1).mean().item()
        metrics_agg_mode = self.rollout_cache.history[-1].get('metrics_agg_mode', {})
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        env_metric["num_actions"] = rollout_cache.step

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        batch.meta_info = {"metrics": env_metric}
        return batch