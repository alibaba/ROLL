"""
Geo3K trajectory manager with incremental token appending.

The initial prompt is encoded once, then each observation and response is
appended without re-encoding the conversation history. This preserves the
generated token IDs used for training.

Inference keeps vLLM's compact multimodal prompt IDs, while training keeps
the processor-expanded IDs so image tokens, MRoPE, and visual tensors stay
aligned.
"""
from __future__ import annotations

import inspect
from typing import Optional

import numpy as np
import torch
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.vl_traj_env_manager import VLTrajEnvManager
from roll.utils.constants import GenerateStopReason


class Geo3kTrajEnvManager(VLTrajEnvManager):
    """VLTrajEnvManager with incremental token appending (avoids re-tokenization drift).

    The inherited formatter encodes the initial prompt. Later turns append
    only new observations and responses, and ``formulate_rollouts`` reuses
    those exact token IDs with the initial visual features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_ids: list[int] = []  # compact IDs used by vLLM
        self._train_token_ids: list[int] = []  # processor-expanded IDs used by training
        self._loss_mask: list[int] = []  # 0=prompt/obs, 1=assistant generated
        self._mm_data: Optional[dict] = None  # multimodal data (saved from first turn)
        self._mm_train_inputs: Optional[np.ndarray] = None

    def make_decision(self, rollout_cache: RolloutCache):
        if rollout_cache.step == 0:
            # Encode the complete prompt only on the initial turn.
            lm_input, messages = self.format_messages(rollout_cache)
            # vLLM uses compact multimodal prompt IDs, while training uses the
            # processor-expanded input IDs.
            self._mm_data = None
            self._mm_train_inputs = lm_input.non_tensor_batch.get("multi_modal_inputs")
            if "multi_modal_data" in lm_input.non_tensor_batch:
                self._mm_data = lm_input.non_tensor_batch["multi_modal_data"][0]
                self._token_ids = list(self._mm_data["prompt_token_ids"])
                if self._mm_train_inputs is None:
                    raise RuntimeError("Initial multimodal prompt is missing processor training inputs")
            else:
                self._token_ids = lm_input.batch["input_ids"][0].tolist()
            self._train_token_ids = lm_input.batch["input_ids"][0].tolist()
            self._loss_mask = [0] * len(self._train_token_ids)
        else:
            obs_ids = self._encode_observation_incremental(rollout_cache)
            self._token_ids.extend(obs_ids)
            self._train_token_ids.extend(obs_ids)
            self._loss_mask.extend([0] * len(obs_ids))
            lm_input = self._build_lm_input()
            messages = None  # not used by PolicyProxy.generate

        # Keep the metadata consumed by the inherited formatter consistent.
        rollout_cache.history[-1]["input_ids_length"] = len(self._train_token_ids)
        rollout_cache.history[-1]["prompt_ids_length"] = (
            len(obs_ids) if rollout_cache.step > 0
            else rollout_cache.history[-1]["input_ids_length"]
        )

        if len(self._train_token_ids) >= self.pipeline_config.sequence_length:
            self.logger.warning(
                f"sequence_length = {self.pipeline_config.sequence_length} "
                f"input_ids length = {len(self._train_token_ids)}, "
                f"maybe you should increase the sequence_length"
            )
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        max_new_tokens = min(
            self.env_config["max_tokens_per_step"],
            self.worker_config.generating_args.max_new_tokens,
            self.pipeline_config.sequence_length - len(self._train_token_ids),
        )
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        lm_output: DataProto = self.llm_proxy.generate(
            messages=messages,
            lm_input=lm_input,
            generation_config=generation_config,
        )

        if lm_output is None:
            self.logger.warning(
                f"env_id={self.env_config['env_id']}, turn_idx={rollout_cache.step}, "
                f"llm_proxy.generate returned None → ABORT"
            )
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH

        response_ids = lm_output.batch["responses"][0].tolist()
        self._token_ids.extend(response_ids)
        self._train_token_ids.extend(response_ids)
        self._loss_mask.extend([1] * len(response_ids))

        rollout_cache.history[-1]["response_ids_length"] = len(response_ids)
        return lm_output

    def _encode_observation_incremental(self, rollout_cache: RolloutCache) -> list[int]:
        """Encode only the new observation (not the full conversation history).

        A temporary system prefix preserves the chat-template separators. The
        prefix tokens are stripped before appending the encoded observation.
        """
        content = rollout_cache.history[-1]
        obs = content["observation"]

        pre_step_content = self.pre_step_template.format(turn_idx=rollout_cache.step + 1)
        next_step_content = self.next_step_template.format(
            actions_left=content["actions_left"],
            max_response_length=self.env_config["max_tokens_per_step"],
        )

        if isinstance(obs, str):
            user_content = pre_step_content + obs + next_step_content
        else:
            user_content = pre_step_content + str(obs) + next_step_content

        user_message = {"role": "user", "content": user_content}

        dummy_messages = [{"role": "system", "content": self.agent_system_template}]
        dummy_ids = self.tokenizer.apply_chat_template(
            dummy_messages, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        full_ids = self.tokenizer.apply_chat_template(
            dummy_messages + [user_message], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        obs_ids = full_ids[len(dummy_ids):]

        return list(obs_ids)

    def _build_lm_input(self) -> DataProto:
        """Build DataProto from accumulated token_ids for generate."""
        token_ids = self._token_ids
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        lm_input = DataProto.from_dict(batch)

        if self._mm_data is not None:
            self._mm_data["prompt_token_ids"] = list(token_ids)
            lm_input.non_tensor_batch["multi_modal_data"] = np.array([self._mm_data], dtype=object)

        return lm_input

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """Build a multimodal training sample from the generated token stream.

        Generated token IDs and loss masks remain authoritative. Processor
        outputs are retained only for visual features used to recompute MRoPE.
        """
        if 'observation' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)

        # Only messages are used here; the re-tokenized batch is intentionally
        # discarded because decode -> encode can change generated token IDs.
        _, messages = self.format_messages(rollout_cache)

        if callable(getattr(self.env, "normalize_reward", None)):
            self.env.normalize_reward(messages, rollout_cache, self.tokenizer)

        token_ids = self._train_token_ids
        loss_mask = self._loss_mask
        if len(token_ids) != len(loss_mask):
            raise RuntimeError(
                f"Training token/mask length mismatch: {len(token_ids)} != {len(loss_mask)}"
            )

        scores = [i['reward'] for i in rollout_cache.history]
        episode_score = sum(scores)

        # response_mask: 1 for assistant-generated tokens, 0 for prompt/obs
        response_mask = torch.tensor([loss_mask], dtype=torch.bool)
        # prompt_mask: 1 for prompt (before first response), 0 for rest
        first_response_idx = loss_mask.index(1) if 1 in loss_mask else len(loss_mask)
        prompt_masks = [1] * first_response_idx + [0] * (len(loss_mask) - first_response_idx)
        prompt_mask = torch.tensor([prompt_masks], dtype=torch.bool)
        # scores: episode_score placed at last response token
        last_response_idx = len(loss_mask) - 1 - loss_mask[::-1].index(1) if 1 in loss_mask else len(loss_mask) - 1
        score_tensor = torch.tensor([0] * len(loss_mask), dtype=torch.float).unsqueeze(0)
        score_tensor[0][last_response_idx] = episode_score

        # Re-encoding decoded responses can change token IDs and invalidate the
        # masks and rollout log probabilities.
        input_ids = torch.tensor([token_ids[:last_response_idx + 1]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        # Recompute MRoPE from the exact training tokens and retained visual inputs.
        if self.extra_data_provider is None:
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        else:
            mm_inputs = self._mm_train_inputs[0] if self._mm_train_inputs is not None else {}
            provider_params = inspect.signature(self.extra_data_provider).parameters
            provider_kwargs = {
                key: mm_inputs[key] for key in provider_params if key in mm_inputs
            }
            provider_kwargs.update(input_ids=input_ids, attention_mask=attention_mask)
            position_ids = self.extra_data_provider(**provider_kwargs)["position_ids"]

        # Align masks and rewards with tokens through the final response.
        response_mask = response_mask[:, :last_response_idx + 1]
        prompt_mask = prompt_mask[:, :last_response_idx + 1]
        score_tensor = score_tensor[:, :last_response_idx + 1]

        response_length = response_mask.sum(dim=-1).float().mean().item()

        from roll.utils.functionals import pad_to_length
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        lm_input = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        })
        if self._mm_train_inputs is not None:
            lm_input.non_tensor_batch["multi_modal_inputs"] = self._mm_train_inputs

        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "messages_list": np.array([messages], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
        })

        from roll.utils.functionals import aggregate_metrics
        metrics_agg_mode = rollout_cache.history[-1].get('metrics_agg_mode', {})
        history_metrics = [item.get("metrics", {}) for item in rollout_cache.history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        env_metric["num_actions"] = rollout_cache.step
        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}

        if callable(getattr(self.env, "add_extra_data", None)):
            self.env.add_extra_data(lm_input, messages)

        return lm_input
