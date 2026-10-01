import json
import re
from typing import List

import numpy as np
import torch
from tensordict import TensorDict
import Levenshtein

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.models.model_providers import default_actor_model_provider
from roll.pipeline.diffusion.rewards.dump_mixin import RewardDumpMixin
from roll.platforms import current_platform


class QwenVLJudgeRewardWorker(RewardDumpMixin, Worker):
    OCR_PROMPT = "Please output only the text content from the image without any additional descriptions or formatting."

    def __init__(self, worker_config: WorkerConfig):
        print("[ROLL-SP-DEBUG] QwenVLJudgeRewardWorker.__init__ start", flush=True)
        super().__init__(worker_config=worker_config)
        print("[ROLL-SP-DEBUG] QwenVLJudgeRewardWorker super().__init__ done", flush=True)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.reward_tokenizer = None
        self.strategy = None
        print("[ROLL-SP-DEBUG] QwenVLJudgeRewardWorker.__init__ done", flush=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        if getattr(self.worker_config, "judge_model_type", None) != "inference":
            raise ValueError("QwenVLJudgeRewardWorker only supports judge_model_type='inference'.")
        if self.worker_config.strategy_args is None:
            raise ValueError("QwenVLJudgeRewardWorker requires strategy_args.")
        if self.worker_config.strategy_args.strategy_name != "vllm":
            raise ValueError("QwenVLJudgeRewardWorker requires strategy_args.strategy_name='vllm'.")

        self.strategy = create_strategy(worker=self, sync_wrapper=True)
        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.reward_tokenizer = self.strategy.tokenizer
        self.strategy.offload_states()

        current_platform.init()

    def _tensor_to_pil_image(self, image_tensor: torch.Tensor):
        return self.tensor_to_pil_image(image_tensor)

    def _build_messages(self):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.OCR_PROMPT}]},
        ]

    def _build_judge_batch(self, messages: List[List[dict]], images: List) -> DataProto:
        texts = [self.reward_tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        tokenized = self.reward_tokenizer(texts, return_tensors="pt", padding=True)
        multi_modal_array = np.empty(len(images), dtype=object)
        for idx, (text_ids, image) in enumerate(zip(tokenized["input_ids"], images)):
            multi_modal_array[idx] = {
                "prompt_token_ids": text_ids.tolist(),
                "multi_modal_data": {"image": [image]},
            }
        return DataProto(
            batch=TensorDict(
                {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                },
                batch_size=tokenized["input_ids"].shape[0],
            ),
            non_tensor_batch={"multi_modal_data": multi_modal_array},
        )

    def _compute_ocr_score(self, judge_text: str, ground_truth: str) -> float:
        gt = re.sub(r"\s+", "", ground_truth).lower()
        text = re.sub(r"\s+", "", judge_text).lower()
        if not gt:
            raise ValueError("ground_truth must be non-empty for OCR reward.")
        if gt in text:
            dist = 0
        else:
            dist = Levenshtein.distance(text, gt)
        dist = min(dist, len(gt))
        return 1 - dist / len(gt)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        if "ground_truth" not in data.non_tensor_batch:
            raise ValueError("QwenVLJudgeRewardWorker requires non_tensor_batch['ground_truth'].")
        if "id" not in data.non_tensor_batch:
            raise ValueError("QwenVLJudgeRewardWorker requires non_tensor_batch['id'].")

        ground_truths = data.non_tensor_batch["ground_truth"].tolist()
        sample_ids = data.non_tensor_batch["id"].tolist()
        # Prefer unique per-sample IDs injected by rollout loop; fall back to prompt-level "id"
        unique_sample_ids = data.non_tensor_batch.get("sample_id")
        if unique_sample_ids is not None:
            unique_sample_ids = unique_sample_ids.tolist() if hasattr(unique_sample_ids, "tolist") else list(unique_sample_ids)
        else:
            unique_sample_ids = sample_ids
        images = [self._tensor_to_pil_image(image_tensor) for image_tensor in data.batch["responses"]]

        messages = [self._build_messages() for _ in images]

        judge_batch = self._build_judge_batch(
            messages=messages,
            images=images,
        )
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.reward_tokenizer.eos_token_id, self.reward_tokenizer.pad_token_id]
        generation_config["pad_token_id"] = self.reward_tokenizer.pad_token_id

        output = self.strategy.generate(batch=judge_batch.to(current_platform.device_type), generation_config=generation_config)

        input_len = judge_batch.batch["input_ids"].shape[1]
        judge_ids = output[:, input_len:]
        judge_texts = self.reward_tokenizer.batch_decode(judge_ids, skip_special_tokens=True)
        scores = [self._compute_ocr_score(text, gt) for text, gt in zip(judge_texts, ground_truths)]
        for sample_id, score, judge_text, ground_truth in zip(sample_ids, scores, judge_texts, ground_truths):
            self.logger.info(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "score": score,
                        "judge_response": judge_text,
                        "ground_truth": ground_truth,
                    },
                    ensure_ascii=False,
                )
            )

        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # --- Dump step output if configured ---
        self.maybe_dump_step_output(
            data=data,
            sample_ids=unique_sample_ids,
            images=images,
            reward_scores=scores,
            extra_columns={
                "ground_truth_response": ground_truths,
                "judge_response": judge_texts,
            },
        )

        token_level_rewards = torch.zeros(
            (len(scores), data.batch["all_timesteps"].shape[1]),
            dtype=torch.float32,
        )
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": scores_tensor,
                "scores": scores_tensor,
            },
            non_tensors={
                "identified_words": judge_texts,
            },
        )
        output.meta_info = {"metrics": {"flowgrpo/reward_mean": scores_tensor.float().mean().item()}}
        return output

