from typing import Optional, Union, Dict, List, Any
import asyncio
import json
import re
import torch
import requests
import time
import traceback
import numpy as np
from functools import partial
import uuid
import ray
import tensordict
from tensordict import TensorDict
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider, default_reward_model_provider
from roll.platforms import current_platform
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.utils.context_managers import state_offload_manger
from roll.utils.prompt import *
from roll.datasets.chat_template import get_chat_template
from roll.distributed.scheduler.router import RouterManager


class LLMJudgeRewardWorker(Worker):
    """
    Reward Worker that uses LLM-as-judge to compute rewards.

    Supports three judge_model_type modes:
      - "api": calls an external OpenAI-compatible API
      - "inference": runs a local model via InferenceStrategy (GPU)
      - "cluster": delegates to a shared reward model cluster via Ray RequestScheduler (CPU-only)
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        # LLM judge config
        self.judge_prompt = self.worker_config.judge_prompt if hasattr(self.worker_config, "judge_prompt") else None
        self.judge_prompt = prompt_maps.get(self.judge_prompt, None)
        self.judge_model_type = (
            self.worker_config.judge_model_type if hasattr(self.worker_config, "judge_model_type") else "api"
        )
        self.judge_model_name = (
            self.worker_config.judge_model_name if hasattr(self.worker_config, "judge_model_name") else None
        )
        self.judge_api_url = self.worker_config.judge_api_url if hasattr(self.worker_config, "judge_api_url") else None
        self.judge_api_key = self.worker_config.judge_api_key if hasattr(self.worker_config, "judge_api_key") else None

        # Cluster mode state (populated in _initialize_cluster_mode)
        self.reward_tokenizer = None
        self.chat_template_func = None
        self.reward_scheduler = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.actor_tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)
        if self.judge_model_type == "api":
            self._initialize_api_mode()
        elif self.judge_model_type == "inference":
            self._initialize_inference_mode()
        elif self.judge_model_type == "cluster":
            self._initialize_cluster_mode(pipeline_config)
        else:
            raise ValueError(f"Unsupported judge_model_type: {self.judge_model_type}")

    def _initialize_api_mode(self):
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        print(f"{self.worker_name} initialized with API model")

    def _initialize_inference_mode(self):
        async_strategy = self.worker_config.strategy_args.strategy_name in ["vllm", "sglang"]
        if self.worker_config.strategy_args.strategy_name == "sglang":  # not weight sync, need backup weights
            self.worker_config.strategy_args.strategy_config["enable_weights_cpu_backup"] = True
        if self.worker_config.strategy_args.strategy_name == "vllm":
            self.worker_config.strategy_args.strategy_config["sleep_level"] = 1
        self.strategy = create_strategy(worker=self, sync_wrapper=async_strategy)
        self.strategy.initialize(model_provider=default_reward_model_provider)
        self.tokenizer = self.strategy.tokenizer
        print(f"{self.worker_name} initialized with inference model")
        self.strategy.offload_states()
        current_platform.init()

    def _initialize_cluster_mode(self, pipeline_config):
        if pipeline_config.reward_model is None:
            raise ValueError(
                "judge_model_type='cluster' requires pipeline_config.reward_model to be configured"
            )
        self.reward_tokenizer = default_tokenizer_provider(pipeline_config.reward_model.model_args)
        template_name = pipeline_config.reward_model.data_args.template
        self.chat_template_func = get_chat_template(template_name, self.reward_tokenizer)

        scheduler_name = f"RewardModelScheduler-{pipeline_config.reward_model.name}"
        reward_scheduler = ray.get_actor(scheduler_name, namespace=RAY_NAMESPACE)
        self.reward_scheduler = RouterManager.create_client_sync(reward_scheduler)
        self.logger.info(f"{self.worker_name} initialized, connected to scheduler: {scheduler_name}")

    def _call_api_model(self, messages: Dict, retry_times=3) -> str:
        from openai import OpenAI

        ouput = ""
        if not self.judge_api_url or not self.judge_api_key:
            raise ValueError("API URL and API key must be provided for API model type")
        while retry_times > 0:
            retry_times -= 1
            try:
                client = OpenAI(
                    api_key=self.judge_api_key,
                    base_url=self.judge_api_url,
                )
                completion = client.chat.completions.create(model=self.judge_model_name, messages=messages)
                output = completion.choices[0].message.content
                total_tokens = completion.usage.total_tokens
                prompt_token = completion.usage.prompt_tokens
                completion_token = completion.usage.completion_tokens
                token_info = {
                    "total_tokens": total_tokens,
                    "prompt_token": prompt_token,
                    "completion_token": completion_token,
                }
                print(token_info)
                if output != None and output != "":
                    break
            except Exception as e:
                print(e)
                continue
        self.logger.info(f"judge model api output: {str(output)}")
        return output

    def _run_local_inference(self, messages: Dict) -> str:
        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")

        template_name = self.worker_config.data_args.template
        chat_template_func = get_chat_template(template_name, self.tokenizer)
        text = chat_template_func(messages)

        tokenized = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(current_platform.device_type)
        attention_mask = tokenized["attention_mask"].to(current_platform.device_type)

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        data = DataProto(
            batch=TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=input_ids.shape[0])
        )
        data = data.to(current_platform.device_type)
        data.meta_info = {"micro_batch_size": self.worker_config.infer_batch_size}

        with torch.no_grad():
            output = self.strategy.generate(batch=data, generation_config=generation_config)
            if isinstance(output, torch.Tensor):
                generate_ids = output[:, len(input_ids[0]) :]
            else:
                generate_ids = output.batch["input_ids"][:, len(input_ids[0]) :]

        output = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        self.logger.info(f"judge model inference output: {str(output)}")
        return output.strip()

    def _parse_score(self, response: str) -> float:
        """Parse score from judge response. Supports 'Score: X' format and yes/no."""
        response_lower = response.lower().strip()

        # Try "Score: X" format first
        match = re.search(r"Score:\s*([0-9.]+)", response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Normalize to [0, 1] if score > 1
            if score > 1.0:
                score = score / 10.0
            return min(max(score, 0.0), 1.0)

        # Try yes/no format
        if "yes" in response_lower:
            return 1.0
        if "no" in response_lower:
            return 0.0

        self.logger.warning(f"Could not parse score from judge response: {response[:200]}")
        return 0.0

    def _format_judge_prompt(self, prompt: str, response: str, reference: str = None) -> str:
        if "user\n" in prompt:
            prompt = prompt.split("user\n")[-1].strip()
        if not self.judge_prompt:
            formatted_prompt = (
                f"You are an expert judge evaluating the quality of a response to a given prompt.\n\n"
                f"Prompt: {prompt}\n\n"
                f"Response: {response}\n\n"
                f"Reference: {reference}\n\n"
                f"Please evaluate the response on a scale from 0 to 10.\n"
                f"Consider factors such as correctness, completeness, clarity, and relevance to the prompt.\n"
                f"Your evaluation should be a single number between 0 and 10.\n"
                f"Note output your score in the following format: Score: your score."
            )
        else:
            formatted_prompt = self.judge_prompt.format(question=prompt, response=response, reference=reference)
        messages = [{"role": "user", "content": formatted_prompt}]
        return messages

    def _get_llm_judgment(self, prompt_id: str, prompt: str, response: str, reference: str = None) -> float:
        messages = self._format_judge_prompt(prompt, response, reference)

        if self.judge_model_type == "api":
            llm_response = self._call_api_model(messages)
        elif self.judge_model_type == "inference":
            llm_response = self._run_local_inference(messages)
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

        score = self._parse_score(llm_response)
        info = {
            "prompt_id": prompt_id,
            "score": score,
            "prompt": prompt,
            "response": response,
            "reference": reference,
            "messages": messages,
            "llm_response": llm_response,
        }
        return score, info

    def _tokenize_single(self, messages: List[Dict]) -> DataProto:
        """Tokenize a single judge prompt into a DataProto for RequestScheduler (cluster mode)."""
        text = self.chat_template_func(messages)
        tokenized = self.reward_tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

        data = DataProto(
            batch=TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=input_ids.shape[0],
            )
        )
        return data

    def _compute_rewards_cluster(self, data: DataProto, metrics: Dict) -> DataProto:
        """Compute rewards via the shared reward model cluster (concurrent async requests)."""
        prompts_text = self.actor_tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        responses_text = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        ground_truths = data.non_tensor_batch.get("ground_truth", [None] * len(prompts_text))
        prompt_ids = data.non_tensor_batch.get("id", [str(i) for i in range(len(prompts_text))])

        # Prepare generation config for judge model
        generation_config = self.worker_config.generating_args.to_dict()

        # Format judge prompts, tokenize, and send concurrent async requests
        async def generate_all():
            tasks = []
            for i, (prompt, response, reference) in enumerate(zip(prompts_text, responses_text, ground_truths)):
                messages = self._format_judge_prompt(prompt, response, reference)
                single_data = self._tokenize_single(messages)
                single_data.meta_info = {
                    "src_rank": self.rank_info.rank,
                    "pad_to_seq_len": False,
                    "generation_config": generation_config,
                }
                request_id = f"reward_{self.rank_info.rank}_{uuid.uuid4().hex[:8]}_{i}"

                # Use RouterClient's async method for concurrent processing
                task = self.reward_scheduler.generate_request(
                    req=single_data, request_id=request_id, uid=self.rank_info.rank
                )
                tasks.append(task)

            # Gather all results concurrently
            return await asyncio.gather(*tasks)

        # Run async tasks in sync context
        results = asyncio.run(generate_all())

        # Parse scores from judge responses
        scores = []
        for i, result in enumerate(results):
            if result is None:
                self.logger.warning(f"Sample {prompt_ids[i]}: judge request returned None, scoring 0.0")
                scores.append(0.0)
                continue

            judge_text = self.reward_tokenizer.batch_decode(result.meta_info["output_token_ids"], skip_special_tokens=True)[0]
            score = self._parse_score(judge_text)
            scores.append(score)

            self.logger.info(
                json.dumps(
                    {
                        "prompt_id": prompt_ids[i],
                        "score": score,
                        "judge_response": judge_text[:500],
                    },
                    ensure_ascii=False,
                )
            )

        # Return reward DataProto
        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": scores_tensor,
                "scores": scores_tensor,
            }
        )
        output.meta_info = {"metrics": metrics}
        self.logger.info(f"Computed rewards for {len(scores)} samples via reward model cluster")
        return output

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}

        if self.judge_model_type == "inference" and self.strategy:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/compute_rewards",
                is_offload_states=is_offload_states,
            ):
                return self._compute_rewards_impl(data, metrics)
        elif self.judge_model_type == "cluster":
            return self._compute_rewards_cluster(data, metrics)
        else:
            return self._compute_rewards_impl(data, metrics)

    def _compute_rewards_impl(self, data: DataProto, metrics: Dict):
        prompts_text_list = self.actor_tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        scores = []
        for prompt_id, prompt_txt, response, reference in zip(
            data.non_tensor_batch["id"], prompts_text_list, response_text_list, data.non_tensor_batch["ground_truth"]
        ):
            score, info = self._get_llm_judgment(prompt_id, prompt_txt, response, reference)
            scores.append(score)
            self.logger.info(f"{json.dumps(info, ensure_ascii=False)}")

        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = scores_tensor

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )

        output.meta_info = {"metrics": metrics}
        print(f"Computed rewards for {len(scores)} samples")
        return output
