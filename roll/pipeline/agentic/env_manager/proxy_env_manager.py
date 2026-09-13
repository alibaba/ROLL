import asyncio
from datetime import datetime
from contextlib import nullcontext
import json
import re
from threading import Lock
from typing import Any, Dict, List, Optional
import time
import uuid

import logging
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import List, TensorDict
from transformers import PreTrainedTokenizer
from fastapi import Request
from fastapi.responses import JSONResponse

from roll.pipeline.agentic.llm_proxy import create_llm_proxy, BaseLLMProxy
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager
from roll.pipeline.agentic.env_manager.message_tracker import MessageTracker
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.distributed.scheduler.router import RouterManager
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger

from roll.pipeline.agentic.agent_runner.base import AgentRunner
from roll.utils.import_utils import safe_import_class

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.entrypoints.openai.protocol import Tool


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("rock").setLevel(logging.WARNING)


class ProxyEnvManager(BaseEnvManager):
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
        super().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RouterManager = generate_scheduler

        # EnvManager states
        self.rollout_cache = None
        self.group_seed = None
        self.episode_id = None
        self.running = False
        self.use_thread_lock = self.env_config.get("use_thread_lock", False) # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()

        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=None
        )
        self.tool_call_parser = self.env_config.config.get("tool_call_parser", "qwen25")
        
        runner_cls_path = self.env_config["agent_runner_cls"]
        runner_cls = safe_import_class(runner_cls_path)
        if runner_cls is None:
            raise ImportError(f"Failed to import agent_runner_cls: {runner_cls_path}")
        base_url = f"http://127.0.0.1:{self.env_config.get('proxy_port', 8000)}"
        self.agent_runner: AgentRunner = runner_cls(
            base_url=base_url,
            env_id=self.env_config["env_id"],
            env_config=self.env_config,
            worker_config=self.worker_config,
        )
        self.reward_granularity: str = self.env_config.config.get("reward_granularity", "pass_rate")  # "pass_rate" | "binary"
        self.history = []
        self.trajectory_mode: str = self.env_config.config.get("trajectory_mode", "traj")  # "step" | "traj"
        self.message_tracker: Optional[MessageTracker] = None
        self.reset_stats()

    def run_rollout_loop(self, data: DataProto):
        """
        Continuously play episodes until data collection is complete.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']

        while True:
            self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config['group_id'], self.env_config['env_id']))
            start_step = self.current_step

            if self.episode_id is None:
                break

            self.history = []
            self.message_tracker = MessageTracker(
                self.tokenizer,
                enable_thinking=self.env_config.config.get("enable_thinking", False),
                enable_fork=self.env_config.config.get("enable_fork", False),
            )
            self.reset_stats()

            seed = self.group_seed + self.episode_id

            episode_result = self.agent_runner.run_job(seed)
            if episode_result.status == "NoData":
                break
            result = episode_result.to_dict()

            self.running = False 
            rollout = self.formulate_rollouts(result)

            ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout, self.env_config['env_id']))
            self.running = True
        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, None, self.env_config['env_id']))

    def reset_stats(self):
        """Reset per-episode statistics before each new episode."""
        self.traj_start_time = time.time()
        self.last_response_finish_time = None
        self.tools = None
        self.log_stats = {
            "step_rt": [],           # total round-trip time per step (request in → response out)
            "pure_infer_time": [],   # pure LLM generation time
            "env_exec_time": [],     # sandbox execution time (last response end → next request start)
            "proxy_overhead": [],    # proxy logic overhead (tokenizer, parser, etc.)
            "response_length": [],
            "tokens_per_second": [],
            "current_step": [],
        }
        self.message_tracker = MessageTracker(
            self.tokenizer,
            enable_thinking=self.env_config.config.get("enable_thinking", False),
            enable_fork=self.env_config.config.get("enable_fork", False),
        )

    async def process_request(self, request: Request):
        """
        Handle HTTP requests from the Harbor agent via ProxyServer (Push mode).

        Delegates inference to _process_request_dict and wraps the result in
        a JSONResponse. Sequence-length errors surface as 400; other errors as 500.

        TODO: TITO, prefix agg
        """
        if not self.running:
            self.logger.warning("Dropped a late request because the episode is finalizing.")
            return JSONResponse(status_code=410, content={"error": "Episode finished"})

        try:
            body = await request.json()
            response_dict = await self._process_request_dict(body)
            return JSONResponse(content=response_dict)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e), "stop_reason": "max_length"})
        except Exception as e:
            self.logger.error(f"Inference process error: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def _process_request_dict(self, body: Dict) -> Dict:
        """
        Core inference handler shared by Push (HTTP) and Pull (ModelService) modes.

        Args:
            body: Parsed JSON dict with messages, tools, tool_choice fields.

        Returns:
            OpenAI-compatible chat completion dict.

        Raises:
            ValueError: If the prompt exceeds sequence_length.
        """
        self.logger.info(f"Received request from Harbor Agent for env_id: {self.env_config['env_id']}")
        step_start_time = time.time()

        if self.last_response_finish_time:
            env_work_time = step_start_time - self.last_response_finish_time
            self.log_stats["env_exec_time"].append(env_work_time)

        messages = body.get('messages', [])
        raw_tools = body.get('tools', [])
        # Use "none" to get raw text output instead of parsed tool_calls
        tool_choice = body.get('tool_choice', 'auto')

        if raw_tools:
            self.tools = raw_tools

        tools_obj = [Tool.model_validate(t) for t in raw_tools] if raw_tools else []

        prompt_ids = self.message_tracker.process_messages(
            messages, tools=self.tools, add_generation_prompt=True,
        )
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        # Huggingface Transformers prefer position_ids to be 0-based.
        position_ids = attention_mask.cumsum(dim=-1) - 1

        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])
        lm_input.meta_info["tools"] = self.tools
        lm_input.meta_info["tool_choice"] = tool_choice

        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            self.logger.warning(
                f"sequence_length={self.pipeline_config.sequence_length} "
                f"input_ids length={input_ids.shape[1]}, maybe you should increase the response_length"
            )
            raise ValueError("Sequence length exceeded")

        max_new_tokens = min(
            self.env_config["max_tokens_per_step"],
            self.worker_config.generating_args.max_new_tokens,
            self.pipeline_config.sequence_length - input_ids.shape[1],
        )
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        generation_config["enable_thinking"] = self.env_config.config.get("enable_thinking", False)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        infer_start = time.time()
        lm_output = await asyncio.to_thread(
            self.llm_proxy.generate,
            messages=messages,
            lm_input=lm_input,
            generation_config=generation_config,
        )
        infer_end = time.time()
        pure_infer_duration = infer_end - infer_start

        if lm_output is None:
            self.logger.error("LLM Proxy returned None (Inference Aborted), retrying...")
            lm_input.meta_info["src_rank"] = self.env_config["env_id"]
            lm_output = await asyncio.to_thread(
                self.llm_proxy.generate,
                messages=messages,
                lm_input=lm_input,
                generation_config=generation_config,
            )

        if lm_output is None:
            raise RuntimeError("LLM Proxy failed after retry — check model name and API key in llm_proxy config")

        response_ids = lm_output.batch['responses'][0].tolist()
        resp_len = len(response_ids)

        total_step_rt = time.time() - step_start_time
        overhead = total_step_rt - pure_infer_duration

        self.log_stats["step_rt"].append(total_step_rt)
        self.log_stats["pure_infer_time"].append(pure_infer_duration)
        self.log_stats["proxy_overhead"].append(overhead)
        self.log_stats["response_length"].append(resp_len)
        self.log_stats["current_step"].append(len(self.history))
        if total_step_rt > 0.01:
            self.log_stats["tokens_per_second"].append(resp_len / total_step_rt)

        infer_logprobs = None
        if "infer_logprobs" in lm_output.batch.keys():
            infer_logprobs = lm_output.batch['infer_logprobs'][0][-resp_len:].tolist()

        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        finish_reason = "stop"
        tool_calls = None

        if tool_choice != "none" and tools_obj:
            # qwen25 parser requires <tool_call>\n{...}\n</tool_call> (with newlines).
            # Some external models emit the no-newline variant; normalize before parsing.
            if "<tool_call>" in text:
                text = re.sub(r"<tool_call>(?!\n)", "<tool_call>\n", text)
                text = re.sub(r"(?<!\n)</tool_call>", "\n</tool_call>", text)
            tool_calls, text, finish_reason = self.process_tool_calls(text, tools_obj, finish_reason)

        message_out = {"role": "assistant", "content": text}
        if tool_calls:
            message_out["tool_calls"] = tool_calls

        # Record assistant response in MessageTracker (unified for both step and traj modes)
        self.message_tracker.record_response(
            msg_pos=len(messages),
            resp_msg=message_out,
            response_ids=response_ids,
            logprobs=infer_logprobs,
        )

        with self.thread_lock:
            self.history.append({
                "prompt_messages": messages,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_message": message_out,
                "tools": self.tools,
                "logprobs": infer_logprobs,
                "finish_reason": finish_reason,
                "timestamp": time.time(),
            })

        self.last_response_finish_time = time.time()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": message_out,
                "finish_reason": finish_reason,
            }],
        }


    def process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: str,
    ) -> tuple[Optional[List[Dict]], str, str]:
        """Parse tool calls from model output using sglang FunctionCallParser."""
        parser = FunctionCallParser(tools, self.tool_call_parser)

        if parser.has_tool_call(text):
            try:
                text, call_info_list = parser.parse_non_stream(text)
                tool_calls = []
                for call_info in call_info_list:
                    tool_id = f"call_{uuid.uuid4().hex[:24]}"
                    args = call_info.parameters
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": call_info.name,
                            "arguments": args if args else "{}",
                        },
                    })
                return tool_calls, text, "tool_calls"
            except Exception as e:
                self.logger.error(f"Tool call parsing error: {e}")
                return None, text, finish_reason

        return None, text, finish_reason


    def _select_episode_score(self, harbor_result: Dict) -> float:
        """Pick the training reward from the runner result based on ``reward_granularity``.

        ``reward_granularity`` names which result field to use as the reward (the runner only
        reports facts; this is the single decision point):
        - "pass_rate": the continuous FAIL_TO_PASS rate, added by the runner when the evaluation
          report.json is available; falls back to the 0/1 ``score`` when it is missing.
        - "binary" (or any value without a matching field): falls back to the 0/1 ``score``,
          which is the binary verifier reward itself.
        """
        return float(harbor_result.get(self.reward_granularity, harbor_result.get("score", 0.0)))

    def formulate_rollouts(self, harbor_result: Dict) -> DataProto:
        """
        Convert Harbor execution results to training data format.
        Uses MessageTracker.get_trajectory_data(mode) to unify step/traj modes.
        """       
        tag = self.env_config.get('tag', 'proxy')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        traj_group_id = f"{tag}_{self.env_config['group_id']}_{self.episode_id}_{self.group_seed}"
        traj_id = f"{traj_group_id}_{self.env_config['env_id']}_{timestamp}"

        episode_score = self._select_episode_score(harbor_result)
        if harbor_result.get("agent_exit_reason"):
            self.logger.warning(f"[Harbor] agent_exit_reason: {harbor_result['agent_exit_reason']}")
        trajectory_samples = self.message_tracker.get_trajectory_data(mode=self.trajectory_mode)

        if not trajectory_samples:
            self.logger.info(f"[PLACEHOLDER_ROLLOUT] Creating for episode_id: {self.episode_id}")
            trajectory_samples = [{
                "token_ids": [self.tokenizer.pad_token_id],
                "response_masks": [0],
                "logprobs": [0.0],
                "messages": [],
            }]
            episode_score = 0.0

        # Step mode: reward only on last step; traj mode: episode_score on each branch
        is_step_mode = self.trajectory_mode == "step"
        step_rewards: List[float] = []
        if is_step_mode:
            step_rewards = [0.0] * (len(trajectory_samples) - 1) + [episode_score]

        samples: List[DataProto] = []
        seq_len = self.pipeline_config.sequence_length
        
        logic_pct = float(harbor_result.get('agent_execution_time_ratio', 0.0))
        env_setup_pct = float(harbor_result.get('env_setup_time_ratio', 0.0))
        agent_setup_pct = float(harbor_result.get('agent_setup_time_ratio', 0.0))
        total_sandbox_duration = float(harbor_result.get('time_total_sec', 0.0))

        for idx, traj_data in enumerate(trajectory_samples):
            token_ids_list = traj_data["token_ids"]
            response_masks_list = traj_data["response_masks"]
            logprobs_list = traj_data["logprobs"]
            messages = traj_data.get("messages", [])

            reward = step_rewards[idx] if is_step_mode else episode_score

            # Build tensors
            input_ids = torch.tensor(token_ids_list, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones(1, len(token_ids_list), dtype=torch.long)
            response_mask = torch.tensor(response_masks_list, dtype=torch.bool).unsqueeze(0)

            first_resp_idx = (
                response_masks_list.index(1) if 1 in response_masks_list else len(response_masks_list)
            )
            prompt_masks = [1] * first_resp_idx + [0] * (len(token_ids_list) - first_resp_idx)
            prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)

            score_tensor = torch.zeros(1, len(token_ids_list), dtype=torch.float)
            score_tensor[0, -1] = reward

            position_ids = attention_mask.cumsum(dim=-1) - 1

            # Padding
            input_ids = pad_to_length(input_ids, length=seq_len, pad_value=self.tokenizer.pad_token_id)
            attention_mask = pad_to_length(attention_mask, length=seq_len, pad_value=0)
            position_ids = pad_to_length(position_ids, length=seq_len, pad_value=0)
            response_mask = pad_to_length(response_mask, length=seq_len, pad_value=0)
            prompt_mask = pad_to_length(prompt_mask, length=seq_len, pad_value=0)
            score_tensor = pad_to_length(score_tensor, length=seq_len, pad_value=0)
            
            
            sample_traj_id = f"{traj_id}_b{idx}" if len(trajectory_samples) > 1 else traj_id
            state_hash = f"step_{idx}"
            trajectory_data_dict = {
                "trajectory_id": sample_traj_id,
                "timestamp": timestamp,
                "num_actions": len(self.history),
                "branch_or_step_idx": idx,
                "num_branches_or_steps": len(trajectory_samples),
                "reward_info": {"episode_reward": episode_score, "step_reward": reward},
                "messages": messages,
                "finish_reason": harbor_result.get('finish_reason', 'unknown'),
            }
            messages_json = json.dumps(messages)

            lm_input = DataProto(
                batch=TensorDict({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "response_mask": response_mask,
                    "prompt_mask": prompt_mask,
                    "scores": score_tensor,
                }, batch_size=input_ids.shape[0]),
                non_tensor_batch={
                    "env_ids": np.array([self.env_config['env_id']], dtype=object),
                    "group_ids": np.array([self.env_config['group_id']], dtype=object),
                    "traj_group_id": np.array([traj_group_id], dtype=object),
                    "traj_id": np.array([sample_traj_id], dtype=object),
                    "tags": np.array([tag], dtype=object),
                    "step_scores": np.array([reward], dtype=object),
                    "episode_scores": np.array([episode_score], dtype=object),
                    "state_hash": np.array([state_hash], dtype=object),
                    "step": np.array([idx], dtype=object),
                    "trajectory_data": np.array([json.dumps(trajectory_data_dict)], dtype=object),
                    "messages": np.array([messages_json], dtype=object),
                    "tools": np.array([json.dumps(self.tools)], dtype=object),
                    "exp_name": np.array([self.pipeline_config.exp_name], dtype=object),
                },
            )

            # Logprobs
            logprobs_tensor = torch.tensor(logprobs_list, dtype=torch.float).unsqueeze(0)
            logprobs_tensor = pad_to_length(logprobs_tensor, length=seq_len, pad_value=0)
            lm_input.batch["infer_logprobs"] = logprobs_tensor[:, 1:]

            samples.append(lm_input)

        batch = DataProto.concat(samples)

        # Metrics
        total_resp = sum(sum(s["response_masks"]) for s in trajectory_samples)
        avg_resp_len = float(total_resp / len(trajectory_samples))
        
        # 性能分位数统计 (P99 Analysis)
        def get_stats(data_list):
            if not data_list: return 0.0, 0.0, 0.0
            return float(np.mean(data_list)), float(np.percentile(data_list, 99)), float(np.max(data_list))
        
        avg_rt, p99_rt, max_rt = get_stats(self.log_stats["step_rt"])
        inf_avg, inf_p99, inf_max = get_stats(self.log_stats["pure_infer_time"])
        env_avg, env_p99, env_max = get_stats(self.log_stats["env_exec_time"])
        ovh_avg, ovh_p99, ovh_max = get_stats(self.log_stats["proxy_overhead"]) 
        
        env_metric: Dict[str, float] = {
            "num_actions": len(self.history),
            "response_length": avg_resp_len,
            "episode_score": episode_score,
            "traj_time_total": round(float(time.time() - self.traj_start_time), 4),
            # --- 宏观利用率 ---
            "efficiency/logic_pct": logic_pct,        # 推理+沙箱执行时间占比
            "efficiency/env_setup_pct": env_setup_pct,  # Harbor基础设施准备时间占比
            "efficiency/agent_setup_pct": agent_setup_pct, # Agent内部依赖安装时间占比
            # --- 微观性能指标 ---
            "latency/pure_inf_avg": round(inf_avg, 4),
            "latency/pure_inf_p99": round(inf_p99, 4),
            "latency/pure_inf_max": round(inf_max, 4),
            "latency/env_exec_avg": round(env_avg, 4),
            "latency/env_exec_p99": round(env_p99, 4),
            "latency/env_exec_max": round(env_max, 4),
            "latency/proxy_ovh_avg": round(ovh_avg, 4),
            "latency/proxy_ovh_p99": round(ovh_p99, 4),
            "latency/proxy_ovh_max": round(ovh_max, 4),
            "latency/step_rt_avg": round(avg_rt, 4),
            "latency/step_rt_p99": round(p99_rt, 4),
            "latency/step_rt_max": round(max_rt, 4),
            "throughput/step_per_sec": round(len(trajectory_samples) / (time.time() - self.traj_start_time), 4) 
        }
        prompt_lengths = [len(s["token_ids"]) - sum(s["response_masks"]) for s in trajectory_samples]
        resp_lengths = [sum(s["response_masks"]) for s in trajectory_samples]
        total_tokens = sum(len(s["token_ids"]) for s in trajectory_samples)
        env_metric["prompt_length"] = float(np.mean(prompt_lengths)) if prompt_lengths else 0
        env_metric["response_length"] = float(np.mean(resp_lengths)) if resp_lengths else 0
        env_metric["total_tokens"] = float(total_tokens)
        env_metric["num_branches_or_steps"] = len(trajectory_samples)
        batch.meta_info = {
            "metrics": {f"env/{tag}/{k}": v for k, v in env_metric.items()}
        }
        batch.meta_info["metrics"]["env/response_length"] = avg_resp_len
        batch.meta_info["COLUMMNS_CONFIG"] = [
            ["trajectory_data", "string"],
            ["messages", "string"],
            ["tools", "string"],
            ["exp_name", "string"],
        ]

        return batch


