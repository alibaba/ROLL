import copy
import json
import time
from datetime import datetime
from typing import List, Union, Dict, Optional

import numpy as np
import ray
import torch
from codetiming import Timer
from tensordict import TensorDict

from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.token_mask_utils import convert_list_content_str
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.utils.constants import GenerateStopReason, EpisodeStopReason
from roll.utils.functionals import pad_to_length, aggregate_metrics
from roll.utils.hash_utils import compute_object_hash


class AgentNativeStepEnvManager(TrajEnvManager):
    """
    Used for native like format.
    You can extend your format_messages as needed.
    For swe/tb native env
    # TODO: 增加业务指标，性能/error/timeout
    """
    log_stats: Dict
    failure_mode: str
    env_reset_failed: bool
    stop_reason: EpisodeStopReason
    tools: List[Dict]
    traj_start_time: float

    def run_rollout_loop(self, data: DataProto):
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        with Timer(name="reset", logger=None) as reset_timer:
            rollout_cache: RolloutCache = self.reset()
        self.log_stats["reset_time"] = round(reset_timer.last, 4)
        start_step = self.current_step
        max_reset_retries = 0
        while self.running and rollout_cache is not None:

            if self.env_reset_failed:
                max_reset_retries += 1
                self.logger.error(f"[ROLLOUT_LOOP] Failed! - due to sandbox initialization failure...")
                rollout: DataProto = self.create_placeholder_rollout(self.episode_id)
                rollout.meta_info["drop_flag"] = True

                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout, self.env_config['env_id']))
                self.env.close()
                if max_reset_retries > 3:
                    backoff_time = min(3600, 10 * max_reset_retries)
                    self.logger.warning(f"[ROLLOUT_LOOP] Avoidance mode - Backing off for {backoff_time}s (retry #{max_reset_retries})")
                    time.sleep(backoff_time)
                else:
                    time.sleep(10)
                with Timer(name="reset", logger=None) as reset_timer:
                    rollout_cache = self.reset()
                self.log_stats["reset_time"] = round(reset_timer.last, 4)
                start_step = self.current_step
                continue

            max_reset_retries = 0
            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
                if stop_reason == GenerateStopReason.MAX_LENGTH:
                    self.stop_reason = EpisodeStopReason.MAX_LENGTH
                elif stop_reason == GenerateStopReason.ABORT:
                    self.stop_reason = EpisodeStopReason.ABORT
            self.log_stats["current_step"].append(self.current_step)
            self.log_stats["generate_time"].append(round(generate_timer.last))

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason in [GenerateStopReason.FINISH, GenerateStopReason.MAX_LENGTH]:
                    rollout_cache: RolloutCache = self.step(lm_output)
            self.log_stats["step_time"].append(round(step_timer.last, 4))

            if self.running and rollout_cache.terminated:
                rollout: DataProto = self.formulate_rollouts(rollout_cache)
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout, self.env_config['env_id']))

                rollout_cache = self.reset()
                start_step = self.current_step

        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, None, self.env_config['env_id']))

    def reset(self) -> Optional[RolloutCache]:
        self.log_stats = {"generate_time": [], "step_time": [], "current_step": [], "reset_time": 0.0, "response_length": [], "tokens_per_second": []}
        self.stop_reason = EpisodeStopReason.FINISH
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(
            self.env_config['group_id'],
            self.env_config['env_id']
        ))
        if self.episode_id is None:
            assert not self.running
            return None

        seed = self.group_seed + self.episode_id
        self.traj_start_time = time.time()
        observation, info = self.env.reset(seed=seed)
        if observation is None:
            return None

        if self.env.env_reset_failed:
            self.env_reset_failed = True
            self.logger.error(f"[ENV_RESET] Failed! - Environment reset failed, observation: {json.dumps(observation, ensure_ascii=False)}, env_reset_failed: {self.env.env_reset_failed}")
            self.failure_mode = info.get("failure_mode", "Sandbox Initialization Failed")
            self.stop_reason = EpisodeStopReason.ENV_RESET_FAILED
        else:
            self.env_reset_failed = False

        self.tools = info.get("tools", [])
        self.rollout_cache.history.append({
            "observation": copy.deepcopy(observation),
            "messages": None,     # agent input messages
            **info,
        })
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        if llm_output.batch is not None:
            response = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=False)[0]
        else:
            response = self.stop_reason
        observation, reward, terminated, truncated, info = self.env.step(action=response)

        self.rollout_cache.step += 1

        # terminated 完全由swe|tb env决定
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.stop_reason = EpisodeStopReason.MAX_STEPS
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['llm_response'] = response
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append({
            "observation": copy.deepcopy(observation),
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None
        })
        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        lm_input = self.format_messages(rollout_cache)
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

        content = self.rollout_cache.history[-1]
        input_messages = content['observation']

        lm_output: DataProto = self.llm_proxy.generate(messages=input_messages,
                                                       lm_input=lm_input,
                                                       generation_config=generation_config)

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output.batch['responses'][0]
        response_ids = response_ids.tolist()

        if "infer_logprobs" in lm_output.batch.keys():
            infer_logprobs = lm_output.batch['infer_logprobs'][0][-len(response_ids):]
            content["infer_logprobs"] = infer_logprobs.tolist()

        content["response_ids"] = response_ids
        content["messages"].append({"role": "assistant", "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)})
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, rollout_cache: RolloutCache) -> DataProto:
        current_cache = rollout_cache.history[-1]

        messages: List[Dict] = current_cache["observation"]

        prompt_ids = self.tokenizer.apply_chat_template(convert_list_content_str(messages, parse_tool_call_parameter_to_dict=self.pipeline_config.parse_tool_call_parameter_to_dict),
                                                        tools=self.tools,
                                                        tokenize=True, add_generation_prompt=True, enable_thinking=False)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        # Huggingface Transformers prefer position_ids to be 0-based.
        # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
        # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
        # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
        position_ids = attention_mask.cumsum(dim=-1) - 1
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])

        current_cache["prompt_ids"] = prompt_ids
        current_cache['state_hash'] = compute_object_hash(messages)
        current_cache['messages'] = messages
        return lm_input

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """
        Construct step-wise training samples from the collected trajectory.
        TODO: 相同前序合并优化
              样本构造方法：
                - 按messages构造response_id
                - 按response_id构造，纯step_wise用
        """
        last_observation = []
        if 'observation' in rollout_cache.history[-1]:
            last_observation = rollout_cache.history[-1]['observation']
            rollout_cache.history.pop(-1)

        samples: List[DataProto] = []
        step_rewards = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(step_rewards)

        # Initialize lists for step length statistics
        step_prompt_length_list = []
        step_response_length_list = []

        all_messages: List[List[Dict]] = [] # 可能包含多条轨迹，相同前序的为一条messages
        messages = None
        for step, history in enumerate(rollout_cache.history):
            if "response_ids" not in history:
                break

            # Collect step length statistics
            step_prompt_length_list.append(len(history["prompt_ids"]))
            step_response_length_list.append(len(history["response_ids"]))

            token_ids = history["prompt_ids"] + history["response_ids"]
            response_masks = [0] * len(history["prompt_ids"]) + [1] * len(history["response_ids"])
            input_ids =torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
            response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
            infer_logprobs = []
            if "infer_logprobs" in history:
                infer_logprobs = [0] * len(history["prompt_ids"]) + history["infer_logprobs"]

            generate_time = self.log_stats["generate_time"][len(self.log_stats["response_length"])]
            self.log_stats["response_length"].append(len(history["response_ids"]))
            if generate_time > 0.01:
                tokens_per_second = len(history["response_ids"]) / generate_time
                self.log_stats["tokens_per_second"].append(tokens_per_second)
            else:
                self.log_stats["tokens_per_second"].append(0.0)

            first_response_idx = response_masks.index(1)
            prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
            prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
            score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
            score_tensor[0][-1] = history['reward']
            # Huggingface Transformers prefer position_ids to be 0-based.
            # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
            # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
            # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
            position_ids = attention_mask.cumsum(dim=-1) - 1

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
                    "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
                    "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
                    "tags": np.array([self.rollout_cache.tag], dtype=object),
                    "step_scores": np.array([history["reward"]], dtype=object), # step-level reward, return by env
                    "episode_scores": np.array([episode_score], dtype=object),
                    "state_hash": np.array([history['state_hash']], dtype=object),
                    "step": np.array([step], dtype=object),
                    "trajectory_data": np.array([None], dtype=object),
                    "messages": np.array([None], dtype=object),
                    "tools": np.array([None], dtype=object),
                    "exp_name": np.array([self.pipeline_config.exp_name], dtype=object),
                }
            )
            if len(infer_logprobs):
                infer_logprobs = torch.tensor(infer_logprobs, dtype=torch.float).unsqueeze(0)
                infer_logprobs = pad_to_length(infer_logprobs, length=self.pipeline_config.sequence_length, pad_value=0)
                lm_input.batch["infer_logprobs"] = infer_logprobs[:, 1:]

            samples.append(lm_input)
            messages = history["messages"]

        # TODO: 需要更细致的处理
        #       可选的方式是，将content + tool_use dict 替换回response
        all_messages.append(messages)
        batch: DataProto = DataProto.concat(samples)

        response_length = batch.batch["response_mask"].float().sum(-1).mean().item()
        metrics_agg_mode = self.rollout_cache.history[-1].get('metrics_agg_mode', {})
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        env_metric["num_actions"] = rollout_cache.step
        env_metric["env_timeout"] = getattr(self.env, "env_timeout", False)
        timing_metric = {
            "traj_time_env_total": round(float(time.time() - self.traj_start_time), 4),
            "traj_time_reset": round(float(self.log_stats["reset_time"]), 4),
            "traj_time_step": round(float(np.mean(self.log_stats["step_time"])), 4),
            "traj_time_step_min": round(float(np.min(self.log_stats["step_time"])), 4),
            "traj_time_step_max": round(float(np.max(self.log_stats["step_time"])), 4),
            "traj_time_generate": round(float(np.mean(self.log_stats["generate_time"])), 4),
            "traj_time_generate_min": round(float(np.min(self.log_stats["generate_time"])), 4),
            "traj_time_generate_max": round(float(np.max(self.log_stats["generate_time"])), 4),
            "traj_time_generate_sum": round(float(np.sum(self.log_stats["generate_time"])), 4),
            "traj_time_response_length": round(float(np.mean(self.log_stats["response_length"])), 4),
            "traj_time_response_length_min": round(float(np.min(self.log_stats["response_length"])), 4),
            "traj_time_response_length_max": round(float(np.max(self.log_stats["response_length"])), 4),
            "traj_time_tokens_per_second": round(float(np.mean(self.log_stats["tokens_per_second"])), 4),
            "traj_time_tokens_per_second_min": round(float(np.min(self.log_stats["tokens_per_second"])), 4),
            "traj_time_tokens_per_second_max": round(float(np.max(self.log_stats["tokens_per_second"])), 4),
        }
        length_metric = {
            "response_length": float(response_length),
            "step_prompt_length": round(float(np.mean(step_prompt_length_list)), 2),
            "step_prompt_length_min": round(float(np.min(step_prompt_length_list)), 2),
            "step_prompt_length_max": round(float(np.max(step_prompt_length_list)), 2),
            "step_response_length": round(float(np.mean(step_response_length_list)), 2),
            "step_response_length_min": round(float(np.min(step_response_length_list)), 2),
            "step_response_length_max": round(float(np.max(step_response_length_list)), 2),
        }

        env_metric.update(timing_metric)
        env_metric.update(length_metric)

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        batch.meta_info = {"metrics": env_metric}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        start_step = self.log_stats["current_step"][0]
        end_step = self.log_stats["current_step"][-1]
        last_step_info = rollout_cache.history[-1]
        failure_mode = last_step_info.get("failure_mode", "")
        traj_id = f"{rollout_cache.tag}_{start_step}_{end_step}_{rollout_cache.group_id}_{rollout_cache.env_id}_{self.episode_id}_{self.group_seed}_{timestamp}"
        trajectory_data = {
            "trajectory_id": traj_id,
            "timestamp": timestamp,
            "current_step": self.current_step,
            "env_info":{
                "env_id": rollout_cache.env_id,
                "group_id": rollout_cache.group_id,
                "tag": rollout_cache.tag,
                "seed": self.group_seed,
                "episode_id": self.episode_id,
                "max_steps": self.env_config.max_steps,
                "mode": self.mode,
                "sequence_length": self.pipeline_config.sequence_length,
                **self.env.env_info
            },
            "timing_info": {
                "traj_save_time": datetime.now().isoformat(),
                **timing_metric
            },
            "length_info": {
                "trajectory_length": rollout_cache.step,
                "num_actions": rollout_cache.step,
                "terminated": rollout_cache.terminated,
                "truncated": rollout_cache.truncated,
                **length_metric
            },
            "reward_info": {
                "episode_reward": episode_score,
                "step_rewards": step_rewards,
                "first_round_reward": step_rewards[0] if step_rewards else 0,
                "final_reward": step_rewards[-1] if step_rewards else 0
            },
            "failure_info": {
                "failure_mode": last_step_info.get("failure_mode", ""),
                "stop_reason": self.stop_reason.name,
                "error_messages": last_step_info.get("error_messages", []),
                "test_output": last_step_info.get("test_output", ""),
                "has_failure": bool(failure_mode and failure_mode not in ['', 'none']),
                "failure_step": rollout_cache.step,
            },
            "metrics": env_metric,
            "last_observation": last_observation
        }

        # stepwise 样本只存一份traj data
        batch.non_tensor_batch["trajectory_data"][-1] = json.dumps(trajectory_data)
        batch.non_tensor_batch["messages"][-1] = json.dumps(all_messages)
        batch.non_tensor_batch["tools"][-1] = json.dumps(self.tools)

        # 避免 trajectory_data dict 过大，导致写入/读取odps失败
        colummns_config = [
            ["trajectory_data", "string"],
            ["messages", "string"],
            ["tools", "string"],
            ["exp_name", "string"],
        ]
        batch.meta_info["COLUMMNS_CONFIG"] = colummns_config
        return batch

    def create_placeholder_rollout(self, episode_id):
        """
                Create a minimal placeholder rollout with response_mask=1 to skip loss calculation.
                """
        self.logger.info(f"[PLACEHOLDER_ROLLOUT] failure_mode: {self.failure_mode}")

        seq_len = length=self.pipeline_config.sequence_length
        input_ids = torch.full((1, seq_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
        position_ids = torch.zeros((1, seq_len), dtype=torch.long)
        response_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        prompt_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        score_tensor = torch.zeros((1, seq_len), dtype=torch.float)

        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        }, batch_size=1)


        infer_logprobs = torch.zeros((1, seq_len - 1), dtype=torch.float)
        lm_input.batch["infer_logprobs"] = infer_logprobs

        lm_input.non_tensor_batch = {
            "env_ids": np.array([self.env_config['env_id']], dtype=object),
            "group_ids": np.array([self.env_config['group_id']], dtype=object),
            "tags": np.array([self.env_config['tag']], dtype=object),
            "step_scores": np.array([0], dtype=object),
            "episode_scores": np.array([0], dtype=object),
            "state_hash": np.array([''], dtype=object),
            "step": np.array([0], dtype=object),
            "trajectory_data": np.array([None], dtype=object),
            "messages": np.array([None], dtype=object),
            "tools": np.array([None], dtype=object),
            "exp_name": np.array([self.pipeline_config.exp_name], dtype=object),
        }

        traj_group_id = f"{self.env_config['tag']}_{self.env_config['group_id']}_{episode_id}_{self.group_seed}"
        traj_id = f"{traj_group_id}_{self.env_config['env_id']}"
        lm_input.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * lm_input.batch.batch_size[0], dtype=object)
        lm_input.non_tensor_batch["traj_id"] = np.array([traj_id] * lm_input.batch.batch_size[0], dtype=object)

        colummns_config = [
            ["trajectory_data", "string"],
            ["messages", "string"],
            ["tools", "string"],
            ["exp_name", "string"],
        ]
        lm_input.meta_info["COLUMMNS_CONFIG"] = colummns_config
        lm_input.meta_info["metrics"] = {}
        return lm_input



class GroupFilter:
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        self.config = config
        self.env_manager_config = env_manager_config
        self.mode = mode
        self.global_filter_stats = {"total": 0, "filtered": 0}

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        self.global_filter_stats["total"] += 1
        should_drop = False
        for data in group:
            if data.meta_info.get("drop_flag", False):
                should_drop = True

        if not should_drop:
            return False

        current_global_filter_ratio = (
            self.global_filter_stats["filtered"] / self.global_filter_stats["total"]
            if self.global_filter_stats["total"] > 0 else 0.0
        )

        if current_global_filter_ratio >= 0.5:
            return False

        if (self.global_filter_stats["filtered"] + 1) / self.global_filter_stats["total"] > 0.5:
            return False

        self.global_filter_stats["filtered"] += 1
        return True