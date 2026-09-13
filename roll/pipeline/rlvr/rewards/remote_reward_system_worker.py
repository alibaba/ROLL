# Copyright 2025 alibaba-inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import requests

import concurrent.futures
from datetime import datetime 
from typing import Optional, Union, Dict, List, Any, Tuple

from codetiming import Timer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.logging import get_logger

logger = get_logger()


class RewardSystemHandler:
    def __init__(self, 
                 reward_system_config: Dict, 
                 reward_manager_config: Dict, 
                 reward_server_task_timeout: int = 300):
        """
        初始化奖励系统处理器。

        Args:
            reward_system_config: 包含 xrl_authorization、URL、模式等配置。
            reward_manager_config: 奖励管理器配置，透传给后端。
            reward_server_task_timeout: 任务最大服务时间，单位为秒。超过这个时间后，任务会被服务端取消。
        """
        
        xrl_authorization = reward_system_config.get("xrl_authorization", 
                                                     "t-29imkiykio27pmju")
        submit_url = reward_system_config.get("submit_url", 
                                              "http://xrl.alibaba-inc.com/apis/custom/reward-system/v1/submit_task")
        get_result_url = reward_system_config.get("get_result_url", 
                                                  "http://xrl.alibaba-inc.com/apis/custom/reward-system/v1/task_result")
        
        mode = reward_system_config.get("reward_system_type", "online")
        assert mode in ["online", "pre"], f"Invalid xrl mode: {mode}"
        
        project_id = reward_system_config.get("project_id", "roll_default_task_id")
        experiment_id = reward_system_config.get("experiment_id", "roll_default_task_id")
        global_step = reward_system_config.get("global_step", 0)
        priority = reward_system_config.get("priority", 1)

        self.xrl_authorization = xrl_authorization
        
        self.submit_url = submit_url
        self.get_result_url = get_result_url

        self.headers = {
            "Content-Type": "application/json",
            "XRL-Authorization": f"Bearer {xrl_authorization}",
            "BusinessType": mode
            }
        
        # TODO: add dynamic priority
        self.priority = priority
        self.project_id = project_id
        self.experiment_id = experiment_id
        self.global_step = global_step
        self.reward_server_task_timeout = reward_server_task_timeout
        
        self.reward_manager_config = reward_manager_config

    def submit_task(
        self,
        prompt: str,
        response: str,
        reward_info: Dict,
        data_id: str,
        rollout_id: str,
        project_id: str = None,
        experiment_id: str = None,
        global_step: str = None,
        priority: int = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        timeout: int = 20
    ) -> Optional[str]:
        """
        提交 reward 异步计算任务。
        Args:
            prompt (str): _description_
            response (str): _description_
            reward_info (Dict): _description_
            data_id (str): _description_
            rollout_id (str): _description_
            project_id (str, optional): _description_. Defaults to None.
            experiment_id (str, optional): _description_. Defaults to None.
            global_step (str, optional): _description_. Defaults to None.
            priority (int, optional): _description_. Defaults to None.
            max_retries (int, optional): _description_. Defaults to 5.
            retry_delay (float, optional): _description_. Defaults to 1.0.
            timeout (int, optional): _description_. Defaults to 20.

        Returns:
            str: reward_request_id，失败时返回 None。
        """
        if priority is None:
            priority = self.priority
        if project_id is None:
            project_id = self.project_id
        if experiment_id is None:
            experiment_id = self.experiment_id
        if global_step is None:
            global_step = self.global_step


        payload = {
            "priority": priority,
            "project_id": project_id,
            "experiment_id": experiment_id,
            "global_step": global_step,
            "data_id": data_id,
            "rollout_id": rollout_id,
            "prompt": prompt,
            "response": response,
            "reward_info": reward_info,
            "reward_manager": self.reward_manager_config,
            "timeout": self.reward_server_task_timeout
        }

        reward_request_id = None
        task_time_start = 0

        for attempt in range(1, max_retries + 1):
            try:
                task_time_start = time.time()
                response = requests.post(
                    self.submit_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()  # 如果状态码不是 2xx，会抛出 HTTPError
                response_json = response.json()
                reward_request_id = response_json.get("reward_request_id")
                errcode = int(response_json.get("errcode", -1))
                
                if errcode == 0:
                    break
                else:
                    errmsg = response_json.get("errmsg", "未知错误")
                    logger.warning(f"Try {attempt} out of {max_retries} times, {errcode=}, {errmsg=}; {payload=}")

            except Exception as e:
                logger.warning(f"Try {attempt} out of {max_retries} times, {e}")

            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries:
                time.sleep(retry_delay)
        if reward_request_id is None:
            debug_info = {
                "project_id": project_id,
                "experiment_id": experiment_id,
                "global_step": global_step,
                "data_id": data_id,
                "rollout_id": rollout_id,
                "prompt": prompt,
                "response": response
            }
            logger.warning(f"Reward system submit task failed. {debug_info=}")
            
        return reward_request_id, task_time_start

    def get_task_result(
        self,
        reward_request_id: str,
        task_time_start: float,
        retry_delay: float = 2.0,
        timeout: int = 20
        ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """
        轮训查询任务结果
        """
        def extract_time_costs(task_time_start, response_json):
            time_info = {
                "local_request_time_cost": 0,
                "server_task_time_cost": 0,
                "plugin_time_cost": []
            }
            
            time_info["local_request_time_cost"] = time.time() - task_time_start
            if response_json:
                server_log_start_time = response_json.get("start_time", None)
                server_log_end_time = response_json.get("end_time", None)
                if server_log_start_time and server_log_end_time:
                    try:
                        server_log_start_time = datetime.fromisoformat(server_log_start_time)
                        server_log_end_time = datetime.fromisoformat(server_log_end_time)
                        time_info["server_task_time_cost"] = (server_log_end_time - server_log_start_time).total_seconds()
                    except Exception as e:
                        logger.warning(f"Get Reward System server time cost failed. {e}")
                dimention_reward_score = response_json.get("dimention_reward_score", {})
                
                for key, value in dimention_reward_score.items():
                    debug_info = value.get("debug_info", {})
                    for key, value in debug_info.items():
                        time_info["plugin_time_cost"].append({key: value.get('time_cost', 0)})
                
            return time_info

        payload = {"reward_request_id": reward_request_id}
        
        reward_score = None
        dimention_reward_score = None
        local_request_time_cost = 0
        attempt = 0
        
        # 预留 5s 查询余量，避免临界超时任务漏查。
        while local_request_time_cost < self.reward_server_task_timeout + 5:
            time_info = None
            response_json = None
            attempt += 1
            
            try:
                response = requests.get(
                    self.get_result_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )
                time_info = extract_time_costs(task_time_start, response_json)
                local_request_time_cost = time_info["local_request_time_cost"]
                response.raise_for_status()
                response_json = response.json()
                errcode = int(response_json.get("errcode", -1))
                errmsg = response_json.get("errmsg", "unknown error")
                if errcode == 0:
                    reward_score = response_json.get("reward_score")
                    dimention_reward_score = response_json.get("dimention_reward_score")
                    break
                # 1 表示任务还在执行中
                elif errcode == 1:
                    pass
                else:
                    logger.warning(f"Try {attempt} times,"
                                   f"{time_info=}s; {errcode=}, {errmsg=}; {payload=}")
            except Exception as e:
                logger.warning(f"Try {attempt} times, {time_info=}; {e}")
            
            if time_info is None:
                time_info = extract_time_costs(task_time_start, response_json)
            if local_request_time_cost > self.reward_server_task_timeout:
                logger.warning(f"Reward system get task result timeout. {time_info=}; {reward_request_id=}")
                break

            time.sleep(retry_delay)
        
        if reward_score is None:
            logger.warning(f"Reward system reward calculation failed. reward_server_task_timeout: "
                           f"{self.reward_server_task_timeout}; {time_info=}; {reward_request_id=}")

        return reward_score, dimention_reward_score

    def get_reward(
        self,
        prompt: str,
        response: str,
        reward_info: Dict,
        data_id: str,
        rollout_id: str,
        project_id: str = None,
        experiment_id: str = None,
        global_step: str = None,
        priority: int = None,
        ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """
        提交任务并等待结果（同步阻塞）

        Args:
            prompt (str): _description_
            response (str): _description_
            reward_info (Dict): _description_
            data_id (str): _description_
            rollout_id (str): _description_
            project_id (str, optional): _description_. Defaults to None.
            experiment_id (str, optional): _description_. Defaults to None.
            global_step (str, optional): _description_. Defaults to None.
            priority (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        reward_request_id, task_time_start = self.submit_task(
            prompt=prompt,
            response=response,
            reward_info=reward_info,
            data_id=data_id,
            rollout_id=rollout_id,
            project_id=project_id,
            experiment_id=experiment_id,
            global_step=global_step,
            priority=priority,
        )
        if reward_request_id is None:
            return None, None
        return self.get_task_result(reward_request_id, task_time_start)

    def multi_thread_request_reward_system(self, input_list) -> List[float]:
        """
        多线程批量请求奖励分数。

        input_list 元素格式:
            (project_id, experiment_id, global_step, data_id, rollout_id, reward_info, prompt, response, response_token_str)
        注意：response_token_str 当前未使用，保留以备扩展。
        """
        def thread_request_worker(
                                index, 
                                project_id, 
                                experiment_id, 
                                global_step, 
                                data_id, 
                                rollout_id, 
                                reward_info, 
                                prompt, 
                                response
                                ):

            for attempt in range(5):
                score, dimention_reward = self.get_reward(
                    prompt=prompt,
                    response=response,
                    reward_info=reward_info,
                    data_id=data_id,
                    rollout_id=rollout_id,
                    project_id=project_id,
                    experiment_id=experiment_id,
                    global_step=global_step,
                )

                if score is not None:
                    break
  
            if score is None:
                debug_info = {
                    "project_id": project_id,
                    "experiment_id": experiment_id,
                    "global_step": global_step,
                    "data_id": data_id,
                    "rollout_id": rollout_id,
                    "prompt": prompt,
                    "response": response,
                    "reward_info": reward_info
                }
                logger.warning(f"Failed to get reward from rewared system. debug_info: {debug_info}")
                score = 0

            return index, score

        output_list = [None] * len(input_list)
        max_workers = max(min(32, len(input_list)), 1)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(thread_request_worker, index, *item): index
                for index, item in enumerate(input_list)
            }
            # 等待完成并按索引填充结果
            for future in concurrent.futures.as_completed(future_to_index):
                index, result = future.result()
                output_list[index] = result
        
        for index, result in enumerate(output_list):
            if result is None:
                output_list[index] = 0
        
        return output_list


class RemoteRewardSystemWorker(Worker):
    """
    Reward Worker that uses Reward System to compute rewards.
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.reward_manager_config = self.worker_config.reward_manager_config
        

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass
    
    def _uniform_reward_ref_info_adaper(self, data: DataProto) -> List[Dict]:
        """
        Normalize various input data formats into a uniform reward reference format.

        Args:
            data (DataProto): Input data containing non-tensor metadata.

        Returns:
            List[Dict[str, Any]]: List of reward reference info dicts.

        Raises:
            ValueError: If data format is not supported.
        """
        def is_code_format_data(reward_manager_config, non_tensor_batch):
            """Check if data is in code evaluation format."""
            if (
                non_tensor_batch.get("test_cases", [""])[0]
                and non_tensor_batch.get("case_type", [""])[0]
                ):
                return True
            for item in reward_manager_config:
                if "code" in item["plugin_name"]:
                    return True
            return False

        non_tensor_batch = data.non_tensor_batch
        # Case 1: Already in standard format
        if "reward_ref_info" in non_tensor_batch:
            reward_ref_info = non_tensor_batch["reward_ref_info"]
        # Case 2: Code-related data (detected by fields or config)
        elif is_code_format_data(self.reward_manager_config, non_tensor_batch):
            reward_ref_info = [
                {"case_type": case_type, "test_cases": test_cases} for case_type, test_cases in zip(
                        non_tensor_batch["case_type"],
                        non_tensor_batch["test_cases"],
                    )
            ]
        # Case 3: Ground truth available
        elif "ground_truth" in non_tensor_batch:
            reward_ref_info = [
                {"ground_truth": ground_truth} for ground_truth in non_tensor_batch["ground_truth"]
                ]
        # Unsupported format
        else:
            known_keys = list(non_tensor_batch.keys())
            raise ValueError(
                f"Unsupported data format for reward system. "
                f"Available keys: {known_keys}. "
                f"Expected one of: 'reward_ref_info', 'ground_truth', or code fields ('case_type', 'test_cases')."
            )
        
        return reward_ref_info

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto) -> DataProto:
        """
        Compute rewards using remote reward system.

        Args:
            data (DataProto): Input batch with prompts, responses, and metadata.

        Returns:
            DataProto: Output with token-level and response-level rewards.
        """
        global_step = data.meta_info.get("global_step", 0)
        reward_system_config = data.meta_info.get("reward_system_config", {})
        project_id = reward_system_config.get("project_id", 'roll_default_experiment_id')
        experiment_id = reward_system_config.get("experiment_id", 'roll_default_task_id')
        
        reward_system_handler = RewardSystemHandler(reward_system_config, self.reward_manager_config)

        prompts_text_list = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        
        data_id_list = data.non_tensor_batch["id"]
        rollout_id_list = data.non_tensor_batch["rollout_id"]
        
        reward_ref_info_list = self._uniform_reward_ref_info_adaper(data)
        
        
        reward_timer = Timer("reward_timer")
        input_list = []
        
        with reward_timer:
            for data_id, prompt, resp_text, reward_ref_info, rollout_id in zip(
                data_id_list, 
                prompts_text_list, 
                response_text_list, 
                reward_ref_info_list,
                rollout_id_list,
                ):
                prompt = prompt.replace("<|endoftext|>", "").replace("<pad>", "")
                prompt = prompt.replace("assistant\n<think>", "")
                
                input_list.append([
                    project_id, 
                    experiment_id, 
                    global_step, 
                    data_id, 
                    rollout_id, 
                    reward_ref_info, 
                    prompt, 
                    resp_text
                    ])

        scores = reward_system_handler.multi_thread_request_reward_system(input_list)

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

        return output

