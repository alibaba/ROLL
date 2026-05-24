import itertools
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import ray
from gem import Env
from omegaconf import OmegaConf

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.pipeline.agentic.env.rock.sandbox_manager_v2 import FailureMode, RunSessionResponse, SandboxManagerV2
from roll.utils.constants import RAY_NAMESPACE, EpisodeStopReason
from roll.utils.logging import get_logger


class RockTBNativeEnv(Env):
    """
    Terminal-bench native environment for iflow native mode.

    This environment provides Terminal-bench functionality using the iflow native
    architecture with ROCK model service integration. It handles terminal-based
    tasks with integrated agent interaction.
    """

    def __init__(
        self,
        exp_mode: str = "train",
        group_id: int = 0,
        num_env_groups: int = 1,
        max_steps: int = 80,
        mode: str = "train",
        xrl_authorization: str = "",
        sandbox_base_url: str = "https://xrl.alibaba-inc.com",
        user_id: str = "0000",
        experiment_id: str = "test",
        auto_clear_seconds: int = 60 * 60,
        run_region: str = "",
        agent_timeout_sec: int = 60 * 40,
        test_timeout_sec: int = 60 * 10,
        max_execute_time: float = 60 * 50,
        max_env_time: float = 60 * 50,
        timeout: int = 300,
        startup_timeout: int = 600,
        debug: bool = False,
        # Agent有关的参数都放这里, 直接遵从ROCK Agent的格式
        agent_config: dict = None,
        max_multi_session_num: int = 1,
        # Terminal-bench specific parameters
        dataset_name: Optional[str] = "data/swe_bench_verified_mini.jsonl",
        train_idx_range: Tuple[int, int] = (0, int(1e4)),  # 训练集任务ID范围
        val_idx_range: Tuple[int, int] = (0, int(1e4)),  # 验证集任务ID范围
        pass_low_threshold: float = 0.0,
        pass_high_threshold: float = 0.0,
        id_key: str = "id",
        question_key: str = "prompt",
        sandbox_image_key: str = "sandbox_image",
        task_name_key: str = "task_name",
        run_region_key: str = "run_region",
        start_script_key: str = "start_script",
        tag_key: str = "tag",
        test_files: List[str] = None,
        seed: int = 0,
        group_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Terminal-bench native environment.
        支持的tools: roll/pipeline/agentic/tools/iflow/iflow_config.py:279
        """

        # Basic environment parameters
        self.exp_mode = exp_mode
        self.group_id = group_id
        self.num_env_groups = num_env_groups
        self.max_steps = max_steps
        self.mode = mode
        self.xrl_authorization = xrl_authorization
        self.sandbox_base_url = sandbox_base_url
        self.user_id = user_id
        self.experiment_id = experiment_id
        self.auto_clear_seconds = auto_clear_seconds
        self.run_region = run_region
        self.startup_timeout = startup_timeout

        self.agent_timeout_sec = agent_timeout_sec
        self.test_timeout_sec = test_timeout_sec
        self.max_env_time = max_env_time

        if not isinstance(agent_config, dict):
            self.agent_config = OmegaConf.to_container(agent_config, resolve=False)
        else:
            self.agent_config = agent_config
        if "run_cmd" in self.agent_config:
            self.agent_config["run_cmd"] = self.agent_config["run_cmd"].replace("<<PROMPT>>", "${prompt}")

        self.debug = debug
        self.test_files = test_files or []
        self.max_multi_session_num = max_multi_session_num

        # Terminal-bench specific parameters
        self.dataset_name = dataset_name
        self.train_idx_range = train_idx_range
        self.val_idx_range = val_idx_range
        self.pass_low_threshold = pass_low_threshold
        self.pass_high_threshold = pass_high_threshold
        self.max_execute_time = max_execute_time
        self.timeout = timeout
        self.seed = seed
        self.group_key = group_key

        # Data keys for terminal-bench
        self.id_key = id_key
        self.question_key = question_key
        self.sandbox_image_key = sandbox_image_key
        self.task_name_key = task_name_key
        self.run_region_key = run_region_key
        self.start_script_key = start_script_key
        self.tag_key = tag_key
        self.data_line = {}
        self.task_id = -1
        self.task_name = ""
        self.prompt = ""
        self.sandbox_image = ""
        self.tag = ""
        self.start_script = ""

        self.time_start = 0
        self.rollout_time = 0.0
        self.traj_tool_execute_time = 0

        # runtime related
        self.current_step = 0
        self.session_num = 0
        self.current_session_step = 0
        self.logger = get_logger()
        self.sandbox_manager: Optional[SandboxManagerV2] = None
        self.test_output = ""
        self.failure_mode = ""  # 只要跟env交互过程中failure_mode不为空,交互就是失败的
        self.stop_reason = ""  # terminated的原因
        self.error_messages = []  # 记录交互过程中的error_messages
        self.reward, self.terminated, self.truncated = 0, False, False
        self.env_failed = False
        self.env_timeout = False
        self.env_reset_failed = False
        self.agent_timeout = False
        self.is_closed = False
        self.sandbox_ip = ""
        self.sandbox_id = ""

        # 数据集读取
        if self.mode == "train":
            global_dataset_mode = "sample"
        elif self.mode == "val":
            global_dataset_mode = "traversal"
        else:
            global_dataset_mode = self.mode

        self.dataset = GlobalDataset.options(
            name=f"{self.mode}_{self.dataset_name}", get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote(dataset_name=self.dataset_name, mode=global_dataset_mode)

        # 数据过滤
        idx_range = self.val_idx_range if self.mode == "val" else self.train_idx_range
        idx_list = self._parse_idx_range(idx_range)

        ray.get(
            self.dataset.filter.remote(
                filter_name="filter_idx_range", function=lambda x: int(x[self.id_key]) in idx_list
            )
        )

        self.dataset_manager = GlobalDatasetManager.options(
            name=f"{self.mode}_dataset_manager", get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))


    def _parse_idx_range(self, idx_range: Union[List[int], str]) -> List[int]:
        """
        Parse idx_range parameter into a list of integers.

        Args:
            idx_range: Can be either:
                - A list of integers: [0, 1, 2, 3]
                - A tuple of two integers (start, end): (0, 100)
                - A string that can be evaluated as List[int]: 'list(range(0, 8))'
                - ListConfig or similar iterable from Hydra/OmegaConf

        Returns:
            List of integers representing the valid indices.
        """
        if isinstance(idx_range, str):
            # String format: evaluate it safely
            try:
                idx_list = eval(idx_range)
                if not isinstance(idx_list, (list, tuple)):
                    raise ValueError(f"Evaluated idx_range must be a list or tuple, got {type(idx_list)}")
                return list(idx_list)
            except Exception as e:
                self.logger.error(f"Failed to evaluate idx_range string '{idx_range}': {e}")
                raise ValueError(f"Invalid idx_range string format: {idx_range}") from e
        else:
            # Handle list, tuple, ListConfig, or any iterable sequence
            try:
                # Convert to list to handle ListConfig and similar objects from Hydra
                idx_list = list(idx_range)

                # Check if it's a range tuple (start, end) or a list of indices
                if len(idx_list) == 2 and all(isinstance(x, (int, float)) for x in idx_list):
                    # Assume it's a range tuple
                    start, end = int(idx_list[0]), int(idx_list[1])
                    return list(range(start, end + 1))
                else:
                    # It's a list of specific indices
                    return [int(x) for x in idx_list]
            except (TypeError, ValueError) as e:
                self.logger.error(f"Failed to parse idx_range: {idx_range}, type: {type(idx_range)}")
                raise TypeError(f"idx_range must be List[int], str, or iterable, got {type(idx_range)}") from e

    def reset(self, seed=None) -> Tuple[List[Dict], Dict]:
        """Reset the environment and start a new episode."""
        super().reset(seed)
        self.clean_record()

        data_line: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data_line is None:
            return None, {}

        self.task_id = data_line[self.id_key]
        self.data_line = data_line

        self._reset_logger(seed)

        # Extract terminal-bench specific fields
        self.prompt = data_line[self.question_key]
        self.sandbox_image = data_line[self.sandbox_image_key]
        self.task_name = data_line[self.task_name_key]
        self.run_region = data_line.get(self.run_region_key, "")
        self.start_script = data_line.get(self.start_script_key, "")
        self.tag = data_line.get(self.tag_key, "")

        if "swebench" in self.sandbox_image:
            self.agent_config["project_path"] = "/testbed"
        else:
            self.agent_config["project_path"] = "/app"

        # Prepare phase
        self.start_sandbox()

        self.setup_sandbox_env()

        # iflow-cli --prompt {instruction}
        observation, tools, error_msg = self.reset_agent_status(prompt=self.prompt)

        return observation, {
            "tools": tools,
            "error_msg": error_msg,
            "failure_mode": self.failure_mode,
            "task_name": self.task_name,
        }

    def step(self, action: str) -> Tuple[Union[List[Dict], str], SupportsFloat, bool, bool, dict[str, Any]]:
        self.current_step += 1
        self.current_session_step += 1
        info = {}

        self.logger.info(f"[ENV_STEP] START - GroupID: {self.group_id}, Step:{self.current_step}, Response:{action}")
        if self.rollout_time > self.max_env_time:
            action = EpisodeStopReason.ENV_TIMEOUT
            self.env_timeout = True
            self.logger.error(f"[ENV_STEP] Failed! - Environment timeout after {self.max_env_time / 60} minutes")

        if self.exp_mode == "eval_gt":
            self.logger.info(f"[EVAL_GT] START - Running ground truth solution for task: {self.task_name}")
            observation, reward, terminated, truncated, info = self.sandbox_manager.run_ground_truth_solution(
                self.agent_timeout_sec, self.task_id, self.task_name
            )
            if not info.get("success", "True"):
                self.error_messages.append(observation)
            action = EpisodeStopReason.EVAL_GT

        if isinstance(action, EpisodeStopReason) and action in [
            EpisodeStopReason.MAX_LENGTH,
            EpisodeStopReason.ENV_TIMEOUT,
            EpisodeStopReason.EVAL_GT,
        ]:
            # 控制类action，主动终止，终止时计算一次reward
            observation, tools, error_msg, self.reward = self.check_terminated(
                next_request_payload="", force_terminated=True
            )
            return observation, self.reward, True, True, {**info}

        response_payload, info = self.sandbox_manager.format_response_payload(response=action)
        request_response: RunSessionResponse = self.sandbox_manager.fetch_agent_request(
            index=self.current_session_step, response_payload=response_payload
        )
        if request_response.exit_code != 0:
            # 交互失败，当前SESSION_END, 再次尝试交互
            self.env_reset_failed = True
            self.failure_mode = FailureMode.MODEL_SERVICE_ANTI_CALL_LLM_FAILED
            error_msg = f"[{self.failure_mode}]: {request_response.failure_reason}\t{request_response.output}"
            self.error_messages.append(error_msg)
            self.logger.error(error_msg)
            next_request_payload = "SESSION_END"
        else:
            next_request_payload = request_response.output

        next_request_payload = next_request_payload.strip().split("\n")[-1]

        observation, tools, error_msg, self.reward = self.check_terminated(next_request_payload=next_request_payload)
        if "SESSION_END" != next_request_payload:
            observation, tools, error_msg = self.sandbox_manager.get_messages_and_tools(
                request_payload=next_request_payload
            )

        if not observation:
            self.failure_mode = FailureMode.MODEL_SERVICE_ANTI_CALL_LLM_FAILED
            error_msg = f"[{self.failure_mode}]: {request_response.failure_reason}\t{request_response.output}"
            self.error_messages.append(error_msg)
            self.logger.error(error_msg)
            self.terminated = True
            self.truncated = True

        action_is_valid = info.get("action_is_valid", False)
        metrics = {
            "env_timeout": self.env_timeout,
            "env_reset_failed": self.env_reset_failed,
            "action_is_valid": action_is_valid,
            "success": self.reward > 0,
            "raw_reward": self.reward,
            "current_step": self.current_step,
            "task_id": self.task_id,
        }

        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "raw_reward": "last",
        }
        info_new = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "failure_mode": self.failure_mode,
            "error_messages": self.error_messages,
            "stop_reason": self.stop_reason,
            "test_output": self.test_output,
        }
        info.update(info_new)
        self.logger.info(
            f"[ENV_STEP] Success! - Step {self.current_step} finished, task: {self.task_name}, reward: {self.reward}, success: {self.reward > 0}"
        )
        return observation, self.reward, self.terminated, self.truncated, info

    def check_terminated(self, next_request_payload: str, force_terminated=False):
        # Initialize termination states for this step
        self.terminated = False
        self.truncated = False

        # Check step limit termination first
        step_limit_reached = self.current_step >= self.max_steps

        # Check if episode should end (either by session end or step limit)
        should_end_episode = ("SESSION_END" == next_request_payload) or step_limit_reached or force_terminated

        observation = next_request_payload
        tools = []
        error_msg = ""
        if should_end_episode:
            self.session_num += 1

            # Always calculate reward when ending a session/episode
            self.reward, error_info, test_output = self.calculate_reward()

            if step_limit_reached:
                self.terminated = True
                self.truncated = True
                self.stop_reason = "Reached maximum steps"
                observation = f"ERROR: Reached maximum steps {self.max_steps}"
                self.logger.info(f"[MAX_STEPS] Reached maximum steps ({self.max_steps}), truncating episode")
            elif force_terminated:
                self.terminated = True
                self.truncated = True
                self.stop_reason = "Force terminated"
                observation = f"self.stop_reason: {self.stop_reason}"
                self.logger.info("[truncated] truncating episode")
            elif "SESSION_END" == next_request_payload:
                self.terminated = True
                self.stop_reason = "Session ended naturally"

            # Check if we should start a new session or terminate completely
            if self.session_num < self.max_multi_session_num and not step_limit_reached and not force_terminated:
                # Start new session with test output as prompt
                prompt_with_test = f"{test_output}"
                observation, tools, error_msg = self.reset_agent_status(prompt=prompt_with_test)
                # Reset termination flags since we're starting a new session
                self.terminated = False
                self.truncated = False
            else:
                # Final termination - either reached max sessions or step limit
                self.terminated = True
                if step_limit_reached:
                    self.truncated = True
        if self.terminated:
            self.close()
        return observation, tools, error_msg, self.reward

    def clean_record(self):
        """Clean all episode-specific records."""
        self.data_line = {}
        self.task_id = -1
        self.task_name = ""
        self.prompt = ""
        self.sandbox_image = ""
        self.reward, self.terminated, self.truncated = 0, False, False
        self.env_failed = False
        self.env_timeout = False
        self.env_reset_failed = False
        self.agent_timeout = False
        self.is_closed = False
        self.current_step = 0
        self.current_session_step = 0
        self.error_messages.clear()
        self.failure_mode = ""
        self.test_output = ""
        self.sandbox_ip = ""
        self.sandbox_id = ""

    def start_sandbox(self):
        """Initialize ROCK service and start base image."""
        self.logger.info(
            f"[SANDBOX_INIT] START - Initializing sandbox with image: {self.sandbox_image}, task: {self.task_name}"
        )
        self.sandbox_manager = SandboxManagerV2(
            sandbox_image=self.sandbox_image,
            logger=self.logger,
            # xrl相关的参数
            xrl_authorization=self.xrl_authorization,
            sandbox_base_url=self.sandbox_base_url,
            user_id=self.user_id,
            experiment_id=self.experiment_id,
            startup_timeout=self.startup_timeout,
            default_timeout=self.timeout,
            # iflow相关的参数
            agent_config=self.agent_config,
            # others
            run_region=self.run_region,
            start_script=self.start_script,
            dataset_tag=self.tag,
            test_files=self.test_files,
            task_name=self.task_name,
            debug=self.debug,
        )

        if not self.sandbox_manager.is_environment_available:
            self.env_reset_failed = True
        else:
            self.env_reset_failed = False
            self.logger.info(
                f"[ENV_RESET] Success! - TaskID: {self.task_id}, TaskName: {self.task_name}, image: {self.sandbox_image}, sandbox_ip: {self.sandbox_manager.sandbox_ip}, sandbox_id: {self.sandbox_manager.sandbox_id}"
            )
            self.sandbox_ip = self.sandbox_manager.sandbox_ip
            self.sandbox_id = self.sandbox_manager.sandbox_id

    def _reset_logger(self, seed):
        """Reset logger for new episode."""
        self.logger = get_logger()
        self.logger.info(f"start reset, task_id: {self.task_id}, task_name: {self.task_name}")

    def setup_sandbox_env(self):
        """Setup terminal-bench specific environment if needed."""
        # TODO: env setup相关的，不应该放sandbox_manager里
        pass

    def reset_agent_status(self, prompt):
        """
        交互失败返回 observation, tools, error_msg = [], [], ""
        """
        self.current_session_step = 0
        observation, tools, error_msg = [], [], ""

        start_agent_response = self.sandbox_manager.start_agent(prompt=prompt)
        if start_agent_response.exit_code != 0:
            self.env_reset_failed = True
            self.failure_mode = FailureMode.AGENT_START_FAILED
            error_msg = f"[{self.failure_mode}]: {start_agent_response.failure_reason}\t{start_agent_response.output}"
            self.error_messages.append(error_msg)
            self.logger.error(error_msg)
            self.terminated = True
            return observation, tools, error_msg

        # self.sandbox_manager.anti_call_llm(), 拿到iflow-cli的初始请求，解出messages和tools
        request_response: RunSessionResponse = self.sandbox_manager.fetch_agent_request(
            index=self.current_session_step
        )

        request_payload = request_response.output
        request_payload = request_payload.strip().split("\n")[-1]

        if request_response.exit_code != 0 or not request_payload:
            self.env_reset_failed = True
            self.failure_mode = FailureMode.MODEL_SERVICE_ANTI_CALL_LLM_FAILED
            error_msg = f"[{self.failure_mode}]: {request_response.failure_reason}\t{request_response.output}"
            self.error_messages.append(error_msg)
            self.logger.error(error_msg)
            self.terminated = True
            return observation, tools, error_msg

        observation, tools, error_msg = self.sandbox_manager.get_messages_and_tools(request_payload=request_payload)

        if error_msg or not observation:
            self.env_reset_failed = True
            self.failure_mode = FailureMode.MODEL_SERVICE_ANTI_CALL_LLM_FAILED
            self.error_messages.append(error_msg)
            self.terminated = True

        return observation, tools, error_msg

    def calculate_reward(self):
        """Calculate reward for the current episode."""
        self.logger.info(
            f"[REWARD_CALC] START - Calculating reward for task_id: {self.task_id}, task_name: {self.task_name}"
        )
        # TODO: run_tests逻辑应该在env层面， SandboxManager应该做成跟sandbox交互的工具类
        self.reward, error_info, self.test_output, test_command = self.sandbox_manager.run_tests(
            self.test_files, self.test_timeout_sec, self.task_name
        )
        return self.reward, error_info, self.test_output

    def close(self):
        """Close the environment and release resources."""
        if not self.debug:
            self.sandbox_manager.stop_sandbox()

    @property
    def env_info(self) -> Dict:
        """Get environment information."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "sandbox_image": self.sandbox_image,
            "sandbox_ip": self.sandbox_manager.sandbox_ip if self.sandbox_manager else None,
            "sandbox_id": self.sandbox_manager.sandbox_id if self.sandbox_manager else None,
        }


if __name__ == "__main__":
    """Test Terminal-bench native environment"""
    max_steps = 5
    env = RockTBNativeEnv(
        mode="train",
        sandbox_base_url= "http://localhost:8080",
        user_id="xxx",
        experiment_id="test-for-rock",
        dataset_name="data/swe_bench_verified_example.jsonl",
        train_idx_range=(476, 476),
        max_steps=5,
        max_multi_session_num=1,
        debug=False,
        test_files=["/terminal-bench-datasets/datasets/swebench-verified"],
        agent_config={
            "agent_type": "default",
            "run_cmd": "iflow -p <<PROMPT>> --yolo",
            "runtime_env_config": {
                "type": "node",
                "npm_registry": "https://registry.npmmirror.com",
                "custom_install_cmd": "wget --retry-connrefused --tries=10 --waitretry=2 -O ~/iflow-cli.tgz 'http://cloud.iflow.cn/iflow-cli/iflow-ai-iflow-cli-for-roll-0-4-4-v5.tgz' && npm i -g ~/iflow-cli.tgz",
            },
            "env": {
                "IFLOW_apiKey": "test",
                "IFLOW_baseUrl": "http://localhost:8080/v1",
                "IFLOW_modelName": "ROME",
                "IFLOW_searchApiKey": "88888888",
                "IFLOW_selectedAuthType": "openai-compatible",
                "IFLOW_disableAutoUpdate": "true",
                "IFLOW_tokensLimit": "128000",
                "IFLOW_shellTimeout": "360000",
                "IFLOW_coreTools": "Edit,exit_plan_mode,glob,list_directory,multi_edit,plan,read plan,read_file,read_many_files,save_memory,Search,Shell,task,web_fetch,web_search,write_file,xml_escape",
            },
            "model_service_config": {"type": "local", "enabled": True},
            "pre_init_cmds": [
                {"command": 'wget -q https://xrl-sandbox-bucket.oss-cn-hangzhou.aliyuncs.com/uv-files/uv-x86_64-unknown-linux-gnu.tar.gz && tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz --strip-components=1 -C /usr/local/bin && uv --version', "timeout_seconds": 30},
            ],
        },
    )
    obs, info = env.reset(seed=42)

    print(f"\n[observation]\n{obs}")
    print(f"\n[info]\n{info}")

    test_actions = [
        "Let me check the current directory.<tool_call><function=list_directory><parameter=path>/app</parameter></function></tool_call>",
        "ok, thank you.",
    ]
    i = 0
    for action in itertools.cycle(test_actions):
        obs, reward, terminate, truncated, info = env.step(action)
        print(f"\n[observation]-{i}\n{obs}")
        print(f"\n[reward]-{i}\n{reward}")
        print(f"\n[terminate]-{i}\n{terminate}")
        print(f"\n[truncated]-{i}\n{truncated}")
        print(f"\n[info]-{i}\n{info}")

        i += 1
        if (terminate or truncated) and i < max_steps:
            break
