import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rock.sdk.sandbox.client import Sandbox
from rock.sdk.sandbox.speedup.types import SpeedupType
from typing_extensions import override

from roll.pipeline.agentic.env.rock.agent_manager import AgentManager
from roll.pipeline.agentic.env.rock.sanbox_manager import FailureMode, RunSessionResponse, RunStatus, SandboxManager
from roll.pipeline.agentic.tools.action_parser import ActionParser, Qwen3CoderActionParser


logging.getLogger("httpx").setLevel(logging.ERROR)


class SandboxManagerV2(SandboxManager):
    """
    SandboxManagerV2 is a SandboxManager customized for the rl-rock SDK, distinct from rock-rl.
    There are slight differences in import paths and SDK method names.

    rl-rock provides improved Agent lifecycle management and support. It offers lightweight
    interfaces such as agent.install and agent.run for launching, making it easier to migrate
    across different Agents.

    When you need to use rl-rock, it is recommended to use SandboxManagerV2.
    """
    def __init__(
        self,
        sandbox_image: str,
        logger,
        xrl_authorization: str = "",
        sandbox_base_url: str = "http://localhost:8080",
        user_id: str = "0000",
        experiment_id: str = "test",
        agent_config: dict = {
            "agent_type": "iflow-cli",
            "agent_version": "0.0.1",
        },
        run_region: str = "",
        start_script: str = "",
        dataset_tag: str = "",
        test_files: List[str] = None,
        task_name: str = "",
        debug: bool = False,
        default_timeout: float = 60.0,
        startup_timeout: float = 600.0,
        install_agent_timeout: float = 1200.0,
        default_head_content_limit: int = 10 * 1024 * 1024,
    ):
        self.sandbox: Sandbox = None
        self.sandbox_image = sandbox_image
        self.logger = logger
        self.xrl_authorization = xrl_authorization
        self.sandbox_base_url = sandbox_base_url
        self.user_id = user_id
        self.experiment_id = experiment_id

        self.agent_config = agent_config

        self.run_region = run_region
        self.start_script = start_script
        self.dataset_tag = dataset_tag
        self.test_files = test_files
        self.task_name = task_name
        self.debug = debug

        self.active_sessions = {}
        self.is_initialized = False
        self.agent_session_name = "agent"
        self.test_session_name = "test"

        self.max_retry = 3
        self.backoff = 2.0
        self.startup_timeout = startup_timeout
        self.install_agent_timeout = install_agent_timeout

        self.image_id = sandbox_image
        self.auto_clear_seconds = 60 * 60
        self.default_timeout = default_timeout
        self.head_content_limit = default_head_content_limit

        self.failure_mode = FailureMode.NONE
        self.run_status = RunStatus.SUCCESS
        self.error_messages = []

        self.is_environment_available = False
        self.initialization_error = None

        # Model service client properties
        self.proxy_session_name = "model_service"
        self.error_suffix = ""

        self.action_parser: ActionParser = Qwen3CoderActionParser()  # TODO: 支持更多类型的aciton parser
        self.agent_manager: AgentManager = None

        self._initialize_sandbox_with_times()

    @override
    def upload_file(self, file_path: Union[str, Path], target_path: str, max_retry: int = 3, backoff: float = 2.0):
        """Upload a file to the sandbox"""
        for attempt in range(1, max_retry + 1):
            self.logger.debug(
                f"[upload_file, {attempt}/{max_retry}] image_id:{self.image_id}, file_path:{file_path} target_path: {target_path}, sandbox_id:{self.sandbox.sandbox_id}"
            )
            try:
                response = asyncio.run(
                    self.sandbox.upload_by_path(str(file_path), target_path)
                )  # rl-rock use upload_by_path to replace aupload
                return response.success, response.message
            except Exception as exc:
                self.logger.error(
                    f"image_id:{self.image_id}, file_path:{file_path} target_path: {target_path}, upload failed: {str(exc)}, "
                    f"sandbox_id:{self.sandbox.sandbox_id}"
                )
                if attempt == max_retry:
                    return False, f"upload_file exp:{str(exc)}"
                time.sleep(backoff * attempt)
        return False, "upload_file failed"

    def _upload_and_execute_script(
        self, script_content: str, script_name: str, session_name: str, timeout: int = 300, log_filename: str = None
    ) -> Tuple[bool, str]:
        """
        上传并执行脚本

        Args:
            script_content: 脚本内容
            script_name: 脚本文件名
            session_name: 会话名称
            timeout: 超时时间（秒）
            log_filename: 日志文件名（可选）

        Returns:
            (成功标志, 错误信息)
        """
        try:
            script_path = f"/tmp/{script_name}"

            # 上传脚本
            is_success, message = self._upload_settings(script_content, "/tmp", script_name)
            if not is_success:
                return False, f"Failed to upload script {script_name}: {message}"

            # 执行脚本
            if log_filename is None:
                log_filename = f"{script_name.replace('.sh', '')}_info.txt"

            run_status, result = self.run_session_with_timeout(
                session_name,
                f"bash {script_path}",
                timeout,
                log_filename,
            )

            if run_status != RunStatus.SUCCESS:
                return False, f"Script execution failed: {run_status}, {result}"

            return True, ""

        except Exception as e:
            return False, f"Error executing script {script_name}: {str(e)}"

    def _setup_speedup(self, session_name: str) -> Tuple[bool, str]:
        """
        根据环境配置加速

        Args:
            session_name: 会话名称

        Returns:
            (成功标志, 错误信息)
        """
        try:
            if self.run_region == "sg":
                # sg 环境不需要加速配置
                self.logger.info("Skipping speedup configuration for sg region")
                return True, ""

            elif self.dataset_tag == "terminal_bench_v2.0":
                # terminal_bench 环境：只配置 APT 加速（阿里云内网源）
                self.logger.info("Configuring APT speedup for terminal_bench environment...")
                result = asyncio.run(
                    self.sandbox.network.speedup(
                        speedup_type=SpeedupType.APT,
                        speedup_value="http://yum.tbsite.net/mirrors",
                        timeout=300,
                    )
                )
                if result.exit_code != 0:
                    self.logger.warning(f"APT speedup skipped: {result.output} ({result.failure_reason})")
                return True, ""

            else:
                # 默认环境：配置 APT 和 PIP 加速（阿里云公网源）
                self.logger.info("Configuring APT and PIP speedup...")

                # 配置 APT 加速
                apt_result = asyncio.run(
                    self.sandbox.network.speedup(
                        speedup_type=SpeedupType.APT,
                        speedup_value="http://mirrors.cloud.aliyuncs.com",
                        timeout=300,
                    )
                )
                if apt_result.exit_code != 0:
                    self.logger.warning(f"APT speedup skipped: {apt_result.output} ({apt_result.failure_reason})")

                # 配置 PIP 加速
                pip_result = asyncio.run(
                    self.sandbox.network.speedup(
                        speedup_type=SpeedupType.PIP,
                        speedup_value="http://mirrors.cloud.aliyuncs.com/pypi/simple/",
                        timeout=60,
                    )
                )
                if pip_result.exit_code != 0:
                    self.logger.warning(f"PIP speedup skipped: {pip_result.output} ({pip_result.failure_reason})")

                return True, ""

        except Exception as e:
            error_msg = f"Error during speedup configuration: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    @override
    def _install_agent(self, session_name: str) -> Tuple[bool, str]:
        """Install and configure the agent in the sandbox"""
        self.agent_manager: AgentManager = AgentManager(self.sandbox, self.agent_config)

        try:
            # 配置加速
            is_success, message = self._setup_speedup(session_name)
            if not is_success:
                return False, f"Speedup configuration failed: {message}"

            # 安装 agent
            self.agent_manager.install_agent()

            return True, ""

        except Exception as e:
            error_msg = f"Error during agent installation: {e}, {self.error_suffix}"
            self.logger.error(error_msg)
            return False, error_msg

    @override
    def format_response_payload(self, response: str) -> Tuple[str, Dict]:
        """
        用action_parser代替了原来iflow_cli_tool的parse逻辑
        """
        self.logger.debug(f"[FORMAT_RESPONSE] START - Processing response of length: {len(response)}")

        # Extract tool calls and content upfront
        tool_calls = []
        content = response
        has_tool_calls = "<tool_call>" in response
        action_is_valid = False
        info = {}
        # Parse tool calls if present
        if has_tool_calls:
            self.logger.debug(f"[FORMAT_RESPONSE] Tool calls detected in response")
            try:
                is_parsed, parsed_tool_calls = self.action_parser.parse_action(response)
                if is_parsed and parsed_tool_calls:
                    tool_calls = parsed_tool_calls
                    # Extract the text content before tool calls
                    content_parts = response.split("<tool_call>")
                    content = content_parts[0] if content_parts else response
                    self.logger.debug(f"[FORMAT_RESPONSE] Tool calls formatted successfully")
                    action_is_valid = True
                else:
                    # Tool call parsing failed, treat as regular response
                    content = response
                    self.logger.debug(f"[FORMAT_RESPONSE] Tool call parsing failed, treating as regular response")

            except Exception as parse_exc:
                self.logger.error(f"[FORMAT_RESPONSE] Error parsing tool calls: {str(parse_exc)}, {self.error_suffix}")
                # Fallback to regular response
                content = response
        else:
            # No tool calls, conversation finished
            action_is_valid = True

        # Build response payload uniformly
        response_payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ]
        }

        # Add tool calls if present
        if tool_calls:
            response_payload["choices"][0]["message"]["tool_calls"] = tool_calls

        # Convert to JSON string
        response_payload_json = json.dumps(response_payload, ensure_ascii=False)
        self.logger.debug(f"[FORMAT_RESPONSE] Success! - Payload length: {len(response_payload_json)}")
        info["action_is_valid"] = action_is_valid
        return response_payload_json, info

    @override
    def fetch_agent_request(
        self, index: int, response_payload: Optional[str] = None, timeout: float = None
    ) -> RunSessionResponse:
        try:
            result = self.agent_manager.anti_call_llm(index, response_payload, timeout=timeout)
            return RunSessionResponse(exit_code=0, output=result)
        except Exception as e:
            return RunSessionResponse(exit_code=1, failure_reason=str(e))

    @override
    def start_agent(self, prompt: str) -> RunSessionResponse:
        try:
            if not self.is_environment_available:
                return RunSessionResponse(
                    exit_code=1,
                    failure_reason=f"Environment is not available: {self.initialization_error}",
                )

            if not self.is_initialized:
                return RunSessionResponse(
                    exit_code=1,
                    failure_reason=f"Sandbox/agent is not initialized: {self.initialization_error}",
                )
            self._wait_model_service_ready(timeout=60, interval=1)
            self.agent_manager.start_agent(prompt=prompt)
            return RunSessionResponse(exit_code=0, output="Agent started Successfully")
        except Exception as e:
            return RunSessionResponse(exit_code=1, failure_reason=str(e))

    def _wait_model_service_ready(self, timeout: int = 60, interval: int = 1) -> None:
        import time

        deadline = time.time() + timeout
        last_output = ""

        while time.time() < deadline:
            try:
                result = asyncio.run(
                    self.sandbox.arun(
                        cmd="curl -sf --connect-timeout 1 http://localhost:8080/health",
                        session=self.agent_session_name,
                    )
                )
                if result.exit_code == 0:
                    self.logger.info("[MODEL_SERVICE] Ready: localhost:8080/health")
                    return
                last_output = result.output
            except Exception as e:
                last_output = str(e)

            time.sleep(interval)

        raise RuntimeError(f"Model service not ready after {timeout}s, last_output={last_output}")

    @override
    def process_model_response(
        self, response: str, agent_timeout_sec: int, task_id: str, task_name: str = ""
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError("process_model_response can't be used in SandboxManager V2")

    @override
    def stop_sandbox(self):
        if self.agent_manager is not None:
            try:
                self.agent_manager.close(timeout=10.0)
            except Exception as e:
                self.logger.error(f"agent_manager.close failed: {e}")

        super().stop_sandbox()
