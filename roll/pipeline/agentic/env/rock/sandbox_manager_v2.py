import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpcore
import httpx
from rock.actions import BashAction, BashObservation, Command, CommandResponse, CreateBashSessionRequest

# rl-rock SDK Imports
from rock.sdk.sandbox.client import Sandbox
from rock.sdk.sandbox.config import SandboxConfig
from rock.sdk.sandbox.speedup.types import SpeedupType

# Internal Project Dependencies
from roll.pipeline.agentic.env.rock.agent_manager import AgentManager
from roll.pipeline.agentic.tools.action_parser import ActionParser, Qwen3CoderActionParser


logging.getLogger("httpx").setLevel(logging.ERROR)


class RunStatus:
    """Status codes for sandbox operations"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"
    CREATE_BOX = "create_box"
    SANDBOX_START_FAILED = "sandbox_start_failed"
    INFERENCE = "inference"
    INFERENCE_FAILED = "inference_failed"
    TEST = "test"
    TEST_FAILED = "test_failed"
    EXCEPTION = "exception"


class FailureMode:
    """Failure modes for terminal-bench operations"""
    NONE = "none"
    UNSET = "unset"
    AGENT_TIMEOUT = "agent_timeout"
    UNKNOWN_AGENT_ERROR = "unknown_agent_error"
    TEST_TIMEOUT = "test_timeout"
    UNKNOWN_TEST_ERROR = "unknown_test_error"
    PARSE_ERROR = "parse_error"
    SANDBOX_START_FAILED = "sandbox_start_failed"
    SANDBOX_CREATE_SESSION_FAILED = "sandbox_create_session_failed"
    RUN_SANDBOX_COMMAND_FAILED = "run_sandbox_command_failed"
    RUN_SANDBOX_UPLOAD_FAILED = "run_sandbox_upload_failed"
    RUN_SANDBOX_EXCEPTION = "run_sandbox_exception"
    RUN_CLI_TYPE_NOT_SUPPORT = "run_cli_type_not_support"
    AGENT_INSTALLATION_FAILED = "agent_installation_failed"
    IMAGE_NOT_FOUND_EXCEPTION = "image_not_found_exception"

    TOOL_CALL_PARSE_FAILED = "tool_call_parse_failed"
    TOOL_EXECUTION_TIMEOUT = "tool_execution_timeout"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    TOOL_EXECUTION_EXCEPTION = "tool_execution_exception"
    TOOL_RESPONSE_PROCESSING_FAILED = "tool_response_processing_failed"
    MODEL_RESPONSE_PROCESSING_EXCEPTION = "model_response_processing_exception"

    TEST_SESSION_CREATION_FAILED = "test_session_creation_failed"
    TEST_DIRECTORY_CREATION_FAILED = "test_directory_creation_failed"
    TEST_FILE_UPLOAD_FAILED = "test_file_upload_failed"

    START_SCRIPT_FAILED = "start_script_failed"

    IFLOW_SYSINFO_COMMAND_FAILED = "iflow_sysinfo_command_failed"
    IFLOW_SYSINFO_PARSE_FAILED = "iflow_sysinfo_parse_failed"
    IFLOW_SYSINFO_EXCEPTION = "iflow_sysinfo_exception"

    MODEL_SERVICE_START_FAILED = "model_service_start_failed"
    MODEL_SERVICE_ANTI_CALL_LLM_FAILED = "model_service_anti_call_llm_failed"
    AGENT_START_FAILED = "agent_start_failed"


class RunSessionResponse:
    """Response object for sandbox session operations"""
    def __init__(self, exit_code: int =None, output: str =None, failure_reason: str =None):
        self.exit_code = exit_code
        self.output = output
        self.failure_reason = failure_reason

    def __str__(self):
        return f"exit_code: {self.exit_code}, output: {self.output}, failure_reason: {self.failure_reason}"


class SandboxManagerV2:
    """
    Unified sandbox and session management utility.
    Handles environment initialization, session management, and integrates with IFlowCLITool.
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


    def  _initialize_sandbox_with_times(self):
        self.logger.info(f"[SANDBOX_INIT] START - Image ID: {self.image_id}")
        self.sandbox_id = ""

        max_init_attempts = 3
        is_success = False
        sandbox_ip = None
        reason = ""
        for attempt in range(1, max_init_attempts + 1):
            self.logger.info(f"[SANDBOX_INIT] Attempt [{attempt}/{max_init_attempts}] - Initializing sandbox")
            try:
                is_success, sandbox_ip, reason = self._initialize_sandbox()
                if is_success and sandbox_ip:
                    self.logger.info(f"[SANDBOX_INIT] Success on attempt {attempt}! - Sandbox started successfully with IP: {sandbox_ip}, sandbox_id: {self.sandbox_id}")
                    break
                else:
                    if attempt < max_init_attempts:
                        wait_time = 120.0 * attempt
                        time.sleep(wait_time)
            except Exception as e:
                self.logger.error(f"[{attempt}/{max_init_attempts}] image_id:{self.image_id} create_session e:{e}, sandbox_id:{self.sandbox.sandbox_id}")
                if attempt < max_init_attempts:
                    wait_time = 120.0 * attempt
                    time.sleep(wait_time)

        self.sandbox_ip = sandbox_ip
        if is_success and sandbox_ip:
            self.logger.info(f"[SANDBOX_INIT] Final Success! - Sandbox started successfully with IP: {sandbox_ip}, sandbox_id: {self.sandbox_id}")
            self.is_environment_available = True
        else:
            self.logger.error(f"[SANDBOX_INIT] Final Failure! - Failed to start sandbox after {max_init_attempts} attempts: {reason}, sandbox_image: {self.image_id}, sandbox_ip: {self.sandbox_ip}, sandbox_id: {self.sandbox_id}")
            self.is_environment_available = False
            self.initialization_error = f"Failed to initialize sandbox after {max_init_attempts} attempts: {reason}, sandbox_ip: {sandbox_ip}, sandbox_image: {self.image_id}"


    def _initialize_sandbox(self):
        """Initialize sandbox and create sessions during environment construction"""
        sandbox_ip = None
        self.logger.info(f"[SANDBOX_START] START - Starting sandbox with image: {self.image_id}")
        try:
            success, sandbox_ip = self.start_sandbox(
                max_retry=self.max_retry,
                backoff=self.backoff
            )
            if success and sandbox_ip:
                self.logger.info(f"[SANDBOX_START] Success! - Sandbox environment initialized with IP: {sandbox_ip}, sandbox_id: {self.sandbox_id}")
            else:
                self.logger.error("! - Failed to initialize sandbox environment")
                self.failure_mode = FailureMode.SANDBOX_START_FAILED
                time.sleep(20.0)
                return False, sandbox_ip, "Failed to start sandbox"
        except Exception as e:
            self.logger.error(f"[SANDBOX_START] Failed! - Error initializing sandbox environment: {e}")
            self.failure_mode = FailureMode.SANDBOX_START_FAILED
            time.sleep(20.0)
            return False, sandbox_ip, "Failed to start sandbox"

        is_alive_response = asyncio.run(self.sandbox.is_alive())
        if not is_alive_response.is_alive:
            self.logger.error("[SANDBOX_START] Failed! - Sandbox is not alive")
            self.failure_mode = FailureMode.SANDBOX_START_FAILED
            return False, sandbox_ip, "sandbox_util is not alive"
        self.sandbox_ip = sandbox_ip

        self.error_suffix = f"sandbox_ip: {self.sandbox_ip}, sandbox_id: {self.sandbox_id}, sandbox_image: {self.image_id}"

        # 创建session
        self.logger.info(f"[SESSION_CREATE] START - Creating session: {self.agent_session_name}")
        try:
            success = self.create_session(session=self.agent_session_name)
            if success:
                self.active_sessions[self.agent_session_name] = {
                    "created_at": time.time(),
                    "last_used": time.time()
                }
                self.logger.info(f"[SESSION_CREATE] Success! - Session '{self.agent_session_name}' created successfully")
            else:
                self.logger.error(f"[SESSION_CREATE] Failed! - Failed to create session '{self.agent_session_name}'")
                self.failure_mode = FailureMode.SANDBOX_CREATE_SESSION_FAILED
                return False, sandbox_ip, f"Failed to create session '{self.agent_session_name}'"
        except Exception as e:
            self.logger.error(f"[SESSION_CREATE] Failed! - Error creating session '{self.agent_session_name}': {e}")
            self.failure_mode = FailureMode.SANDBOX_CREATE_SESSION_FAILED
            return False, sandbox_ip, f"Failed to create session '{self.agent_session_name}'"

        # 初始化iflow-cli
        self.logger.info("[AGENT_INSTALL] START - Installing IFlowCLITool")
        try:
            success, message = self._install_agent(self.agent_session_name)
            if success:
                self.is_initialized = True
                self.logger.info("[AGENT_INSTALL] Success! - Sandbox and sessions initialized successfully")
            else:
                self.logger.error(f"[AGENT_INSTALL] Failed! - Agent installation failed: {message}, {self.error_suffix}")
                self.failure_mode = FailureMode.AGENT_INSTALLATION_FAILED
                return False, sandbox_ip, f"Agent installation failed: {message}, {self.error_suffix}"
        except Exception as e:
            self.logger.error(f"[AGENT_INSTALL] Failed! - Error during sandbox initialization: {e}, {self.error_suffix}")
            self.failure_mode = FailureMode.AGENT_INSTALLATION_FAILED
            return False, sandbox_ip, f"Agent installation failed: {str(e)}"

        # 启动初始化服务
        if self.start_script:
            self.logger.info("[AGENT_START] START - Starting start.sh")
            try:
                success = self._run_start_script(self.agent_session_name, self.start_script)
                if success:
                    self.logger.info("[AGENT_START] Success! - run start.sh successfully")
                else:
                    self.logger.warning("⚠️ [AGENT_START] Warning! - run start.sh failed")
            except Exception as e:
                self.logger.warning(f"[AGENT_START] ERROR! - run start.sh failed, error: {e}")

        # 清理沙盒环境中的测试文件
        self.logger.info("[CLEANUP] START - Cleaning up problematic files in sandbox")
        try:
            success, message = self._cleanup_problematic_files(self.agent_session_name)
            if success:
                self.logger.info("[CLEANUP] Success! - Problematic files cleaned up successfully")
            else:
                self.logger.warning(f"⚠️ [CLEANUP] Warning! - Some files could not be cleaned up: {message}")
        except Exception as e:
            self.logger.warning(f"⚠️ [CLEANUP] Warning! - Error during file cleanup: {e}")

            return False, sandbox_ip, f"Agent installation failed: {str(e)}, {self.error_suffix}"

        return True, sandbox_ip, ""

    def start_sandbox(self, max_retry: int = 3, backoff: float = 20.0):
        """Start a sandbox instance"""
        try:
            start = time.time()
            config = SandboxConfig(
                base_url=self.sandbox_base_url,
                image=self.image_id,
                auto_clear_seconds=self.auto_clear_seconds,
                startup_timeout=self.startup_timeout,
                user_id = self.user_id,
                experiment_id=self.experiment_id,
                xrl_authorization=self.xrl_authorization
            )
            sandbox = Sandbox(config)

            asyncio.run(sandbox.start())
            cost = time.time() - start
            self.logger.debug(f"image_id:{self.image_id}, sandbox_id:{sandbox.sandbox_id}, sandbox ip: {sandbox.host_ip},  start sandbox cost:{cost}")
            self.sandbox = sandbox
            self.sandbox_id = sandbox.sandbox_id
            return True, sandbox.host_ip
        except Exception as e:
            self.logger.error(f"image_id:{self.image_id}, start_sandbox e:{e}")
            time.sleep(20.0)
        return False, None

    def create_session(self, session: str, max_retry: int = 3, backoff: float = 20.0):
        """Create a session in the sandbox"""
        for attempt in range(1, max_retry + 1):
            try:
                asyncio.run(
                    self.sandbox.create_session(CreateBashSessionRequest(session=session, startup_source=["/root"
                                                                                                          "/.bashrc"],
                                                                         env_enable=True, env={"HOME": "/root",
                                                                                               "IFLOW_ENV":"train",
                                                                                               "DISABLE_SEND_PV":"1",
                                                                                               "HF_ENDPOINT":"https://hf-mirror.com",
                                                                                               "UV_INDEX_URL":"https://mirrors.aliyun.com/pypi/simple/"})))
                return True
            except Exception as e:
                self.logger.error(
                    f"[{attempt}/{max_retry}] image_id:{self.image_id}, session:{session} create_session e:{e}, sandbox_id:{self.sandbox.sandbox_id}")
                if attempt == max_retry:
                    return False
                time.sleep(backoff * attempt)
        return False

    def run_in_session(self, command: str, session: str, max_retry: int = 3, backoff: float = 10.0, timeout: float = None, execute_enable: bool = False):
        """
        Run a command in a session with retry logic for errors and empty outputs.
        """
        self.logger.debug(f"[RUN_SESSION] START - image_id:{self.image_id}, sandbox_ip: {self.sandbox_ip}, session:{session}, command: {json.dumps(command, ensure_ascii=False)}, sandbox_id:{self.sandbox.sandbox_id}")

        response = RunSessionResponse()
        last_error_msg = ""

        if timeout is None:
            timeout = self.default_timeout

        for attempt in range(1, max_retry + 1):
            wait_time = backoff * attempt
            try:
                self.logger.debug(f"[RUN_SESSION] Attempt [{attempt}/{max_retry}] - Executing command in session '{session}' with timeout {timeout}s")

                async def run_with_timeout():
                    if execute_enable:
                        execute_rsp: CommandResponse = await self.sandbox.execute(
                            Command(command=["/bin/bash", "-c", command])
                        )
                        return BashObservation(
                            output=execute_rsp.stdout,
                            exit_code=execute_rsp.exit_code,
                            failure_reason=execute_rsp.stderr,
                        )
                    else:
                        return await self.sandbox.run_in_session(
                            BashAction(session=session, command=command, check="silent"))

                session_ret = asyncio.run(
                    asyncio.wait_for(run_with_timeout(), timeout=timeout))

                response = session_ret
                if session_ret.exit_code is None:
                    response.exit_code = -1
                if session_ret.output is None:
                    response.output = ""
                if session_ret.failure_reason is None:
                    response.failure_reason = ""

                self.logger.debug(f"[RUN_SESSION] Attempt [{attempt}/{max_retry}] Result - exit_code: {response.exit_code}, output_length: {len(response.output) if response.output else 0}")
                self.logger.debug(f"[RUN_SESSION] SUCCESS - Command executed on attempt {attempt}/{max_retry}, sandbox_id: {self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}")
                return response

            except asyncio.TimeoutError:
                timeout_msg = f"Command execution timed out after {timeout} seconds, sandbox_id: {self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}, command: {command}"
                response.exit_code = -1
                response.output = timeout_msg
                response.failure_reason = "TIMEOUT"
                if attempt == max_retry:
                    self.logger.error(f"[RUN_SESSION] FAILED - All {max_retry} attempts timed out. Timeout: {timeout}s, Sandbox ID: {self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}, command: {command}")
                    return response
                self.logger.info(f"[RUN_SESSION] TimeoutError, last_error_msg: {last_error_msg}, Sandbox ID: {self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}")
                if attempt < max_retry:
                    time.sleep(wait_time)
            except Exception as exc:
                # Get detailed error information
                error_type = type(exc).__name__
                error_module = exc.__class__.__module__
                error_args = exc.args if hasattr(exc, 'args') else ()

                # Build comprehensive error message
                if str(exc).strip():
                    last_error_msg = str(exc)
                else:
                    last_error_msg = f"{error_module}.{error_type}: {error_args}"

                # Add additional context for empty ReadError
                if isinstance(exc, (httpx.ReadError, httpcore.ReadError)) and not str(exc).strip():
                    last_error_msg = f"{error_type}: Network read error with no details - args: {error_args}"

                last_error_msg = f"{last_error_msg}, sandbox_id:{self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}, command: {command}"
                response.exit_code = -1
                response.output = last_error_msg
                response.failure_reason = last_error_msg

                self.logger.info(f"traceback.format_exc: {traceback.format_exc()}, last_error_msg: {last_error_msg}")
                if "/bin/bash: line 1: " in last_error_msg:
                    return response
                self.logger.error(f"[RUN_SESSION] EXCEPTION - Attempt [{attempt}/{max_retry}] failed with exception: {last_error_msg}, sandbox_id: {self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}")
                if attempt == max_retry:
                    self.logger.error(f"[RUN_SESSION] FAILED - All {max_retry} attempts failed due to exceptions. Final error: {last_error_msg}, sandbox_id: {self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}")
                    return response
                if attempt < max_retry:
                    self.logger.info(f"[RUN_SESSION] Retrying attempt [{attempt}/{max_retry}] in {wait_time} seconds, sandbox_id: {self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}")
                    time.sleep(wait_time)

        self.logger.error("[RUN_SESSION] UNEXPECTED - Reached end of retry loop unexpectedly")
        return response

    def stop_sandbox(self):
        if self.agent_manager is not None:
            try:
                self.agent_manager.close(timeout=10.0)
            except Exception as e:
                self.logger.error(f"agent_manager.close failed: {e}")

        try:
            if self.sandbox is None:
                return True
            asyncio.run(self.sandbox.stop())
            self.sandbox = None
            return True
        except Exception as e:
            self.logger.error(f"image_id:{self.image_id}, stop_sandbox e:{e}, sandbox_id:{self.sandbox.sandbox_id}")
            return False


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


    def run_session_with_timeout(self, session_name: str, command: str, timeout, output_file, interval=10):
        """Run a command with timeout and save output to a file"""
        try:
            start_time = time.time()
            # Handle None timeout - use a default value of 300 seconds (5 minutes)
            if timeout is None:
                timeout = self.default_timeout
                self.logger.debug(f"[RUN_SESSION] Timeout is None, using default timeout of {timeout} seconds")
            end_time = start_time + timeout
            content = ''
            if command == "":
                return RunStatus.SUCCESS, "Command is empty"
            response = self.run_in_session(
                command=f"nohup {command} < /dev/null > {output_file} 2>&1 &",
                session=session_name,
            )
            if response.exit_code != 0:
                if "511" in response.output or "/bin/bash: line 1: " in response.output:
                    self.logger.warning("HTTP 511 error or syntax error detected, attempting file-based command execution workaround")
                    try:
                        response, output_file = self._run_command_via_file(command=command, session=session_name)

                        if response.exit_code == 0:
                            self.logger.info("File-based command execution succeeded")
                        else:
                            self.logger.warning("File-based command execution also failed")
                            return RunStatus.FAILED, "run command failed: " + response.output
                    except Exception as file_exc:
                        self.logger.error(f"File-based workaround failed: {str(file_exc)}")
                        return RunStatus.FAILED, "run command failed: " + str(file_exc)
                else:
                    return RunStatus.FAILED, "run command failed: " + response.output

            pid = self._extract_pid(response.output)
            if len(pid) == 0:
                time.sleep(1)
                content = self.read_content(session_name, output_file)
                return RunStatus.SUCCESS, content

            while time.time() < end_time:
                if not self.is_process_running(session_name, pid):
                    content = self.read_content(session_name, output_file, True)
                    return RunStatus.SUCCESS, content

                if time.time() >= end_time:
                    content = self.read_content(session_name, output_file, True)
                    return RunStatus.TIMEOUT, content
                time.sleep(interval)
        except Exception as e:
            self.logger.error(
                f"run_session_with_timeout exception, sandbox_id:{self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}, command:{command}, exp:{str(e)}")
            return RunStatus.UNKNOWN_ERROR, "run command exception: " + str(e)
        return RunStatus.TIMEOUT, "run command timeout"

    def run_script_with_timeout(self, session_name: str, script_text: str, timeout, max_retry: int = 3) -> RunSessionResponse:
        """
        Execute shell script text by uploading it as a .sh file and then executing.

        Args:
            session_name: Name of the session to execute in
            script_text: Multi-line shell script text to execute
            timeout: Timeout in seconds
            max_retry: Number of times to retry

        Returns:
            RunSessionResponse object with exit_code, output, and failure_reason
        """
        self.logger.info(f"[RUN_SCRIPT] START - Executing script text in session '{session_name}'")

        # Handle None timeout - use a default value of 300 seconds (5 minutes)
        if timeout is None:
            timeout = self.default_timeout
            self.logger.debug(f"[RUN_SCRIPT] Timeout is None, using default timeout of {timeout} seconds")

        # Generate a unique output file for the script
        timestamp = int(time.time())
        output_file = f"/tmp/script_output_{timestamp}_{hash(script_text) % 10000}.sh.out"

        # Generate a unique script name
        script_hash = hash(script_text) % 10000
        script_path = f"/tmp/script_{timestamp}_{script_hash}.sh"

        # Ensure session exists (session management from execute_command)
        if not self.is_initialized:
            response = RunSessionResponse()
            response.exit_code = -1
            response.output = "Sandbox environment not initialized"
            response.failure_reason = "ENVIRONMENT_NOT_INITIALIZED"
            return response

        try:
            if session_name not in self.active_sessions:
                if not self.create_managed_session(session_name):
                    response = RunSessionResponse()
                    response.exit_code = -1
                    response.output = f"Failed to create session '{session_name}'"
                    response.failure_reason = "SESSION_CREATION_FAILED"
                    return response

            self.active_sessions[session_name]["last_used"] = time.time()

            # Upload the script content as a .sh file (similar to _install_agent approach)
            is_success, message = self._upload_settings(script_text, "/tmp", f"script_{timestamp}_{script_hash}.sh")
            if not is_success:
                response = RunSessionResponse()
                response.exit_code = -1
                response.output = f"Failed to upload script file: {message}"
                response.failure_reason = "SCRIPT_UPLOAD_FAILED"
                return response

            # Make the script executable
            chmod_response = self.run_in_session(f"chmod +x {script_path}", session_name)
            if chmod_response.exit_code != 0:
                response = RunSessionResponse()
                response.exit_code = -1
                response.output = f"Failed to make script executable: {chmod_response.output}"
                response.failure_reason = "SCRIPT_CHMOD_FAILED"
                return response

            # Call run_session_with_timeout which handles nohup and process monitoring
            run_status, result = self.run_session_with_timeout(
                session_name=session_name,
                command=f"bash {script_path}",  # Execute the uploaded script file
                timeout=timeout,
                output_file=output_file,
            )

            # Convert RunStatus and result to RunSessionResponse
            response = RunSessionResponse()
            response.output = result

            if run_status == RunStatus.SUCCESS:
                response.exit_code = 0
            elif run_status == RunStatus.TIMEOUT:
                response.exit_code = 1
                response.failure_reason = "TIMEOUT"
            elif run_status == RunStatus.FAILED:
                response.exit_code = 1
                response.failure_reason = "FAILED"
            else:  # UNKNOWN_ERROR or others
                response.exit_code = -1
                response.failure_reason = run_status

            # Clean up the uploaded script file
            self.run_in_session(f"rm -f {script_path}", session_name)

            return response

        except Exception as e:
            self.logger.error(
                f"run_script_with_timeout exception, sandbox_id:{self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}, script_text:{script_text[:100]}..., exp:{str(e)}")

            # Clean up the uploaded script file on exception
            try:
                self.run_in_session(f"rm -f {script_path}", session_name)
            except:
                pass  # Ignore cleanup errors

            response = RunSessionResponse()
            response.exit_code = -1
            response.output = f"run script exception: {str(e)}"
            response.failure_reason = "EXCEPTION"
            return response

    def _extract_pid(self, output: str) -> str:
        """Extract process ID from command output"""
        lines = output.splitlines()
        if not lines:
            return ""
        last_line = lines[-1].strip()
        import re
        match = re.match(r'\[\d+\]\s+(\d+)$', last_line)
        return match.group(1) if match else ""

    def is_process_running(self, session_name: str, pid):
        """Check if a process is still running"""
        response = self.run_in_session(command=f"kill -0 {pid}", session=session_name)
        return response.exit_code == 0

    def read_content(self, session_name, output_file, execute_enable: bool = False):
        """Read content from a file in the sandbox"""
        cmd = f"head -c {self.head_content_limit} {output_file}"
        res = self.run_in_session(command=cmd, session=session_name, execute_enable=execute_enable)
        if res.exit_code == 0:
            return res.output
        else:
            print(f"{cmd} error, sandbox_id:{self.sandbox.sandbox_id}")
            return ''

    def _run_command_via_file(self, command: str, session: str) -> RunSessionResponse:
        """
        Workaround for 511 errors: write command to file, upload it, then execute
        """
        self.logger.info(f"[FILE_WORKAROUND] START - Attempting file-based execution for command: {command[:100]}...")
        response = RunSessionResponse()
        temp_script_name = f"temp_command_{session}_{int(time.time())}.sh"
        temp_output_file = f"temp_output_{session}_{int(time.time())}.txt"

        try:
            if "tool_calls" in command:
                tool_call = command
                script_content = f"""#!/bin/bash
# Auto-generated script to workaround network issues
# Session: agent, Attempt: 1

set -e

# Store the JSON payload in a variable using a here document
read -r -d '' JSON_PAYLOAD << 'EOF' || true
{tool_call}
EOF

# Execute the command with the JSON payload
nohup iflow -t "$JSON_PAYLOAD" < /dev/null > {temp_output_file} 2>&1 &
"""
            else:
                script_content = f"""#!/bin/bash
# Auto-generated script to workaround network issues
# Session: agent, Attempt: 1

set -e
nohup {command} < /dev/null > {temp_output_file} 2>&1 &
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
                temp_file.write(script_content)
                temp_file_path = temp_file.name

            self.logger.debug(f"[FILE_WORKAROUND] Created temporary script: {temp_file_path}")

            try:
                self.logger.debug(f"[FILE_WORKAROUND] Uploading script to sandbox: {temp_script_name}")
                upload_success, upload_message = self.upload_file(
                    temp_file_path,
                    f"{temp_script_name}"
                )

                if not upload_success:
                    self.logger.error(f"[FILE_WORKAROUND] Failed to upload script file: {upload_message}")
                    response.exit_code = -1
                    response.output = f"Upload failed: {upload_message}"
                    response.failure_reason = "SCRIPT_UPLOAD_FAILED"
                    return response, temp_output_file

                self.logger.debug(f"[FILE_WORKAROUND] Successfully uploaded script to {temp_script_name}")

                chmod_command = f"chmod +x {temp_script_name}"
                self.logger.debug(f"[FILE_WORKAROUND] Making script executable: {chmod_command}")

                chmod_ret = asyncio.run(
                    self.sandbox.run_in_session(
                        BashAction(session=session, command=chmod_command, check="silent")))

                if chmod_ret.exit_code != 0:
                    self.logger.error(f"[FILE_WORKAROUND] chmod command failed: {chmod_ret.output}")
                    response.exit_code = -1
                    response.output = f"chmod failed: {chmod_ret.output}"
                    response.failure_reason = "CHMOD_FAILED"
                    return response, temp_output_file

                self.logger.debug("[FILE_WORKAROUND] Script made executable successfully")

                exec_command = f"bash {temp_script_name}"
                self.logger.info(f"[FILE_WORKAROUND] Executing script via run_session_with_timeout: {exec_command}")
                session_ret = asyncio.run(
                    self.sandbox.run_in_session(
                        BashAction(session=session, command=exec_command, check="silent")))
                response = session_ret
                if session_ret.exit_code is None:
                    response.exit_code = -1
                if session_ret.output is None:
                    response.output = ""
                if session_ret.failure_reason is None:
                    response.failure_reason = ""
                return response, temp_output_file
            except Exception as exec_exc:
                self.logger.error(f"[FILE_WORKAROUND] Failed to execute script via file method: {exec_exc}")
                response.exit_code = -1
                response.output = f"Script execution failed: {str(exec_exc)}"
                response.failure_reason = "SCRIPT_EXECUTION_FAILED"
                return response, temp_output_file


        except Exception as e:
            self.logger.error(f"[FILE_WORKAROUND] File-based command execution failed: {str(e)}")
            response.exit_code = -1
            response.output = f"File method error: {str(e)}"
            response.failure_reason = "FILE_METHOD_ERROR"
            return response, temp_output_file


    def create_managed_session(self, session_name: str) -> bool:
        """
        Create a new managed session in the sandbox with tracking.
        """
        is_alive_response = asyncio.run(self.sandbox.is_alive())
        if not is_alive_response.is_alive:
            print("sandbox_util is not alive")
            return False

        try:
            success = self.create_session(session=session_name)

            if success:
                self.active_sessions[session_name] = {
                    "created_at": time.time(),
                    "last_used": time.time()
                }
                self.logger.debug(f"Session '{session_name}' created successfully")
            else:
                self.logger.error(f"Failed to create session '{session_name}'")

            return success

        except Exception as e:
            self.logger.error(f"Error creating session '{session_name}': {e}")
            return False


    def close(self):
        """Close the sandbox environment and cleanup resources."""
        try:
            self.stop_sandbox()
            self.active_sessions.clear()
            self.is_initialized = False
            self.logger.debug("Sandbox environment closed successfully")

        except Exception as e:
            self.logger.error(f"Error closing sandbox environment: {e}")


    def _run_start_script(self, session_name: str, start_script: str) -> bool:
        """Run the start script in the sandbox"""
        try:
            start_script_path = None
            for test_file_path in self.test_files:
                test_path = Path(test_file_path)
                if not test_path.exists():
                    self.logger.warning(f"Test path not found: {test_file_path}")
                    continue

                if test_path.is_dir():
                    script_path = test_path / self.task_name / start_script
                    if script_path.exists():
                        start_script_path = str(script_path)
                        break

            if not start_script_path:
                self.logger.warning(f"Start script {start_script} not found for task {self.task_name}")
                return False

            pwd_response = self.run_in_session("pwd", session_name)
            if pwd_response.exit_code == 0:
                current_working_dir = pwd_response.output.strip()
            else:
                current_working_dir = "/app"

            target_path = os.path.join(current_working_dir, start_script)
            is_success, message = self.upload_file(start_script_path, target_path)
            if not is_success:
                self.logger.error(f"[RUN_START_SCRIPT] Failed to upload start script: {message}")
                self.failure_mode = FailureMode.START_SCRIPT_FAILED
                return False

            if self.dataset_tag == "safety":
                pass

            chmod_response = self.run_in_session(f"chmod +x {target_path}", session_name)

            bg_command = f"nohup bash {target_path} > start_agent_script.log 2>&1 &"
            bg_response = self.run_in_session(bg_command, session_name)
            time.sleep(5)
            if bg_response.exit_code == 0:
                self.logger.info("[RUN_START_SCRIPT] Start script launched in background successfully")
                # Give the script a moment to start
                time.sleep(2)
                return True
            else:
                self.logger.warning(f"[RUN_START_SCRIPT] Failed to launch start script in background: {bg_response.output}")
                self.failure_mode = FailureMode.START_SCRIPT_FAILED
                return False

        except Exception as e:
            self.logger.error(f"[RUN_START_SCRIPT] Error running start script: {e}")
            self.failure_mode = FailureMode.START_SCRIPT_FAILED
            return False

    def _upload_settings(self, content: str, directory: str, filename: str) -> tuple:
        """Upload settings to the sandbox"""

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file.write('\n')
            temp_filename = temp_file.name
        try:
            is_success, message = self.upload_file(temp_filename, f"{directory}/{filename}")
            return is_success, message
        finally:
            import os
            os.unlink(temp_filename)

    def _cleanup_problematic_files(self, session_name: str) -> Tuple[bool, str]:
        problematic_paths = [
            "/app/tests",
            "/app/run-tests.sh",
            "/app/solution.sh",
            "/app/task.yaml",
            "/app/test.sh",
            "/app/Dockerfile",
            "/app/docker-compose.yaml",
            "/app/setup_apt_source.sh",
            "/app/install_iflow_info.txt",
            "/app/start_agent_script.log"
        ]

        paths_str = " ".join(problematic_paths)
        cleanup_command = f"rm -rf {paths_str} 2>/dev/null || true"

        try:
            self.logger.info(f"[CLEANUP] START - Attempting to remove problematic files: {paths_str}")

            remove_response = self.run_in_session(cleanup_command, session_name)

            check_command = f"ls -la {paths_str} 2>/dev/null | grep -E '^[d-]' || echo 'All files cleaned'"
            check_response = self.run_in_session(check_command, session_name)

            cleanup_results = []
            if check_response.output and "All files cleaned" in check_response.output:
                self.logger.info("[CLEANUP] Success! - All problematic files have been cleaned up")
                cleanup_results = [f"Cleaned: {path}" for path in problematic_paths]
                return True, "; ".join(cleanup_results)
        except Exception as e:
            error_msg = f"Exception during cleanup: {str(e)}"
            self.logger.warning(f"⚠️ [CLEANUP] {error_msg}")
            return False, error_msg


    def start_agent(self, prompt: str) -> RunSessionResponse:
        try:
            self.agent_manager.start_agent(prompt=prompt)
            return RunSessionResponse(exit_code=0, output="Agent started Successfully")
        except Exception as e:
            return RunSessionResponse(exit_code=1, failure_reason=str(e))



    def _compress_directory(self, local_dir: Union[str, Path], archive_name: str) -> bool:
        """
        压缩本地目录为 .tar.gz

        Args:
            local_dir: 要压缩的目录路径
            archive_name: 压缩包文件名（包含.tar.gz后缀）

        Returns:
            bool: 压缩成功返回True，失败返回False
        """
        try:
            self.logger.debug(f"[COMPRESS] START - 正在压缩 {local_dir} -> {archive_name}")

            # 使用 gztar 格式压缩
            base_name = archive_name.replace('.tar.gz', '')
            shutil.make_archive(base_name, 'gztar', local_dir)

            self.logger.debug(f"[COMPRESS] Success! - 压缩完成: {archive_name}")
            return True

        except Exception as e:
            self.logger.error(f"[COMPRESS] Failed! - 压缩目录失败 {local_dir}: {e}")
            return False

    def run_tests(self, test_files: List[str], test_timeout_sec: int = 60, task_name: str = "") -> dict:
        """Run tests for the task - upload tests folder via compressed archive and run-tests.sh separately"""
        self.logger.info(f"[TEST_SESSION] START - Creating test session: {self.test_session_name}")
        test_output = ""
        is_success = self.create_session(session=self.test_session_name)
        if not is_success:
            error_msg = "Failed to create test session"
            self.logger.error(f"[TEST_SESSION] Failed! - {error_msg}")
            self.failure_mode = FailureMode.TEST_SESSION_CREATION_FAILED
            self.error_messages.append(f"Test session creation error: {error_msg}")
            return False, FailureMode.TEST_SESSION_CREATION_FAILED, test_output, ""
        else:
            self.logger.info("[TEST_SESSION] Success! - Test session created")

        test_dir = '/tests'
        response = self.run_in_session(f"mkdir -p {test_dir}", self.test_session_name)
        if response.exit_code != 0:
            error_msg = f"Failed to create test directory: {response.output}"
            self.logger.error(f"[TEST_SESSION] Failed! - {error_msg}")
            self.failure_mode = FailureMode.TEST_DIRECTORY_CREATION_FAILED
            self.error_messages.append(f"Test directory creation error: {error_msg}")
            return False, FailureMode.TEST_DIRECTORY_CREATION_FAILED, test_output, ""
        else:
            self.logger.info("[TEST_SESSION] Success! - Test directory created")

        task_dir = None
        for test_file_path in test_files:
            test_path = Path(test_file_path)
            if not test_path.exists():
                self.logger.warning(f"Test path not found: {test_file_path}")
                continue

            if test_path.is_dir():
                task_test_dir = test_path / task_name
                if task_test_dir.exists():
                    task_dir = task_test_dir
                    break

        tests_dir = task_dir / "tests"
        if tests_dir.exists():
            with tempfile.TemporaryDirectory() as temp_dir:
                archive_name = f"{task_name}_tests.tar.gz"
                archive_path = os.path.join(temp_dir, archive_name)

                if self._compress_directory(str(tests_dir), archive_path):
                    self.logger.info(f"[TEST_COMPRESS] Success! - Compressed {tests_dir} to {archive_path}")

                    # 上传压缩包
                    is_success, message = self.upload_file(archive_path, f"{test_dir}/{archive_name}")
                    if not is_success:
                        error_msg = f"Failed to upload test archive {archive_name}: {message}"
                        self.logger.error(error_msg)
                        self.failure_mode = FailureMode.TEST_FILE_UPLOAD_FAILED
                        self.error_messages.append(f"Test archive upload error: {error_msg}")
                        return False, FailureMode.TEST_FILE_UPLOAD_FAILED, test_output, ""

                    # 记录当前工作目录
                    pwd_response = self.run_in_session("pwd", self.test_session_name)
                    if pwd_response.exit_code == 0:
                        current_working_dir = pwd_response.output.strip()
                        self.logger.info(f"[TEST_SESSION] Current working directory recorded: {current_working_dir}")
                    else:
                        current_working_dir = "/app"
                        self.logger.warning(f"[TEST_SESSION] Failed to get current working directory, using default: {current_working_dir}")

                    extract_response = self.run_in_session(f"cd {test_dir} && tar -xzf {archive_name} && rm {archive_name}", self.test_session_name)
                    if extract_response.exit_code != 0:
                        error_msg = f"Failed to extract test archive: {extract_response.output}"
                        self.logger.error(error_msg)
                        self.failure_mode = FailureMode.TEST_FILE_UPLOAD_FAILED
                        self.error_messages.append(f"Test archive extraction error: {error_msg}")
                        return False, FailureMode.TEST_FILE_UPLOAD_FAILED, test_output, ""

                    self.logger.info("[TEST_UPLOAD] Success! - Tests folder uploaded and extracted successfully")
                else:
                    error_msg = f"Failed to compress test directory: {tests_dir}"
                    self.logger.error(error_msg)
                    self.failure_mode = FailureMode.TEST_FILE_UPLOAD_FAILED
                    self.error_messages.append(f"Test compression error: {error_msg}")
                    return False, FailureMode.TEST_FILE_UPLOAD_FAILED, test_output, ""
        else:
            self.logger.warning(f"Tests directory not found: {tests_dir}")


        run_tests_script = task_dir / "run-tests.sh"

        if run_tests_script.exists():
            is_success, message = self.upload_file(str(run_tests_script), f"{test_dir}/run-tests.sh")
            if not is_success:
                error_msg = f"Failed to upload run-tests.sh: {message}"
                self.logger.error(error_msg)
                self.failure_mode = FailureMode.TEST_FILE_UPLOAD_FAILED
                self.error_messages.append(f"Run-tests script upload error: {error_msg}")
                return False, FailureMode.TEST_FILE_UPLOAD_FAILED, test_output, ""
            self.logger.info("[TEST_UPLOAD] Success! - run-tests.sh uploaded successfully")
        else:
            error_msg = f"run-tests.sh not found: {run_tests_script}"
            self.logger.error(error_msg)
            self.failure_mode = FailureMode.TEST_FILE_UPLOAD_FAILED
            self.error_messages.append(f"Run-tests script not found: {error_msg}")
            return False, FailureMode.TEST_FILE_UPLOAD_FAILED, test_output, ""

        self.logger.info("[TEST_SESSION] Success! - All test files uploaded successfully")

        # 执行测试
        sandbox_run_test_scripts = f"{test_dir}/run-tests.sh"
        chmod_response = self.run_in_session(f"chmod +x {sandbox_run_test_scripts}", self.test_session_name)

        cd_response = self.run_in_session(f"cd {current_working_dir}", self.test_session_name)

        test_command = f"bash {sandbox_run_test_scripts}"
        run_status, test_output = self.run_session_with_timeout(
            self.test_session_name,
            test_command,
            60 * 20,
            "test.txt"
        )
        self.logger.info(f"[RUN_TESTS] Completed - Test run status: {run_status}, test_output: {json.dumps(test_output, ensure_ascii=False)}...")
        if run_status != RunStatus.SUCCESS:
            if run_status == RunStatus.TIMEOUT:
                error_msg = "Test execution timed out"
                self.failure_mode = FailureMode.TEST_TIMEOUT
                self.error_messages.append(f"Test timeout error: {error_msg}")
                return False, FailureMode.TEST_TIMEOUT, test_output, test_command
            else:
                error_msg = f"Test execution failed with status: {run_status}"
                self.failure_mode = FailureMode.UNKNOWN_TEST_ERROR
                self.error_messages.append(f"Test execution error: {error_msg}")
                return False, FailureMode.UNKNOWN_TEST_ERROR, test_output, test_command

        with open("test_output.txt", "w") as f:
            f.write(test_output)
        is_resolved = self._parse_test_results(test_output)

        return is_resolved, "", test_output, test_command

    def _parse_test_results(self, test_output: str) -> bool:
        """Parse test results to determine if the task is resolved"""
        if "SWEBench results starts here" in test_output:
            test_output = test_output.split("SWEBench results starts here")[-1]
        if "short test summary info" in test_output:
            test_output = test_output.split("short test summary info")[-1]
        if "test session starts" in test_output:
            test_output = test_output.split("test session starts")[-1]
        if "All tests passed" in test_output:
            return True
        if "PASSED" in test_output and "FAILED" not in test_output:
            return True
        if "PASS" in test_output and "FAIL" not in test_output:
            return True
        return False

    def run_ground_truth_solution(self, agent_timeout_sec: int, task_id: Union[str, int], task_name: str = "") -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Run ground truth solution for evaluation mode.
        """
        self.logger.info(f"[GROUND_TRUTH] START - Running ground truth solution for task: {task_name}")

        self.failure_mode = FailureMode.NONE
        self.error_messages.clear()

        reward = 0.0
        terminated = True
        truncated = False
        observation = ""
        is_valid = True

        try:
            solution_file_path = None
            for test_file_path in self.test_files:
                test_path = Path(test_file_path)
                if not test_path.exists():
                    self.logger.warning(f"Test path not found: {test_file_path}")
                    continue

                if test_path.is_dir():
                    potential_solution_path = test_path / task_name / "solution.sh"
                    if potential_solution_path.exists():
                        solution_file_path = str(potential_solution_path)
                        break

            if not solution_file_path:
                error_msg = f"Ground truth solution file not found: {solution_file_path}"
                self.logger.error(f"[GROUND_TRUTH] Failed! - {error_msg}")
                self.failure_mode = FailureMode.RUN_SANDBOX_UPLOAD_FAILED
                self.error_messages.append(f"Solution file not found: {error_msg}")
                observation = error_msg
                is_valid = False
                info = {"action_is_valid": is_valid}
                return observation, reward, terminated, truncated, info


            pwd_response = self.run_in_session("pwd", self.agent_session_name)
            if pwd_response.exit_code == 0:
                current_working_dir = pwd_response.output.strip()
            else:
                current_working_dir = "/app"

            target_solution_path = os.path.join(current_working_dir, "solution.sh")
            self.logger.info(f"[GROUND_TRUTH] Uploading solution from {solution_file_path} to {target_solution_path}")
            is_success, message = self.upload_file(solution_file_path, target_solution_path)

            chmod_response = self.run_in_session(f"chmod +x {target_solution_path}", self.agent_session_name)

            run_status, result = self.run_session_with_timeout(
                self.agent_session_name,
                f"bash {target_solution_path}",
                60*40,
                "ground_truth_output.txt"
            )

            if run_status == RunStatus.SUCCESS:
                observation = f"Ground truth solution executed successfully. Output: {result[:1000]}..."
                self.logger.info("[GROUND_TRUTH] Success! - Ground truth solution executed successfully")
                is_valid = True
            elif run_status == RunStatus.TIMEOUT:
                error_msg = f"Ground truth solution execution timed out after {agent_timeout_sec}s"
                self.logger.error(f"[GROUND_TRUTH] Failed! - {error_msg}")
                self.failure_mode = FailureMode.TOOL_EXECUTION_TIMEOUT
                self.error_messages.append(f"Execution timeout: {error_msg}")
                observation = f"{error_msg}. Partial output: {result[:500]}..."
                is_valid = False
            else:
                error_msg = f"Ground truth solution execution failed with status: {run_status}"
                self.logger.error(f"[GROUND_TRUTH] Failed! - {error_msg}")
                self.failure_mode = FailureMode.TOOL_EXECUTION_FAILED
                self.error_messages.append(f"Execution failed: {error_msg}")
                observation = f"{error_msg}. Output: {result[:500]}..."
                is_valid = False

        except Exception as e:
            error_msg = f"Exception during ground truth solution execution: {str(e)}"
            self.logger.error(f"[GROUND_TRUTH] Failed! - {error_msg}")
            self.failure_mode = FailureMode.RUN_SANDBOX_EXCEPTION
            self.error_messages.append(f"Execution exception: {error_msg}")
            observation = error_msg
            is_valid = False

        info = {
            "action_is_valid": is_valid,
            "have_tool_call": False,
            "success": is_valid
        }
        self.logger.info(f"[GROUND_TRUTH] Completed - terminated: {terminated}, valid: {is_valid}, failure_mode: {self.failure_mode}")
        return observation, reward, terminated, truncated, info


    def get_messages_and_tools(self, request_payload: str):
        """
        Get messages and tools from iflow-cli request_payload_json_str.

        Args:
            request_payload: JSON string containing the request payload with messages and tools

        Returns:
            Tuple of (messages, tools, error_message)
        """
        self.logger.debug("[GET_MESSAGES_TOOLS] START - Processing request payload")

        # Parse the request payload JSON
        try:
            request_data = json.loads(request_payload)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse request payload JSON: {str(e)}, request_payload: {request_payload}"
            self.logger.error(f"[GET_MESSAGES_TOOLS] Failed! - {error_msg}, {self.error_suffix}")
            self.failure_mode = FailureMode.IFLOW_SYSINFO_PARSE_FAILED
            self.error_messages.append(f"Request payload parse error: {error_msg}")
            return [], [], error_msg

        # Extract messages and tools from the request data
        assert "messages" in request_data, f"Messages not found in request data, {request_data}"

        messages = request_data.get("messages", [])
        tools = request_data.get("tools", [])
        self.logger.debug(f"[GET_MESSAGES_TOOLS] Success! - Extracted {len(messages)} messages and {len(tools)} tools")
        return messages, tools, ""

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
            self.logger.debug("[FORMAT_RESPONSE] Tool calls detected in response")
            try:
                is_parsed, parsed_tool_calls = self.action_parser.parse_action(response)
                if is_parsed and parsed_tool_calls:
                    tool_calls = parsed_tool_calls
                    # Extract the text content before tool calls
                    content_parts = response.split("<tool_call>")
                    content = content_parts[0] if content_parts else response
                    self.logger.debug("[FORMAT_RESPONSE] Tool calls formatted successfully")
                    action_is_valid = True
                else:
                    # Tool call parsing failed, treat as regular response
                    content = response
                    self.logger.debug("[FORMAT_RESPONSE] Tool call parsing failed, treating as regular response")

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


    def fetch_agent_request(
        self, index: int, response_payload: Optional[str] = None, timeout: float = None
    ) -> RunSessionResponse:
        try:
            result = self.agent_manager.anti_call_llm(index, response_payload, timeout=timeout)
            return RunSessionResponse(exit_code=0, output=result)
        except Exception as e:
            return RunSessionResponse(exit_code=1, failure_reason=str(e))
