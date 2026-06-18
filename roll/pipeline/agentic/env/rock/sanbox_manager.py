import asyncio
import ast
import json
import os
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from roll.pipeline.agentic.tools.iflow import iflow_config
from roll.pipeline.agentic.tools.iflow.cli_tool import IFlowCLITool
from roll.utils.logging import get_logger

try:
    from xrl.sdk.common.constants import Constants
    from xrl.sdk.core.config import XRLConfig
    from xrl.sdk.sandbox.client import Sandbox
    from xrl.sdk.sandbox.config import SandboxConfig as XRLSandboxConfig
    from xrl.sdk.sandbox.request import CreateBashSessionRequest, BashAction
except ImportError:
    print("XRL SDK not available. Make sure it's installed.")
    pass

class SandboxConfig(XRLConfig):
    image: str = "for-code-interpreter-registry-vpc.cn-hangzhou.cr.aliyuncs.com/chatos/python:3.11"
    auto_clear_seconds: int = 60 * 60
    route_key: Optional[str] = None
    startup_timeout: float = 360
    memory: str = "8g"
    cpus: float = 2
    base_url: str = "https://xrl.alibaba-inc.com"

    
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
    
    IFLOW_SYSINFO_COMMAND_FAILED = "iflow_sysinfo_command_failed"
    IFLOW_SYSINFO_PARSE_FAILED = "iflow_sysinfo_parse_failed"
    IFLOW_SYSINFO_EXCEPTION = "iflow_sysinfo_exception"


class RunSessionResponse:
    """Response object for sandbox session operations"""
    def __init__(self):
        self.exit_code = None
        self.output = None
        self.failure_reason = None





class SandboxManager:
    """
    Unified sandbox and session management utility.
    Handles environment initialization, session management, and integrates with IFlowCLITool.
    """
    def __init__(self,
                 sandbox_image: str,
                 logger,
                 xrl_authorization: str = "",
                 sandbox_base_url: str = "https://xrl.alibaba-inc.com",
                 run_type: str = "iflow-cli",
                 iflow_base_url: str = "",
                 iflow_api_key: str = "",
                 iflow_search_api_key: str = "",
                 iflow_selected_auth_type: str = "",
                 agent_version: str = "0.0.1",
                 run_region: str = "",
                 debug: bool = False,
                 default_timeout: float = 60.0):
        
        self.sandbox = None
        self.sandbox_image = sandbox_image
        self.logger = logger
        self.xrl_authorization = xrl_authorization
        self.sandbox_base_url = sandbox_base_url
        
        self.run_type = run_type
        self.iflow_base_url = iflow_base_url
        self.iflow_api_key = iflow_api_key
        self.iflow_search_api_key = iflow_search_api_key
        self.iflow_selected_auth_type = iflow_selected_auth_type
        self.agent_version = agent_version
        self.run_region = run_region
        self.debug = debug
        
        self.active_sessions = {}
        self.is_initialized = False
        self.agent_session_name = "agent"
        self.test_session_name = "test"
        
        self.max_retry = 3
        self.backoff = 2.0
        
        self.image_id = sandbox_image
        self.auto_clear_seconds = 60 * 60 
        self.default_timeout = default_timeout
        
        self.failure_mode = FailureMode.NONE
        self.run_status = RunStatus.SUCCESS
        self.error_messages = []
        
        self.is_environment_available = False
        self.initialization_error = None
        
        self.iflow_tool = IFlowCLITool(
            model_name="",
            api_key=self.iflow_api_key,
            base_url=self.iflow_base_url,
            auth_type=self.iflow_selected_auth_type,
            search_api_key=self.iflow_search_api_key,
            debug=self.debug,
            logger=self.logger
        )
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
                        wait_time = 2.0 * attempt
                        time.sleep(wait_time)
            except Exception as e:
                if attempt < max_init_attempts:
                    wait_time = 2.0 * attempt
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
                self.logger.error(f"[SANDBOX_START] Failed! - Failed to initialize sandbox environment")
                self.failure_mode = FailureMode.SANDBOX_START_FAILED
                return False, sandbox_ip, "Failed to start sandbox"
        except Exception as e:
            self.logger.error(f"[SANDBOX_START] Failed! - Error initializing sandbox environment: {e}")
            self.failure_mode = FailureMode.SANDBOX_START_FAILED
            return False, sandbox_ip, "Failed to start sandbox"
        
        
        is_alive_response = asyncio.run(self.sandbox.is_alive())
        if not is_alive_response.is_alive:
            self.logger.error(f"[SANDBOX_START] Failed! - Sandbox is not alive")
            self.failure_mode = FailureMode.SANDBOX_START_FAILED
            return False, sandbox_ip, "sandbox_util is not alive"
        self.sandbox_ip = sandbox_ip
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
        self.logger.info(f"[AGENT_INSTALL] START - Installing IFlowCLITool")
        try:
            success, message = self._install_agent(self.agent_session_name)
            if success:
                self.is_initialized = True
                self.logger.info(f"[AGENT_INSTALL] Success! - Sandbox and sessions initialized successfully")
            else:
                self.logger.error(f"[AGENT_INSTALL] Failed! - Agent installation failed: {message}")
                self.failure_mode = FailureMode.AGENT_INSTALLATION_FAILED
                return False, sandbox_ip, f"Agent installation failed: {message}"
        except Exception as e:
            self.logger.error(f"[AGENT_INSTALL] Failed! - Error during sandbox initialization: {e}")
            self.failure_mode = FailureMode.AGENT_INSTALLATION_FAILED
            return False, sandbox_ip, f"Agent installation failed: {str(e)}"
        
        return True, sandbox_ip, ""


    def start_sandbox(self, max_retry: int = 3, backoff: float = 2.0):
        """Start a sandbox instance"""
        for attempt in range(1, max_retry + 1):
            try:
                start = time.time()
                if self.run_region == "sg":
                    base_url = "http://xrl-sandbox-global.alibaba-inc.com"
                else:
                    base_url = self.sandbox_base_url
                config = SandboxConfig(
                    image=self.image_id,
                    xrl_authorization=self.xrl_authorization,
                    auto_clear_seconds=self.auto_clear_seconds,
                    startup_timeout=360,
                    base_url=base_url
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
                if attempt == max_retry:
                    return False, None
                time.sleep(backoff * attempt)
        return False, None

    def create_session(self, session: str, max_retry: int = 3, backoff: float = 2.0):
        """Create a session in the sandbox"""
        for attempt in range(1, max_retry + 1):
            try:
                asyncio.run(
                    self.sandbox.create_session(CreateBashSessionRequest(session=session, startup_source=["/root"
                                                                                                          "/.bashrc"],
                                                                         env_enable=True, env={"HOME": "/root","IFLOW_ENV":"train","DISABLE_SEND_PV":"1",
                                                                                               "HF_ENDPOINT":"https://hf-mirror.com"})))
                return True
            except Exception as e:
                self.logger.error(
                    f"[{attempt}/{max_retry}] image_id:{self.image_id}, session:{session} create_session e:{e}, sandbox_id:{self.sandbox.sandbox_id}")
                if attempt == max_retry:
                    return False
                time.sleep(backoff * attempt)
        return False

    def run_in_session(self, command: str, session: str, max_retry: int = 3, backoff: float = 2.0, timeout: float = None):
        """
        Run a command in a session with retry logic for errors and empty outputs.
        """
        self.logger.debug(f"[RUN_SESSION] START - image_id:{self.image_id}, sandbox_ip: {self.sandbox_ip}, session:{session}, command: {json.dumps(command, ensure_ascii=False)}, sandbox_id:{self.sandbox.sandbox_id}")
        
        response = RunSessionResponse()
        last_error_msg = ""
        
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout
        
        for attempt in range(1, max_retry + 1):
            try:
                self.logger.debug(f"[RUN_SESSION] Attempt [{attempt}/{max_retry}] - Executing command in session '{session}' with timeout {timeout}s")
                
                async def run_with_timeout():
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
                
                command_succeeded = response.exit_code == 0
                has_output = response.output and response.output.strip()
                
                self.logger.debug(f"[RUN_SESSION] Attempt [{attempt}/{max_retry}] Result - exit_code: {response.exit_code}, output_length: {len(response.output) if response.output else 0}, has_content: {bool(has_output)}")
                
                should_retry = False
                retry_reason = ""
                if "kill" in command:
                    self.logger.debug(f"[RUN_SESSION] Command contains 'kill', not retrying regardless of outcome")
                    return response
                if not command_succeeded:
                    should_retry = True
                    retry_reason = f"Command failed with exit_code: {response.exit_code}"
                elif not has_output and "-t" in command:
                    should_retry = True
                    retry_reason = "Command with -t succeeded but returned empty output"
                
                if not should_retry:
                    self.logger.debug(f"[RUN_SESSION] SUCCESS - Command executed successfully on attempt {attempt}/{max_retry}")
                    return response
                
                if attempt == max_retry and should_retry:
                    self.logger.error(f"[RUN_SESSION] FAILED - All {max_retry} attempts exhausted. Final reason: {retry_reason}, last_error_msg: {last_error_msg}, response: {json.dumps(response.output[:200], ensure_ascii=False)}")
                    return response
                
                if should_retry:
                    wait_time = backoff * attempt
                    time.sleep(wait_time)
                    
            except asyncio.TimeoutError:
                timeout_msg = f"Command execution timed out after {timeout} seconds"
                if attempt == max_retry:
                    self.logger.error(f"[RUN_SESSION] FAILED - All {max_retry} attempts timed out. Timeout: {timeout}s, Sandbox ID: {self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}, command: {command}")
                    response.exit_code = -1
                    response.output = timeout_msg
                    response.failure_reason = "TIMEOUT"
                    return response
                wait_time = backoff * attempt
                time.sleep(wait_time)
            except Exception as exc:
                last_error_msg = str(exc)
                if "/bin/bash: line 1: " in last_error_msg:
                    response.exit_code = -1
                    response.output = last_error_msg
                    response.failure_reason = last_error_msg
                    return response
                self.logger.error(f"[RUN_SESSION] EXCEPTION - Attempt [{attempt}/{max_retry}] failed with exception: {last_error_msg}")
                if attempt == max_retry:
                    self.logger.error(f"[RUN_SESSION] FAILED - All {max_retry} attempts failed due to exceptions. Final error: {last_error_msg}")
                    response.exit_code = -1
                    response.output = last_error_msg
                    response.failure_reason = last_error_msg
                    return response
                wait_time = backoff * attempt
                time.sleep(wait_time)
        
        self.logger.error(f"[RUN_SESSION] UNEXPECTED - Reached end of retry loop unexpectedly")
        return response

    def stop_sandbox(self):
        """Stop the sandbox instance"""
        try:
            if self.sandbox is None:
                return True
            asyncio.run(self.sandbox.stop())
            return True
        except Exception as e:
            self.logger.error(f"image_id:{self.image_id}, stop_sandbox e:{e}, sandbox_id:{self.sandbox.sandbox_id}")
            return False

    def upload_file(self, file_path: Union[str, Path], target_path: str, max_retry: int = 3, backoff: float = 2.0):
        """Upload a file to the sandbox"""
        for attempt in range(1, max_retry + 1):
            self.logger.debug(
                f"[upload_file, {attempt}/{max_retry}] image_id:{self.image_id}, file_path:{file_path} target_path: {target_path}, sandbox_id:{self.sandbox.sandbox_id}")
            try:
                response = asyncio.run(self.sandbox.aupload(str(file_path), target_path))
                return response.success, response.message
            except Exception as exc:
                self.logger.error(
                    f"image_id:{self.image_id}, file_path:{file_path} target_path: {target_path}, upload failed: {str(exc)}, "
                    f"sandbox_id:{self.sandbox.sandbox_id}")
                if attempt == max_retry:
                    return False, f"upload_file exp:{str(exc)}"
                time.sleep(backoff * attempt)
        return False, f"upload_file failed"

    def run_session_with_timeout(self, session_name: str, command: str, timeout, output_file, interval=10):
        """Run a command with timeout and save output to a file"""
        try:
            start_time = time.time()
            end_time = start_time + timeout
            content = ''

            response = self.run_in_session(command=f"nohup {command} < /dev/null > {output_file} 2>&1 &", session=session_name)
            if response.exit_code != 0:
                if "511" in response.output or "/bin/bash: line 1: " in response.output:
                    self.logger.warning(f"HTTP 511 error or syntax error detected, attempting file-based command execution workaround")
                    try:
                        response, output_file = self._run_command_via_file(command=command, session=session_name)
                        
                        if response.exit_code == 0:
                            self.logger.info(f"File-based command execution succeeded")
                        else:
                            self.logger.warning(f"File-based command execution also failed")
                            return RunStatus.FAILED, "run command failed: " + response.output
                    except Exception as file_exc:
                        self.logger.error(f"File-based workaround failed: {str(file_exc)}")
                        return RunStatus.FAILED, "run command failed: " + str(file_exc)
                else:       
                    return RunStatus.FAILED, "run command failed: " + response.output
                
            pid = self._extract_pid(response.output)
            if len(pid) == 0:
                time.sleep(6)
                content = self.read_content(session_name, output_file)
                return RunStatus.SUCCESS, content


            while time.time() < end_time:
                content = self.read_content(session_name, output_file)
                if not self.is_process_running(session_name, pid):
                    content = self.read_content(session_name, output_file)
                    return RunStatus.SUCCESS, content

                if time.time() >= end_time:
                    content = self.read_content(session_name, output_file)
                    return RunStatus.TIMEOUT, content
                time.sleep(interval)
        except Exception as e:
            self.logger.error(
                f"run_session_with_timeout exception, sandbox_id:{self.sandbox.sandbox_id}, sandbox_ip: {self.sandbox_ip}, command:{command}, exp:{str(e)}")
            return RunStatus.UNKNOWN_ERROR, "run command exception: " + str(e)
        return RunStatus.TIMEOUT, "run command timeout"

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

    def read_content(self, session_name, output_file):
        """Read content from a file in the sandbox"""
        res = self.run_in_session(command=f"cat {output_file}", session=session_name)
        if res.exit_code == 0:
            return res.output
        else:
            print(f"cat {output_file} error, sandbox_id:{self.sandbox.sandbox_id}")
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
                
                self.logger.debug(f"[FILE_WORKAROUND] Script made executable successfully")
                
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

    def execute_command(self, session_name: str, command: str) -> RunSessionResponse:
        """
        Execute a command in the specified session with session management.
        """
        if not self.is_initialized:
            response = RunSessionResponse()
            response.exit_code = -1
            response.output = "Sandbox environment not initialized"
            response.failure_reason = "ENVIRONMENT_NOT_INITIALIZED"
            return response
            
        if session_name not in self.active_sessions:
            if not self.create_managed_session(session_name):
                response = RunSessionResponse()
                response.exit_code = -1
                response.output = f"Failed to create session '{session_name}'"
                response.failure_reason = "SESSION_CREATION_FAILED"
                return response
        
        try:
            self.active_sessions[session_name]["last_used"] = time.time()
            
            response = self.run_in_session(
                command=command,
                session=session_name,
                max_retry=self.max_retry,
                backoff=self.backoff
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error executing command in session '{session_name}': {e}")
            response = RunSessionResponse()
            response.exit_code = -1
            response.output = f"Error executing command: {str(e)}"
            response.failure_reason = "COMMAND_EXECUTION_ERROR"
            return response

    def get_session_info(self, session_name: str) -> Optional[Dict]:
        """
        Get information about a session.
        """
        return self.active_sessions.get(session_name)

    def list_sessions(self) -> List[str]:
        """
        List all active sessions.
        """
        return list(self.active_sessions.keys())

    def cleanup_session(self, session_name: str) -> bool:
        """
        Clean up a specific session.
        """
        try:
            if session_name in self.active_sessions:
                del self.active_sessions[session_name]
                self.logger.debug(f"Session '{session_name}' cleaned up")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up session '{session_name}': {e}")
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

    def _install_agent(self, session_name: str) -> Tuple[bool, str]:
        """Install and configure the agent in the sandbox"""
        try:
            response = self.run_in_session("mkdir -p  /installed-agent", session_name)
            if response.exit_code != 0:
                print(response)
                return False, "Failed to create installation directory"
            
            set_up_install_script = iflow_config.iflow_set_up_install_script_roll
            # Qwen3-Coder是先随便设的一个值，用不到
            settings = iflow_config.get_iflow_setting_template(self.iflow_selected_auth_type,
                                                               self.iflow_api_key,
                                                               self.iflow_base_url,
                                                               "Qwen3-Coder",
                                                               self.iflow_search_api_key)
            
            is_success, message = self._upload_settings(set_up_install_script,
                                              "/installed-agent", "install-agent.sh")
            if not is_success:
                return False, f"Failed to upload installation script: {message}"
            
            run_status, result = self.run_session_with_timeout(
                self.agent_session_name,
                "bash /installed-agent/install-agent.sh",
                300,
                "install.txt"
            )
            
            if run_status != RunStatus.SUCCESS:
                return False, f"Installation failed: {run_status}"
            
            settings = iflow_config.get_iflow_setting_template(self.iflow_selected_auth_type,
                                                               self.iflow_api_key,
                                                               self.iflow_base_url,
                                                               "Qwen3-Coder",
                                                               self.iflow_search_api_key)
            if settings:
                is_success, message = self._upload_settings(settings, "~/.iflow", "settings.json")
                is_success, message = self._upload_settings(settings, "/root/.iflow", "settings.json")
                if not is_success:
                    return False, f"Failed to upload settings: {message}"
            
            self.logger.debug("Agent installation completed successfully")
            
            version_commond = "iflow -v"
            response = self.run_in_session(version_commond, self.agent_session_name)
            if response.exit_code != 0:
                return False, f"Failed to get agent version: {response.output}"
            else:
                self.logger.debug(f"!!!!!!!!!!!!!Agent version: {response.output}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Error during agent installation: {e}"
            self.logger.error(error_msg)
            return False, error_msg

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

    def process_model_response(self, response: str, agent_timeout_sec: int, task_id: str, task_name: str = "") -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process model response and return environment step results.
        """
        self.logger.info(f"[MODEL_RESPONSE] START - Processing model response for task: {task_name}")
        
        self.failure_mode = FailureMode.NONE
        self.error_messages.clear()
        
        reward = 0.0
        terminated = False
        truncated = True
        observation = ""
        is_valid = True
        
        if "</think>" in response: 
            response = response.split("</think>")[-1].strip()
            
        if "<tool_call>" in response and "</tool_call>" in response:
            try:
                self.logger.info(f"[TOOL_PARSE] START - Parsing tool call from model response for task_name: {task_name}")
                is_parsed, iflow_cmd = self.iflow_tool.execute_action(response, self.agent_session_name)
                
                if not is_parsed:
                    error_msg = "Could not parse tool call from response"
                    self.logger.error(f"[TOOL_PARSE] Failed! - {error_msg}")
                    self.failure_mode = FailureMode.TOOL_CALL_PARSE_FAILED
                    self.error_messages.append(f"Tool parse error: {error_msg}")
                    observation = f"Error: {error_msg}."
                    is_valid = False
                else:
                    self.logger.info(f"[TOOL_PARSE] Success! - Tool call parsed successfully")
                    
                    self.logger.info(f"[TOOL_EXEC] START - Executing iflow command: {iflow_cmd[:100]}...")
                    run_status, result = self.run_session_with_timeout(self.agent_session_name, iflow_cmd, 360, "command.txt")
                    observation = ""
                    if run_status == RunStatus.SUCCESS:
                        try:
                            observation, is_valid = self.iflow_tool._process_tool_response(result)
                            if not is_valid:
                                error_msg = f"Tool response processing failed"
                                self.failure_mode = FailureMode.TOOL_RESPONSE_PROCESSING_FAILED
                                self.error_messages.append(f"Tool response processing error: {error_msg}")
                                self.logger.error(f"[TOOL_EXEC] Failed! - {error_msg}, observation: {json.dumps(observation, ensure_ascii=False)}")
                            else:
                                self.logger.info(f"[TOOL_EXEC] Success! - Tool execution completed successfully")
                        except Exception as process_exc:
                            error_msg = f"Exception during tool response processing: {str(process_exc)}"
                            self.failure_mode = FailureMode.TOOL_RESPONSE_PROCESSING_FAILED
                            self.error_messages.append(f"Tool response processing exception: {error_msg}")
                            self.logger.error(f"[TOOL_EXEC] Failed! - {error_msg}, observation so far: {json.dumps(observation, ensure_ascii=False)}")
                            observation = f"Error processing tool response: {str(process_exc)}"
                            is_valid = False
                    else: 
                        if run_status == RunStatus.TIMEOUT:
                            error_msg = f"Tool execution timed out"
                            self.failure_mode = FailureMode.TOOL_EXECUTION_TIMEOUT
                        elif run_status == RunStatus.FAILED:
                            error_msg = f"Tool execution failed"
                            self.failure_mode = FailureMode.TOOL_EXECUTION_FAILED
                        elif run_status == RunStatus.EXCEPTION:
                            error_msg = f"Tool execution encountered exception"
                            self.failure_mode = FailureMode.TOOL_EXECUTION_EXCEPTION
                        else:
                            error_msg = f"Unknown tool execution status: {run_status}"
                            self.failure_mode = FailureMode.TOOL_EXECUTION_FAILED
                            self.error_messages.append(f"Unknown execution status: {error_msg}")
                        self.logger.error(f"[TOOL_EXEC] Failed! - {error_msg}")
                        self.error_messages.append(f"Tool execution failed: {error_msg}")
                        observation = f"Error: {error_msg}. Result: {result if result else 'No output'}"
                        is_valid = False
            except Exception as e:
                error_msg = f"Exception during model response processing: {str(e)}"
                self.failure_mode = FailureMode.MODEL_RESPONSE_PROCESSING_EXCEPTION
                self.error_messages.append(f"Model response processing exception: {error_msg}")
                self.logger.error(f"[MODEL_RESPONSE] Failed! - {error_msg}")
                observation = f"Error processing model response: {str(e)}"
                is_valid = False
        else:
            terminated = True
            truncated = False
            self.logger.info(f"[MODEL_RESPONSE] No tool call found in response, treating as final answer")
            
        info = {
            "action_is_valid": is_valid,
            "success": is_valid
        }
        
        self.logger.info(f"[MODEL_RESPONSE] Completed - terminated: {terminated}, valid: {is_valid}, failure_mode: {self.failure_mode}")
        return observation, reward, terminated, truncated, info

    def _build_command(self, instruction: str) -> str:
        """Build the command to run the agent"""
        escaped_instruction = shlex.quote(instruction)
        
        if self.run_type == "iflow-cli":
            return f"iflow -y -p {escaped_instruction}"
        elif self.run_type == "qwen-code":
            return f"qwen -y -p {escaped_instruction}"
        elif self.run_type == "claude-code":
            return f"claude --verbose --output-format stream-json -p {escaped_instruction}"
        else:
            return f"echo 'Unsupported run type: {self.run_type}'"

    def _extract_execution_info(self, text: str) -> tuple:
        """Extract execution info from the agent output"""
        if text is None:
            return "", ""
            
        import re
        pattern = r'<Execution Info>(.*?)</Execution Info>'
        match = re.search(pattern, text, re.DOTALL)
        
        extracted_content = None
        if match:
            extracted_content = match.group(1).strip()
            cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        else:
            cleaned_text = text

        return extracted_content, cleaned_text

    def run_tests(self, test_files: List[str], test_timeout_sec: int = 60, task_name: str = "") -> dict:
        """Run tests for the task"""
        self.logger.info(f"[TEST_SESSION] START - Creating test session: {self.test_session_name}")
        test_output = ""
        is_success = self.create_session(session=self.test_session_name)
        if not is_success:
            error_msg = "Failed to create test session"
            self.logger.error(f"[TEST_SESSION] Failed! - {error_msg}")
            self.failure_mode = FailureMode.TEST_SESSION_CREATION_FAILED
            self.error_messages.append(f"Test session creation error: {error_msg}")
            return False, FailureMode.TEST_SESSION_CREATION_FAILED, test_output
        else:
            self.logger.info(f"[TEST_SESSION] Success! - Test session created")
            
        test_dir = '/tests'
        response = self.run_in_session(f"mkdir -p {test_dir}", self.test_session_name)
        if response.exit_code != 0:
            error_msg = f"Failed to create test directory: {response.output}"
            self.logger.error(f"[TEST_SESSION] Failed! - {error_msg}")
            self.failure_mode = FailureMode.TEST_DIRECTORY_CREATION_FAILED
            self.error_messages.append(f"Test directory creation error: {error_msg}")
            return False, FailureMode.TEST_DIRECTORY_CREATION_FAILED, test_output
        else:
            self.logger.info(f"[TEST_SESSION] Success! - Test directory created")
            
        for test_file_path in test_files:
            test_path = Path(test_file_path)
            if not test_path.exists():
                self.logger.warning(f"Test path not found: {test_file_path}")
                continue
            
            if test_path.is_dir():
                tests_dir = f"{test_path}/{task_name}/tests"
                if os.path.isdir(tests_dir):
                    for test_file_name in os.listdir(tests_dir):
                        test_file_path = os.path.join(tests_dir, test_file_name)
                        if os.path.exists(test_file_path):
                            is_success, message = self.upload_file(test_file_path, f"{test_dir}/{test_file_name}")
                            if not is_success:
                                error_msg = f"Failed to upload test file {test_file_name}: {message}"
                                self.logger.error(error_msg)
                                self.failure_mode = FailureMode.TEST_FILE_UPLOAD_FAILED
                                self.error_messages.append(f"Test file upload error: {error_msg}")
                                return False, FailureMode.TEST_FILE_UPLOAD_FAILED, test_output
                            self.logger.debug(f"Successfully uploaded test file: {test_file_name}")
                else:
                    self.logger.warning(f"Tests path is not a directory: {tests_dir}")
                run_tests_path = f"{test_path}/{task_name}/run-tests.sh"
                if os.path.exists(run_tests_path):
                    is_success, message = self.upload_file(run_tests_path, f"{test_dir}/run-tests.sh")
                    if not is_success:
                        error_msg = f"Failed to upload test file run-tests.sh: {message}"
                        self.logger.error(error_msg)
                        self.failure_mode = FailureMode.TEST_FILE_UPLOAD_FAILED
                        self.error_messages.append(f"Test file upload error: {error_msg}")
                        return False, FailureMode.TEST_FILE_UPLOAD_FAILED, test_output
                    self.logger.debug(f"Successfully uploaded test file: {run_tests_path}")
            else:
                is_success, message = self.upload_file(str(test_path), f"{test_dir}/{test_path.name}")
                if not is_success:
                    error_msg = f"Failed to upload test file: {message}"
                    self.logger.error(error_msg)
                    self.failure_mode = FailureMode.TEST_FILE_UPLOAD_FAILED
                    self.error_messages.append(f"Test file upload error: {error_msg}")
                    return False, FailureMode.TEST_FILE_UPLOAD_FAILED, test_output
                self.logger.debug(f"Successfully uploaded test file: {test_path.name}")
        self.logger.info(f"[TEST_SESSION] Success! - Test Files uploaded successfully")

        test_command = f"bash {test_dir}/run-tests.sh"
        run_status, test_output = self.run_session_with_timeout(
            self.test_session_name,
            test_command,
            360,
            "test.txt"
        )
        self.logger.info(f"✅[RUN_TESTS] Completed - Test run status: {run_status}, test_output: {json.dumps(test_output, ensure_ascii=False)}...")
        if run_status != RunStatus.SUCCESS:
            if run_status == RunStatus.TIMEOUT:
                error_msg = f"Test execution timed out"
                self.failure_mode = FailureMode.TEST_TIMEOUT
                self.error_messages.append(f"Test timeout error: {error_msg}")
                return False, FailureMode.TEST_TIMEOUT, test_output
            else:
                error_msg = f"Test execution failed with status: {run_status}"
                self.failure_mode = FailureMode.UNKNOWN_TEST_ERROR
                self.error_messages.append(f"Test execution error: {error_msg}")
                return False, FailureMode.UNKNOWN_TEST_ERROR, test_output
        
        with open("test_output.txt", "w") as f:
            f.write(test_output)
        is_resolved = self._parse_test_results(test_output)
        
        return is_resolved, "", test_output

    def _parse_test_results(self, test_output: str) -> bool:
        """Parse test results to determine if the task is resolved"""
        if "All tests passed" in test_output:
            return True
        if "PASS" in test_output and "FAIL" not in test_output:
            return True
        return False

    def get_messages(self, question: str):
        """
        Get concatenated messages using iflow-cli framework.
        """
        try:
            escaped_question = shlex.quote(question)
            if escaped_question.startswith("-"):
                escaped_question = escaped_question[1:].strip()
            command = f"iflow --sysinfo {escaped_question}"

            response = self.run_in_session(command, self.agent_session_name)
            
            if response.exit_code != 0:
                error_msg = f"iflow --sysinfo command failed: {response.output}"
                self.logger.error(error_msg)
                self.failure_mode = FailureMode.IFLOW_SYSINFO_COMMAND_FAILED
                self.error_messages.append(f"IFlow sysinfo command error: {error_msg}")
                return [{"role": "user", "content": question}], response.output
                
            try:
                response_output = response.output.split("[\r\n  {\r\n    \"role\": \"system")[-1]
                response_output = "[\r\n  {\r\n    \"role\": \"system" + response_output
                if isinstance(response_output, str):
                    try:
                        messages = json.loads(response_output)
                    except:
                        messages = ast.literal_eval(response_output)
                else:
                    messages = response_output
                self.logger.debug(f"iflow-cli returned {len(messages)} messages")
                return messages, ""
            except Exception as e:
                error_msg = f"Failed to parse iflow sysinfo response: {str(e)}"
                self.logger.error(f"Load messages failed: {response_output}")
                self.failure_mode = FailureMode.IFLOW_SYSINFO_PARSE_FAILED
                self.error_messages.append(f"IFlow sysinfo parse error: {error_msg}")
                return [{"role": "user", "content": question}], f"IFLOW_SYSINFO_PARSE_FAILED, {str(e)}"
        except Exception as e:
            error_msg = f"Exception in get_messages: {str(e)}"
            self.logger.error(f"Error in get_messages: {error_msg}")
            self.failure_mode = FailureMode.IFLOW_SYSINFO_EXCEPTION
            self.error_messages.append(f"IFlow sysinfo exception: {error_msg}")
            return [{"role": "user", "content": question}], f"IFLOW_SYSINFO_EXCEPTION, {str(e)}"
