"""Pull-mode Harbor runner using Rock SDK's ModelService as the LLM transport.

Communication direction: Roll -> Rock (sandbox).
Roll actively drives each inference step via ModelService.anti_call_llm(),
instead of waiting for the agent to call back through ALB/ingress.
"""

import asyncio
import json
import shlex
import uuid
from typing import Any, Dict, Optional

import httpx
from omegaconf import DictConfig

from rock.sdk.job import Job
from rock.sdk.job.operator import Operator
from rock.sdk.job.trial.harbor import HarborTrial
from rock.sdk.sandbox.model_service.base import ModelService, ModelServiceConfig

from roll.pipeline.agentic.agent_runner.base import EpisodeResult
from roll.pipeline.agentic.agent_runner.rock.rock_agent_runner import RockAgentRunner

# Default port avoids 8080 which is occupied by other services in the sandbox.
_DEFAULT_MODEL_SERVICE_PORT = 28080


class ModelServiceHarborTrial(HarborTrial):
    """HarborTrial variant that installs and starts ModelService in the sandbox.

    ``on_sandbox_ready()`` is called by JobExecutor after ``sandbox.start()``
    but before ``trial.setup()``, so the sandbox is available and api_base can
    be patched before ``setup()`` writes the Harbor YAML to disk.
    """

    def __init__(self, config, model_service_port: int = _DEFAULT_MODEL_SERVICE_PORT):
        super().__init__(config)
        self._model_service: Optional[ModelService] = None
        self._model_service_port = model_service_port
        self._sandbox_host_ip: Optional[str] = None

    async def on_sandbox_ready(self, sandbox) -> None:
        """Install and start ModelService; patch api_base before setup() writes Harbor YAML."""
        await super().on_sandbox_ready(sandbox)

        obs = await sandbox.arun("hostname -I 2>/dev/null | awk '{print $1}'")
        self._sandbox_host_ip = obs.output.strip() or getattr(sandbox, "host_ip", None)

        if self._sandbox_host_ip and getattr(self, "_config", None) and self._config.agents:
            self._config.agents[0].kwargs["api_base"] = (
                f"http://{self._sandbox_host_ip}:{self._model_service_port}/v1"
            )

        ms_config = ModelServiceConfig(
            enabled=True,
            type="local",
            install_cmd=f"pip install {shlex.quote('rl-rock[model-service]')} --timeout 600",
            start_cmd=(
                f"rock model-service start --type local"
                f" --host 0.0.0.0 --port {self._model_service_port}"
            ),
            watch_agent_cmd=(
                f"rock model-service watch-agent --pid ${{pid}}"
                f" --host 127.0.0.1 --port {self._model_service_port}"
            ),
        )
        self._model_service = ModelService(sandbox=sandbox, config=ms_config)

        await self._model_service.install()
        await self._model_service.start()

    @property
    def model_service(self) -> Optional[ModelService]:
        return self._model_service

    @property
    def sandbox_host_ip(self) -> Optional[str]:
        return self._sandbox_host_ip


class ModelServiceOperator(Operator):
    """Custom operator that creates a ModelServiceHarborTrial."""

    def __init__(self, model_service_port: int = 8080):
        self._model_service_port = model_service_port

    def apply(self, config) -> list:
        return [ModelServiceHarborTrial(config, model_service_port=self._model_service_port)]


class PullModeRunner(RockAgentRunner):
    """Pull-mode Rock agent runner.

    Roll polls the sandbox via ``ModelService.anti_call_llm()`` to serve each
    LLM request the agent makes, rather than exposing an HTTP endpoint that
    the agent calls back through ALB/ingress.
    """

    def __init__(self, base_url: str, env_id: int, env_config: DictConfig, **kwargs):
        super().__init__(base_url, env_id, env_config, **kwargs)
        self._model_service_port: int = env_config.get("config", {}).get(
            "model_service_port", _DEFAULT_MODEL_SERVICE_PORT
        )

    def run_job(self, seed: int) -> EpisodeResult:
        """Load data, submit a Harbor job in pull mode, and return the result."""
        data_item = self._load_data_item(seed)
        if data_item is None:
            return EpisodeResult(status="NoData", score=0.0)

        instance_id = data_item.get("extra_info", {}).get("instanceid", "unknown")
        job_id = "ms_harbor_job_{}_{}".format(instance_id, uuid.uuid4().hex[:8])

        task_config = self._build_task_config_from_data_item(data_item)

        self.logger.info(f"[ModelServiceRunner] Submitting Pull-mode job: {job_id}")

        metrics = self._run_async_job(
            self._run_pull_job(task_config, job_id), job_id, instance_id,
        )
        return self._build_result_from_metrics(metrics)

    async def _run_pull_job(self, task_config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Core Pull-mode execution: submit -> watch_agent -> inference_loop -> wait."""
        operator = ModelServiceOperator(model_service_port=self._model_service_port)
        config = self._build_job_config(task_config, job_id)

        config.agents[0].kwargs["api_base"] = (
            f"http://127.0.0.1:{self._model_service_port}/v1"
        )

        job = Job(config=config, operator=operator)

        self.logger.info(f"[ModelServiceRunner] Submitting job {job_id}...")
        await job.submit()

        trial_client = job._job_client.trials[0]
        trial: ModelServiceHarborTrial = trial_client.trial
        model_service = trial.model_service

        if model_service is None:
            raise RuntimeError(f"[ModelServiceRunner] ModelService not initialised for job {job_id}")

        harbor_pid = str(trial_client.pid)
        self.logger.info(f"[ModelServiceRunner] Starting watch_agent for pid={harbor_pid}")
        asyncio.ensure_future(model_service.watch_agent(pid=harbor_pid))

        instance_id = task_config.get("data_config", {}).get("instance_id", "unknown")
        self.logger.info(f"[ModelServiceRunner] Starting inference loop for {instance_id}")
        async with httpx.AsyncClient(timeout=3600.0) as http_client:
            await self._inference_loop(model_service, http_client)

        self.logger.info("[ModelServiceRunner] Inference loop done, waiting for Harbor result...")
        result = await job.wait()

        metrics = self._extract_metrics(result, job_id, instance_id)

        try:
            pass_rate = await self._read_pass_rate_from_sandbox(trial_client.sandbox, config)
            if pass_rate is not None:
                metrics["pass_rate"] = pass_rate
        except Exception as e:
            self.logger.warning(f"[ModelServiceRunner] Failed to read report.json: {e}")

        return metrics

    async def _inference_loop(self, model_service: ModelService, http_client: httpx.AsyncClient) -> None:
        """Drive LLM inference for the agent via the ModelService file-based protocol."""
        self.logger.info("[InferenceLoop] Waiting for first LLM request from agent...")
        current_index = 0
        current_response_payload: Optional[str] = None

        max_retries = 3
        retry_delay = 1.0

        while True:
            raw_output = None
            for attempt in range(max_retries):
                try:
                    raw_output = await model_service.anti_call_llm(
                        index=current_index,
                        response_payload=current_response_payload,
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"[InferenceLoop] anti_call_llm error at index {current_index} "
                            f"(attempt {attempt + 1}/{max_retries}), retrying in {retry_delay * (2 ** attempt):.1f}s: {e}"
                        )
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        self.logger.error(
                            f"[InferenceLoop] anti_call_llm failed after {max_retries} attempts "
                            f"at index {current_index}: {e}"
                        )
            if raw_output is None:
                break

            raw_output_stripped = raw_output.strip() if raw_output else ""

            if not raw_output_stripped:
                await asyncio.sleep(0.5)
                continue

            if "SESSION_END" in raw_output_stripped:
                self.logger.info(
                    f"[InferenceLoop] SESSION_END received after {current_index} request(s)"
                )
                break

            request_dict = self._parse_llm_request(raw_output_stripped, current_index + 1)
            if request_dict is None:
                self.logger.warning(
                    f"[InferenceLoop] Unparseable request at index {current_index + 1}, "
                    "sending error response"
                )
                response_dict = self._make_error_response("Failed to parse request")
            else:
                try:
                    resp = await http_client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=request_dict,
                        headers={"Authorization": f"Bearer {self.env_id}"},
                    )
                    resp.raise_for_status()
                    response_dict = resp.json()
                except Exception as e:
                    self.logger.error(
                        f"[InferenceLoop] ProxyServer request error at index {current_index + 1}: {e}"
                    )
                    response_dict = self._make_error_response(str(e))

            current_index += 1
            current_response_payload = json.dumps(response_dict, ensure_ascii=False)

        self.logger.info("[InferenceLoop] Exited.")

    def _parse_llm_request(self, raw_output: str, index: int) -> Optional[Dict[str, Any]]:
        """Parse the request JSON returned by ModelClient.anti_call_llm()."""
        try:
            return json.loads(raw_output)
        except Exception as e:
            self.logger.warning(f"[InferenceLoop] Could not parse request at index {index}: {e}")
            return None

    @staticmethod
    def _make_error_response(error_msg: str) -> Dict[str, Any]:
        """Build a minimal OpenAI-compatible error response."""
        return {
            "id": "error",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"[Roll inference error] {error_msg}"},
                    "finish_reason": "stop",
                }
            ],
        }
