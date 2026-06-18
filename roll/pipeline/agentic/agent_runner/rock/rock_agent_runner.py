import json
import os
from typing import Any, Dict, Optional

import ray
from omegaconf import DictConfig
from rock.actions import Command, ReadFileRequest
from rock.sdk.bench import AgentConfig
from rock.sdk.job import JobConfig

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.pipeline.agentic.agent_runner.base import AgentRunner, EpisodeResult
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.pipeline.agentic.proxy.router import create_proxy_router
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger

# Path to the default Harbor service config YAML (relative to repo root).
_DEFAULT_JOB_CONFIG_PATH = "roll/pipeline/agentic/env/rock/harbor_service_config.yaml"


class RockAgentRunner(AgentRunner):
    """Base AgentRunner for Rock SDK sandbox environments.

    Inherits from AgentRunner and adds Rock-specific logic:
    - Harbor job config building
    - Metrics extraction from Harbor JobResult
    - Evaluation report reading (pass rate from SWE-bench)

    Subclasses:
        PushModeRunner  — agent calls back to Roll via ALB/ingress
        PullModeRunner  — Roll polls sandbox via ModelService.anti_call_llm
    """

    def __init__(
        self,
        base_url: str,
        env_id: int,
        env_config: DictConfig,
        worker_config: Optional[EnvManagerConfig] = None,
        **kwargs,
    ):
        super().__init__(base_url, env_id, env_config, worker_config=worker_config, **kwargs)
        self.logger = get_logger()

        config = dict(env_config.get("config", {}))

        # Dataset for data loading (Rock scenarios need dataset to get task instances)
        assert "dataset_name" in config, "RockAgentRunner requires dataset_name in env_config.config"
        dataset_name = config.get("dataset_name", "")
        self.dataset = GlobalDataset.options(
            name=f"{self.mode}_{dataset_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote(dataset_name=dataset_name, mode=self.mode)
        dataset_manager = GlobalDatasetManager.options(
            name=f"{self.mode}_dataset_manager",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()
        ray.get(dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))

        job_config_path = config.get("job_config_path", _DEFAULT_JOB_CONFIG_PATH)
        self._base_job_config = JobConfig.from_yaml(job_config_path)

        self.listen_port = env_config.get("proxy_port", 8000)
        self.infer_callback_url = self._generate_callback_url()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data_item(self, seed: int) -> Optional[Dict[str, Any]]:
        """Load a data item from the dataset using ``seed``.

        Returns None when the dataset is exhausted.
        """
        if self.dataset is None:
            return None
        data_item: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data_item is not None:
            data_item["seed"] = seed
        return data_item

    # ------------------------------------------------------------------
    # Callback URL
    # ------------------------------------------------------------------

    def _generate_callback_url(self) -> Optional[str]:
        """Build the URL that the sandbox agent uses to reach Roll's ProxyServer."""
        job_name = os.environ.get("TASK_ID")
        if job_name:
            try:
                url = create_proxy_router().get_callback_url(job_name, self.listen_port)
                if url:
                    self.logger.info(f"[Harbor] Generated callback URL: {url}")
                    return url
            except Exception as e:
                self.logger.error(f"Failed to get callback URL: {e}")

        config_url = self.env_config.get("config", {}).get("infer_callback_url")
        if config_url:
            return config_url

        self.logger.info("[Harbor] No remote callback URL available. Agent will use default LLM_BASE_URL.")
        return None

    # ------------------------------------------------------------------
    # Job config
    # ------------------------------------------------------------------

    def _build_job_config(self, task_config: Dict[str, Any], job_id: str) -> JobConfig:
        """Build a Rock SDK JobConfig from task_config."""
        config = self._base_job_config.model_copy(deep=True)
        config.job_name = job_id

        llm_config = task_config.get("llm_config", {})
        runtime_config = task_config.get("runtime_config", {})
        agent_config = task_config.get("agent_config", {})
        data_config = task_config.get("data_config", {})

        config.agents = [
            AgentConfig(
                name="swe-agent-internal",
                model_name="openai/{}".format(llm_config.get("model_name", "default_model")),
                max_timeout_sec=runtime_config.get("task_timeout_sec", 300),
                kwargs={
                    "api_key": llm_config.get("api_key", ""),
                    "api_base": task_config.get("infer_callback_url", ""),
                    "sweagent_config": agent_config.get("scaffold_config", "anthropic"),
                    "max_iterations": agent_config.get("max_iterations", 15),
                    "temperature": llm_config.get("generation_config", {}).get("temperature", 0.99),
                    "max_tokens": llm_config.get("generation_config", {}).get("max_new_tokens", 2048),
                    "num_retries": agent_config.get("num_retries", 4),
                    "tools_parse_function": "function_calling",
                    "full_history": True,
                    "max_observation_length": 10000,
                },
            )
        ]

        if config.datasets:
            config.datasets[0].task_names = [data_config.get("instance_id")]
            config.datasets[0].registry.split = data_config.get("split")
            config.datasets[0].name = data_config.get("dataset")
            config.datasets[0].version = data_config.get("split")

        if config.environment:
            config.environment.env["INSTANCE_ID"] = data_config.get("instance_id")
            config.environment.env["DATASET"] = data_config.get("dataset")
            config.environment.env["SPLIT"] = data_config.get("split")
            config.environment.experiment_id = os.environ["TASK_ID"]

        config.experiment_id = os.environ["TASK_ID"]
        config.verifier.native_config.template.name = f"swe-agent-internal/{data_config.get('dataset')}"

        return config

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _extract_metrics(self, result, job_id: str, instance_id: str) -> Dict[str, Any]:
        """Extract a standardised metrics dict from a Harbor JobResult."""
        metrics: Dict[str, Any] = {
            "job_id": job_id,
            "instance_id": instance_id,
            "status": str(result.status.value if hasattr(result.status, "value") else result.status),
            "exit_code": getattr(result, "exit_code", 0),
            "score": 0.0,
            "agent_exit_reason": "",
            "time_total_sec": 0.0,
            "env_setup_time_ratio": 0.0,
            "agent_setup_time_ratio": 0.0,
            "agent_execution_time_ratio": 0.0,
        }

        if hasattr(result, "trial_results") and result.trial_results:
            trial_result = result.trial_results[0]

            exception_info = getattr(trial_result, "exception_info", None)
            if exception_info:
                metrics["agent_exit_reason"] = str(exception_info)

            verifier_result = getattr(trial_result, "verifier_result", None)
            if verifier_result and getattr(verifier_result, "rewards", None):
                metrics["score"] = verifier_result.rewards.get("reward", 0.0)

            t_total = self._get_duration(trial_result)
            t_env_setup = self._get_duration(getattr(trial_result, "environment_setup", None))
            t_agent_setup = self._get_duration(getattr(trial_result, "agent_setup", None))
            t_logic_exec = self._get_duration(getattr(trial_result, "agent_execution", None))

            metrics["time_total_sec"] = round(t_total, 2)

            if t_total > 0:
                metrics["env_setup_time_ratio"] = round((t_env_setup / t_total) * 100, 2)
                metrics["agent_setup_time_ratio"] = round((t_agent_setup / t_total) * 100, 2)
                metrics["agent_execution_time_ratio"] = round((t_logic_exec / t_total) * 100, 2)

        return metrics

    def _get_duration(self, timing_obj) -> float:
        """Extract elapsed seconds from a timing object or dict."""
        if timing_obj is None:
            return 0.0

        if isinstance(timing_obj, dict):
            start = timing_obj.get("started_at")
            end = timing_obj.get("finished_at")
        else:
            start = getattr(timing_obj, "started_at", None)
            end = getattr(timing_obj, "finished_at", None)

        if not start or not end:
            return 0.0

        try:
            if isinstance(start, str):
                from dateutil import parser

                dt_start = parser.isoparse(start)
                dt_end = parser.isoparse(end)
            else:
                dt_start = start
                dt_end = end

            if dt_start.tzinfo is not None:
                dt_start = dt_start.replace(tzinfo=None)
            if dt_end.tzinfo is not None:
                dt_end = dt_end.replace(tzinfo=None)

            return max(0.0, (dt_end - dt_start).total_seconds())
        except Exception:
            return 0.0

    @staticmethod
    def _extract_tests_status(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pull ``tests_status`` out of a SWE-bench report, tolerant of nesting."""
        if not isinstance(report, dict):
            return None
        if isinstance(report.get("tests_status"), dict):
            return report["tests_status"]
        for value in report.values():
            if isinstance(value, dict) and isinstance(value.get("tests_status"), dict):
                return value["tests_status"]
        return None

    async def _read_pass_rate_from_sandbox(self, sandbox, config) -> Optional[float]:
        """Read the FAIL_TO_PASS pass rate from Harbor's SWE-bench evaluation report.

        Returns:
            FAIL_TO_PASS success ratio in [0.0, 1.0], or None when the report
            is missing or carries no FAIL_TO_PASS tests.
        """
        try:
            job_dir = f"{config.jobs_dir}/{config.job_name}"
            find_result = await sandbox.execute(
                Command(command=["find", job_dir, "-path", "*/verifier/evaluation/report.json"])
            )
            files = [line.strip() for line in (find_result.stdout or "").strip().split("\n") if line.strip()]
            if not files:
                self.logger.info(f"[Harbor][REWARD] no evaluation report.json under {job_dir} -> fallback to binary score")
                return None

            response = await sandbox.read_file(ReadFileRequest(path=files[0]))
            report = json.loads(response.content)
            tests_status = self._extract_tests_status(report)
            if not tests_status:
                self.logger.info(f"[Harbor][REWARD] report.json has no tests_status -> fallback: {files[0]}")
                return None

            f2p = tests_status.get("FAIL_TO_PASS", {})
            p2p = tests_status.get("PASS_TO_PASS", {})
            f2p_success = len(f2p.get("success", []))
            f2p_total = f2p_success + len(f2p.get("failure", []))
            if f2p_total == 0:
                self.logger.info(f"[Harbor][REWARD] report.json has no FAIL_TO_PASS tests -> fallback: {files[0]}")
                return None

            p2p_failure = len(p2p.get("failure", []))
            if p2p_failure > 0:
                self.logger.info(
                    "[Harbor][REWARD] pass_rate=0.0000 (PASS_TO_PASS regressions=%d) FAIL_TO_PASS=%d/%d report=%s",
                    p2p_failure, f2p_success, f2p_total, files[0],
                )
                return 0.0

            pass_rate = f2p_success / f2p_total
            self.logger.info(
                "[Harbor][REWARD] pass_rate=%.4f FAIL_TO_PASS=%d/%d PASS_TO_PASS=%d/%d report=%s",
                pass_rate, f2p_success, f2p_total,
                len(p2p.get("success", [])), len(p2p.get("success", [])) + p2p_failure,
                files[0],
            )
            return pass_rate
        except Exception as e:
            self.logger.warning(f"[Harbor][REWARD] failed to read report.json -> fallback to binary score: {e}")
            return None

    # ------------------------------------------------------------------
    # Task config helpers
    # ------------------------------------------------------------------

    def _build_task_config_from_data_item(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Build common task_config dict from a loaded data item.

        Note: Push-mode callers must add ``infer_callback_url`` to the returned
        dict; Pull mode does not need it.
        """
        instance_id = data_item.get("extra_info", {}).get("instanceid", "unknown")
        return {
            "metadata": {"step_id": 0},
            "data_config": {
                "instance_id": instance_id,
                "dataset": data_item.get("extra_info", {}).get("datasetname", "unknown"),
                "split": data_item.get("extra_info", {}).get("datasettype", "test"),
            },
            "llm_config": {
                "model_name": "glm-5",
                "api_key": str(self.env_id),
                "generation_config": self.worker_config.generating_args.to_dict(),
            },
            "runtime_config": {
                "task_timeout_sec": self.env_config.config.get("task_timeout_sec", 1800),
            },
            "agent_config": {
                "max_iterations": self.env_config.config.get("max_iterations", 15),
                "scaffold_config": self.env_config.config.get("scaffold_config", "anthropic"),
                "num_retries": self.env_config.config.get("num_retries", 4),
            },
        }

    def _build_result_from_metrics(self, metrics: Dict[str, Any]) -> EpisodeResult:
        """Convert the raw metrics dict into an ``EpisodeResult``."""
        return EpisodeResult(
            status=metrics.pop("status", "Unknown"),
            score=float(metrics.pop("score", 0.0)),
            agent_exit_reason=metrics.pop("agent_exit_reason", ""),
            metrics=metrics,
        )

    def _run_async_job(self, coro, job_id: str, instance_id: str) -> Dict[str, Any]:
        """Run an async coroutine on an isolated event loop and return metrics.

        Shared boilerplate for both Push and Pull runners.
        """
        import asyncio
        import concurrent.futures

        try:
            loop = asyncio.new_event_loop()
            private_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            loop.set_default_executor(private_executor)

            try:
                return loop.run_until_complete(coro)
            except Exception as e:
                self.logger.error(f"[RockRunner] Job {job_id} failed: {e}", exc_info=True)
                return {
                    "instance_id": instance_id,
                    "status": "RunnerError",
                    "score": 0.0,
                    "agent_exit_reason": str(e),
                    "job_id": job_id,
                }
            finally:
                private_executor.shutdown(wait=False)
                loop.close()
        except Exception as e:
            self.logger.error(f"[RockRunner] Outer error for job {job_id}: {e}")
            return {"status": "Failed", "score": 0.0, "agent_exit_reason": str(e)}
