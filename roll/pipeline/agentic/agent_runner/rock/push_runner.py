import uuid
from typing import Any, Dict

from rock.sdk.job import Job

from roll.pipeline.agentic.agent_runner.base import EpisodeResult
from roll.pipeline.agentic.agent_runner.rock.rock_agent_runner import RockAgentRunner


class PushModeRunner(RockAgentRunner):
    """Push-mode Rock agent runner.

    The Harbor agent calls back to Roll's ProxyServer through ALB/ingress
    to perform LLM inference.  ``Job.run()`` blocks until the agent finishes.
    """

    def run_job(self, seed: int) -> EpisodeResult:
        """Load data, submit a Harbor job in push mode, and return the result."""
        data_item = self._load_data_item(seed)
        if data_item is None:
            return EpisodeResult(status="NoData", score=0.0)

        instance_id = data_item.get("extra_info", {}).get("instanceid", "unknown")
        job_id = "harbor_job_{}_{}".format(instance_id, uuid.uuid4().hex[:8])

        task_config = self._build_task_config_from_data_item(data_item)
        task_config["infer_callback_url"] = self.infer_callback_url

        self.logger.info(
            "Submitting Rock Harbor job: {} with agent: {}".format(
                job_id, task_config["llm_config"]["model_name"]
            )
        )

        metrics = self._run_async_job(
            self._run_harbor_job(task_config, job_id), job_id, instance_id,
        )
        return self._build_result_from_metrics(metrics)

    async def _run_harbor_job(self, task_config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Run a Harbor job (submit + wait) and return extracted metrics."""
        config = self._build_job_config(task_config, job_id)

        self.logger.info("[Harbor] Running job with config: {}".format(config))

        job = Job(config=config)
        self.logger.info("[Harbor] Job instance created, submitting...")
        await job.submit()
        result = await job.wait()
        self.logger.info("[Harbor] Job completed, result: {}".format(result))

        instance_id = task_config.get("data_config", {}).get("instance_id", "unknown")
        metrics = self._extract_metrics(result, job_id, instance_id)

        try:
            sandbox = job._job_client.trials[0].sandbox
            pass_rate = await self._read_pass_rate_from_sandbox(sandbox, config)
            if pass_rate is not None:
                metrics["pass_rate"] = pass_rate
        except Exception as e:
            self.logger.warning(f"[Harbor] Failed to read report.json: {e}")

        return metrics
