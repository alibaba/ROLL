"""Opt-in CPU test against a running RL-Insight server and Tempo.

Set RL_INSIGHT_SERVER_URL (server API) and ROLL_TEST_TEMPO_URL (Tempo query API).
The subprocesses use real Ray, tracker, checkpoint state and resume code.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
import requests


METRIC = "actor/test_loss"
RESULT_PREFIX = "ROLL_RL_INSIGHT_EXPERIMENT="


def run_driver(root, checkpoint, project, phase):
    import ray
    import rl_insight

    from roll.configs.base_config import TransferBackendArguments
    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline

    class MetricDriver(RLVRPipeline):
        def _initialize(self):
            # No model workers in the CPU integration. Only the replay-only
            # phase needs the attributes the real RLVR loop reads on exit.
            self.generate_schedulers = {}
            self.val_dataset = None

        def _run(self):
            if phase == "replay-only":
                return super()._run()
            for step in range(self.state.step + 1, self.pipeline_config.max_steps):
                metrics = {f"{METRIC}@mean": step + 0.125}
                self.state.step = step
                self.state.log_history.append(metrics)
                self.do_checkpoint(global_step=step)
                self.tracker.log(metrics, step=step)

    config = SimpleNamespace(
        seed=1,
        num_nodes=1,
        num_gpus_per_node=0,
        checkpoint_config={},
        output_dir=str(root),
        save_steps=2,
        to_dict=dict,
        track_with="rl_insight",
        tracker_kwargs={"project": project, "name": "cpu-resume"},
        transfer_backend=TransferBackendArguments(backend_name=None),
        resume_from_checkpoint=checkpoint if phase != "train" else False,
        auto_resume=False,
        async_pipeline=False,
        max_steps=5 if phase == "resume" else 3,
    )
    ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False, object_store_memory=96 * 1024**2)
    try:
        # No model workers to reserve CPUs for. The CPU platform has no
        # device RNG API. WorkerState's metric JSON is saved and loaded as-is.
        with (
            patch("roll.pipeline.base_pipeline.ResourceManager"),
            patch("roll.utils.worker_state.WorkerState.save_rng_state"),
        ):
            driver = MetricDriver(config)
            if phase != "train":
                assert driver.state.step == 2
            context = rl_insight.get_context()
            assert context is not None and context.otel_traces_enabled
            driver.run()
    finally:
        ray.shutdown()
    print(RESULT_PREFIX + context.experiment_name, flush=True)


def wait_for_metrics(tempo_url, project, experiment, expected):
    query = (
        '{resource.service.name = "rl_insight_monitor"'
        ' && span."monitor.trace_segment" = "metric_log"'
        f' && span.project = "{project}" && span.experiment_name = "{experiment}"'
        f' && span."metric.name" = "{METRIC}"}}'
        ' | select(span.global_step, span."metric.name", span."metric.value")'
    )
    deadline = time.monotonic() + 45
    result = None
    while time.monotonic() < deadline:
        response = requests.get(f"{tempo_url.rstrip('/')}/api/search", params={"q": query}, timeout=5)
        response.raise_for_status()
        result = response.json()
        points = set()
        for trace in result.get("traces") or []:
            for group in trace["spanSets"]:
                for span in group["spans"]:
                    attributes = {a["key"]: a["value"] for a in span["attributes"]}
                    points.add(
                        (
                            attributes["metric.name"]["stringValue"],
                            int(attributes["global_step"]["intValue"]),
                            attributes["metric.value"]["doubleValue"],
                        )
                    )
        if points == expected:
            print(json.dumps({"experiment": experiment, "points": sorted(points)}, sort_keys=True))
            return
        time.sleep(0.5)
    pytest.fail(f"Tempo did not return {sorted(expected)} for {experiment}: {result}")


@pytest.mark.skipif(not os.environ.get("ROLL_TEST_TEMPO_URL"), reason="Requires a local RL-Insight/Tempo stack")
def test_checkpoint_resume_preserves_steps_and_flushes_tail(tmp_path):
    project = f"roll-it-{uuid4().hex}"
    tempo_url = os.environ["ROLL_TEST_TEMPO_URL"]
    checkpoint = tmp_path / "train" / "pipeline" / "checkpoint-2"
    experiments = set()
    for phase, last_step in (("train", 2), ("resume", 4), ("replay-only", 2)):
        root = tmp_path / phase
        output = subprocess.run(
            [sys.executable, __file__, str(root), str(checkpoint), project, phase],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "ROLL_LOG_DIR": str(root / "logs")},
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert output.returncode == 0, output.stdout + output.stderr
        (experiment,) = (
            line.removeprefix(RESULT_PREFIX) for line in output.stdout.splitlines() if line.startswith(RESULT_PREFIX)
        )
        assert experiment not in experiments
        experiments.add(experiment)
        wait_for_metrics(
            tempo_url, project, experiment, {(METRIC, step, step + 0.125) for step in range(last_step + 1)}
        )


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), *sys.argv[2:])
