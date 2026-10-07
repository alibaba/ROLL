# ROLL Tinker OpenAI Runtime

This uv project is intentionally minimal. It exists only to run:

```bash
/root/ROLL/roll/pipeline/tinker_backend_runtime/openai_runtime/runtime_pipeline_openai.py
```

It should not include ROLL training, vLLM, Ray, or torch dependencies.

Bootstrap:

```bash
cd /root/ROLL/roll/pipeline/tinker_backend_runtime/openai_runtime
uv sync --python 3.12
```

Use this interpreter from the Tinker runtime config:

```yaml
runtime:
  workdir: /root/ROLL
  launch_command:
    - /root/ROLL/roll/pipeline/tinker_backend_runtime/openai_runtime/.venv/bin/python
    - /root/ROLL/roll/pipeline/tinker_backend_runtime/openai_runtime/runtime_pipeline_openai.py
  env:
    PYTHONPATH: /root/ROLL
```

## Optional HTTP poller

This is an optional lightweight poller for an external OpenAI-compatible HTTP
model endpoint. It is not the native ROLL GPU engine used by the cookbook
rollout. The current cookbook flow uses
`roll/pipeline/tinker_backend_runtime/runtime_pipeline.py` with native ROLL workers and ROCK ModelService.

See the [runtime integration guide](../../../../examples/tinker_backend_runtime/README.md)
or its [Chinese instructions](../../../../examples/tinker_backend_runtime/README_zh.md)
for the public download, GPU environment, and backend lifecycle.
