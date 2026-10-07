# Public local Tinker runtime

This package integrates Tinker sampling and training with ROLL. The runtime type is `ROLL`, the runtime configuration section is `tinker_runtime`, and an embedded engine configuration uses `roll`. The GPU runtime runs from this checkout using the existing training and inference strategies.

The installed `rl-rock` SDK/backend uses a separate control-plane environment.
The original Tinker cookbook coordinates task environments through ROCK
ModelService; this ROLL runtime consumes the model sampling and training
actions. Keep the control-plane and GPU interpreters separate.

## Install the GPU runtime

Start with the Python 3.12 / PyTorch 2.11 / vLLM 0.23 / CUDA 13 base image
specified in the quick-start, with `uv` installed. From the ROLL root:

```bash
bash examples/tinker_backend_runtime/setup_public_runtime.sh \
  --roll-root "$PWD" \
  --venv "$PWD/.venv-tinker-runtime" \
  --python /usr/bin/python3
```

The script inherits the base image's CUDA packages, installs supplemental dependencies from
[requirements_tinker.txt](../../requirements_tinker.txt), and installs SGLang's tool parser with `--no-deps`. It checks
that protected GPU distributions have not changed and runs CPU import checks.
`setup-public-runtime.json` in the venv records versions and the CUDA library
environment to use when launching the runtime. These checks do not run a model.

For cloning both repositories, installing the SDK/backend, downloading public
assets, configuring local paths, and running the original cookbook rollout and
PPO/PPO+KL/GRPO training, use the single
[Tinker quick-start](../../../ROCK/docs/tinker-quick-start.md). That relative
link assumes sibling `ROCK` and `ROLL` checkouts. Run commands, validation
boundaries, and results are maintained there.

## Runtime interfaces

[runtime_pipeline.py](../../roll/pipeline/tinker_backend_runtime/runtime_pipeline.py)
is launched as a module with `python -m
roll.pipeline.tinker_backend_runtime.runtime_pipeline`, avoiding the adjacent
`types.py` shadowing Python's standard library.

Structured chat prompts use the model tokenizer's chat template, including
tools. Sampling preserves prompt token IDs, generated token IDs, logprobs,
raw decoded text, and parsed tool calls. A token-verified terminal transport
EOS is removed only from the response text; the underlying sampling evidence
remains intact. Model paths must identify an existing local model or a public
model source; internal `OPENLM_HUB` downloads are rejected.

GPU sampling alone does not establish a SWE-bench fix or a training reward.
The cookbook's task environment and verifier determine those results; follow
the quick-start for the complete ModelService workflow and cleanup checks.
