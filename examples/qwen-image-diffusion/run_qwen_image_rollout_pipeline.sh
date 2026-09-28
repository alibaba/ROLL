#!/usr/bin/env bash

set -euo pipefail

export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
export PYTHONPATH="$PWD"

ray stop --force
fuser -k /dev/nvidia* || true

CONFIG_PATH=$(basename "$(dirname "$0")")

python examples/start_diffusion_rollout_pipeline.py \
  --config_path "${CONFIG_PATH}" \
  --config_name qwen_image_rollout
