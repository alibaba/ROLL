#!/usr/bin/env bash

export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA

ray stop --force
fuser -k /dev/nvidia*

CONFIG_PATH=$(basename $(dirname $0))

python examples/start_diffusion_pipeline.py \
  --config_path ${CONFIG_PATH} \
  --config_name flow_grpo_fsdp2_80GB \
