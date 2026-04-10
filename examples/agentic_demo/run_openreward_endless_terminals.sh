#!/bin/bash
# Run OpenReward EndlessTerminals REINFORCE training with Qwen3.5-2B.
#
# Prerequisites:
#   pip install openreward   # inside the docker container
#
# Usage (inside roll_openreward_runner container):
#   export OPENREWARD_API_KEY="..."
#   export WANDB_API_KEY="..."
#   cd /home/ubuntu/ALE-latest/ROLL-personal
#   bash examples/agentic_demo/run_openreward_endless_terminals.sh

set -euo pipefail

: "${OPENREWARD_API_KEY:?Set OPENREWARD_API_KEY}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"

export NCCL_NET_PLUGIN=''
export NCCL_TUNER_PLUGIN=''
export NCCL_NET=Socket
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python examples/start_agentic_pipeline.py \
  --config_path agentic_demo \
  --config_name openreward_endless_terminals_reinforce_qwen35_2b
