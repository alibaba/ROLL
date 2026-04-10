#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$PWD:$PYTHONPATH"
python examples/start_agentic_rollout_pipeline.py --config_path $CONFIG_PATH  --config_name agent_rollout_rock_swe

