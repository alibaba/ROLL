#!/bin/bash
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLL_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="$(basename "${SCRIPT_DIR}")"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"
python "$ROLL_PATH/examples/start_agentic_pipeline.py" --config_path "$CONFIG_PATH" --config_name agent_val_frozen_lake_single_node_demo
