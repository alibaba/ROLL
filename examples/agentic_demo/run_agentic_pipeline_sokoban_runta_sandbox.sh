#!/bin/bash
set +x

# Sandboxes are Runta cloud runtimes (https://runta.com/docs/): install the SDK
# and provide a token before starting the pipeline.
#   pip install runta-sdk
#   export RUNTA_TOKEN=...
: "${RUNTA_TOKEN:?Set RUNTA_TOKEN to a Runta API token}"

CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$PWD:$PYTHONPATH"
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agent_val_sokoban_runta_sandbox
