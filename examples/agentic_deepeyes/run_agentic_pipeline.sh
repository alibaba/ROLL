#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name deepeyes_2gpus

