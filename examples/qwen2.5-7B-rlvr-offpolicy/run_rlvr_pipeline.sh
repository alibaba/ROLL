#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_rlvr_pipeline.py --config_path $CONFIG_PATH  --config_name rlvr_config
