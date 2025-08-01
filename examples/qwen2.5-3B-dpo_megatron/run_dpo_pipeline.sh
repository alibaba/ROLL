#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_dpo_pipeline.py --config_path $CONFIG_PATH  --config_name dpo_config
