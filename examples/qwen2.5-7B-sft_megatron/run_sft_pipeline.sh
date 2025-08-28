#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_sft_pipeline.py --config_path $CONFIG_PATH  --config_name sft_config
