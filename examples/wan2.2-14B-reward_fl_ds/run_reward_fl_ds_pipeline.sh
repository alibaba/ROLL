#!/bin/bash
set +x


CONFIG_PATH=$(basename $(dirname $0))
python examples/start_reward_fl_pipeline.py --config_path $CONFIG_PATH  --config_name reward_fl_config
