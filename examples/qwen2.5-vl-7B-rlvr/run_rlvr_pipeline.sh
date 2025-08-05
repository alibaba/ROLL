#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python -m examples.start_rlvr_pipeline --config_path $CONFIG_PATH  --config_name rlvr_config_ds