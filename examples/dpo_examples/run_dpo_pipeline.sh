#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_dpo_pipeline.py --config_path $CONFIG_PATH  --config_name qwen3-30BA3B-dpo_megatron_80GB
