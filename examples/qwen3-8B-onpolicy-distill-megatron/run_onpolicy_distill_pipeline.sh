#!/bin/bash

# On-Policy Distill Pipeline Run Script

# Set environment variables
export RAY_DEDUP_LOGS=1
export USE_MODELSCOPE=1

# Config path
CONFIG_PATH="qwen3-8B-onpolicy-distill-megatron"
CONFIG_NAME="onpolicy_distill_test"

# Run pipeline
python examples/start_onpolicy_distill_pipeline.py \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    "$@"
