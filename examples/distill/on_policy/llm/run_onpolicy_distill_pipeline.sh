#!/bin/bash

# On-Policy Distill Pipeline Run Script

# Set environment variables
export RAY_DEDUP_LOGS=1
export USE_MODELSCOPE=1

# Config path
CONFIG_PATH="distill/on_policy/llm"
CONFIG_NAME="onpolicy_distill_config"

# Run pipeline
python examples/start_onpolicy_distill_pipeline.py \
    --config_path ${CONFIG_PATH} \
    --config_name ${CONFIG_NAME} \
    "$@"
