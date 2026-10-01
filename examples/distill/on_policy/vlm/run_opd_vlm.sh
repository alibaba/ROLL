#!/bin/bash
set +x

CONFIG_PATH="distill/on_policy/vlm"
python examples/start_onpolicy_distill_pipeline.py \
    --config_path $CONFIG_PATH \
    --config_name multi_teacher_opd_config
