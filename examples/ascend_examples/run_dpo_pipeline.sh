#!/bin/bash
set +x

export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export VLLM_ASCEND_ENABLE_NZ=0

CONFIG_PATH=$(basename "$(dirname "$0")")
python examples/start_dpo_pipeline.py \
  --config_path "$CONFIG_PATH" \
  --config_name qwen3_4B_dpo_megatron
