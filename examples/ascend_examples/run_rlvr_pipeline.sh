#!/bin/bash
set +x

export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export VLLM_ASCEND_ENABLE_NZ=0

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_rlvr_pipeline.py --config_path $CONFIG_PATH --config_name qwen3_30b_rlvr_fsdp2
