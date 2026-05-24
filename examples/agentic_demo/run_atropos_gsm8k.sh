#!/bin/bash
# Atropos-ROLL Integration: GSM8K + Qwen2.5-0.5B
# Usage: bash examples/agentic_demo/run_atropos_gsm8k.sh

export VLLM_USE_V1=0
export WANDB_MODE=online
export TQDM_DISABLE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name atropos_gsm8k_grpo_qwen25_0.5b
