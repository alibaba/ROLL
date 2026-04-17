#!/bin/bash
# Atropos-ROLL Integration Demo Runner
# Usage: bash examples/agentic_demo/run_atropos_gsm8k.sh <WANDB_API_KEY>

if [ -z "$1" ]; then
    echo "Usage: $0 <WANDB_API_KEY>"
    exit 1
fi

export WANDB_API_KEY="$1"
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/../atropos:$(pwd)/../atropos/atroposlib

echo "🚀 Starting Atropos-GSM8K Reinforce Demo..."
echo "📊 Tracking on WandB: roll-atropos-integration"

python examples/start_agentic_pipeline.py \
    --config_path agentic_demo \
    --config_name atropos_gsm8k_reinforce_qwen25_3b
