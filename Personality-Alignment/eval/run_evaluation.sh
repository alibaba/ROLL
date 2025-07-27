#!/bin/bash

# Quick evaluation script
echo "Running evaluation for Qwen model results..."

python llm_eval.py \
    --input-file /project/hdtaccuracy/Personality-Alignment/split_data_v5_filtered/filtered_dataset.jsonl \
    --output-file /home/szhangfa/ROLL/Personality-Alignment/eval/qwen_multicard_results.jsonl \
    --evaluator-type qwen \
    --qwen-model-path /project/hdtaccuracy/models/base/Qwen3-8B \
    --inference-batch-size 32
