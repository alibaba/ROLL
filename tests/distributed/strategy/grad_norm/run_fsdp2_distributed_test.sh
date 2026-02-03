#!/bin/bash

set -e

echo "=================================="
echo "FSDP2 Gradient Norm Distributed Test"
echo "=================================="
echo ""

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA is required for this test."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "ERROR: This test requires at least 2 GPUs, but only $NUM_GPUS found."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "Running FSDP2 distributed gradient norm test with 2 GPUs..."
echo ""

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    "${SCRIPT_DIR}/test_fsdp2_grad_norm.py"

echo ""
echo "=================================="
echo "Test completed successfully!"
echo "=================================="

