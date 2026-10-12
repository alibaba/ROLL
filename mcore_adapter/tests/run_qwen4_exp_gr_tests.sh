#!/usr/bin/env bash
# Run from the repository root in the real Megatron/Transformer Engine image.
set -euo pipefail
export RUN_MEGATRON_GR_TESTS=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-}"
GR_TEST_GPUS="${GR_TEST_GPUS:-0,1}"
CUDA_VISIBLE_DEVICES="${GR_TEST_GPUS%%,*}" python3 -m pytest \
  mcore_adapter/tests/test_qwen4_exp_hyperconnection.py -q -ra
CUDA_VISIBLE_DEVICES="$GR_TEST_GPUS" torchrun --nnodes=1 --node-rank=0 \
  --master-addr=127.0.0.1 --master-port="${GR_TEST_PORT:-29621}" --nproc-per-node=2 \
  -m pytest mcore_adapter/tests/test_qwen4_exp_hyperconnection.py \
  -k sequence_parallel_hc_parameters_are_reduced_by_megatron -q -ra
