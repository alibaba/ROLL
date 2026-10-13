# Qwen3.6 / Qwen3.8 27B SFT smoke test

Both models use the `qwen3_5` architecture: 24 query heads, four KV heads,
and interleaved Gated Delta Net (GDN) and full-attention layers. Tensor
parallel size eight replicates KV heads, so each attention output gate
must be sliced to the same local heads as its query.

The adapter also supports `additional_configs.gdn_backend: torch` under
`sft_train.strategy_args.strategy_config`. This selects Megatron's PyTorch
causal convolution and chunked delta-rule implementations for GDN layers.
It is an explicit compatibility fallback for environments where the fused
causal-conv1d or FLA delta-rule kernels are unsuitable. It can be slower
and use more memory than the default `fla` backend. FLA must still be
installed: Megatron uses its L2-normalization kernel in both modes.
Other layers keep their original deterministic settings.

The tested environment uses eight NVIDIA H800 80 GB GPUs, Python 3.12,
PyTorch 2.10.0+cu130, Megatron-Core 0.16.0, Transformers 5.5.4,
Transformer Engine 2.14.1, flash-linear-attention 0.5.0, and Ray 2.48.0.
Use mutually compatible CUDA extension builds; installing the adapter
with `--no-deps` assumes those dependencies and ROLL's common requirements
are already installed. The fallback does not establish compatibility
with every Megatron version accepted by the package metadata.

Run from the ROLL repository root in a compatible CUDA training environment
with eight GPUs, after installing this checkout's ROLL and MCoreAdapter:

```bash
pip install --no-deps -e . -e ./mcore_adapter

MODEL_PATH=/path/to/Qwen3.6-27B \
ROLL_OUTPUT_DIR=./output/qwen3.6-27b-sft-smoke \
TORCH_COMPILE_DISABLE=1 NVTE_TORCH_COMPILE=0 \
python examples/start_sft_pipeline.py \
  --config_path qwen3-27B-sft_megatron --config_name sft_config
```

Repeat with `MODEL_PATH=/path/to/Qwen3.8-27B` and a different output directory.
The example runs three BF16 optimizer steps on 12 included arithmetic
examples, with validation and model checkpoints. These tiny data are for
execution checks only; they do not measure model quality or convergence.
Model-only checkpoints do not include optimizer state for resuming training.

The example uses TP=8, sequence parallelism, and full recomputation. It
disables sequence packing and context parallelism for the Megatron 0.16
GDN fallback. It freezes the vision encoder and exercises text SFT.

The focused distributed regression needs no pretrained weights:

```bash
PYTHONPATH=.:mcore_adapter/src \
TORCH_COMPILE_DISABLE=1 NVTE_TORCH_COMPILE=0 \
torchrun --standalone --nproc_per_node=8 -m pytest \
  tests/models/test_qwen3_5_hybrid.py
```
