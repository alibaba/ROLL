from typing import List

import torch

# Block quant operator
#
# Borrow from transformers
#   https://huggingface.co/docs/transformers/en/quantization/finegrained_fp8
#   https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/quantizers/quantizer_finegrained_fp8.py#L83
#
# May use op from torchao:
#   https://github.com/pytorch/ao/pull/1668
#   https://github.com/volcengine/verl/pull/3084
def per_block_fp8_quant(param_value: torch.Tensor, weight_block_size: List[int]):
    """
    Quantizes weights to FP8 format using Block-wise quantization
    """
    # Get FP8 min/max values
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    block_size_m, block_size_n = weight_block_size

    rows, cols = param_value.shape[-2:]

    if rows % block_size_m != 0 or cols % block_size_n != 0:
        raise ValueError(
            f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
        )
    param_value_orig_shape = param_value.shape

    param_value = param_value.reshape(
        -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
    ).permute(0, 1, 3, 2, 4)

    # Calculate scaling factor for each block
    max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
    scale = fp8_max / max_abs
    scale_orig_shape = scale.shape
    scale = scale.unsqueeze(-1).unsqueeze(-1)

    # Quantize the weights
    quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
    # Reshape back to matrix shape
    quantized_param = quantized_param.reshape(param_value_orig_shape)

    # Construct the final, correct shape for the scales
    num_row_blocks = rows // block_size_m
    num_col_blocks = cols // block_size_n
    # This preserves original batch dimensions, if any
    final_scale_shape = (*param_value_orig_shape[:-2], num_row_blocks, num_col_blocks)
    # Reshape directly to the correct shape and take the reciprocal
    scale = scale.reshape(final_scale_shape).reciprocal()

    # TODO: DeepGemm scales need to be transposed and aligned (said in vLLM fp8.py)?

    # TODO: On B200, DeepGemm only support E8M0 scale

    return quantized_param, scale
