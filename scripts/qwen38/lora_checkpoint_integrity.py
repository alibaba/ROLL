"""Validate native Megatron LoRA matrix metadata without reading tensor storage.

This checks declared pairs, precision and native grouped rank. It does not infer
missing modules or their input/output dimensions from a complete architecture.
"""
import re


def validate_native_lora_metadata(model_state, adapter_config, model_config):
    import torch

    rank = adapter_config.get('r')
    if type(rank) is not int or rank < 1:
        raise ValueError('Native LoRA requires a positive configured rank')
    if adapter_config.get('rank_pattern'):
        raise ValueError('Native LoRA rank_pattern validation is not supported')
    dtype = (torch.bfloat16 if model_config.get('bf16') else
             torch.float16 if model_config.get('fp16') else None)
    pairs = {}
    biases = []
    for name, value in model_state.items():
        if not isinstance(name, str):
            raise ValueError('Native LoRA tensor names must be strings')
        if 'lora_' not in name:
            continue
        # TE uses a zero-length uint8 tensor for absent FP8 state. It is not a
        # trainable adapter matrix and must not count as one.
        if name.endswith('._extra_state'):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                    or value.ndim != 1 or value.numel() != 0):
                raise ValueError(f'Invalid native LoRA TE extra state: {name}')
            continue
        match = re.fullmatch(r'(.+)\.lora_([AB])(?:\.([^.]+))?\.(weight|bias)(\d*)', name)
        if match is None:
            raise ValueError(f'Unsupported native LoRA tensor: {name}')
        stem, side, adapter, kind, index = match.groups()
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.numel() == 0 or value.ndim != (2 if kind == 'weight' else 1)):
            raise ValueError(f'Invalid native LoRA {kind} shape or type: {name}')
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f'Native LoRA dtype disagrees with model precision: {name}')
        key = (stem, adapter, index)
        if kind == 'bias':
            biases.append((key, side, name, value))
        else:
            pairs.setdefault(key, {})[side] = (name, value)
    if not pairs:
        raise ValueError('Native adapter payload contains no LoRA matrix pairs')
    for (stem, adapter, index), pair in pairs.items():
        if set(pair) != {'A', 'B'}:
            raise ValueError(f'Incomplete native LoRA A/B pair: {stem}')
        a, b = pair['A'][1], pair['B'][1]
        expected_rank = rank
        if index:
            topk = model_config.get('moe_router_topk')
            if type(topk) is not int or topk < 1:
                raise ValueError('Grouped native LoRA requires configured moe_router_topk')
            expected_rank = rank // topk
        if expected_rank < 1 or a.shape[0] != expected_rank or b.shape[1] != expected_rank:
            raise ValueError(f'Native LoRA A/B rank mismatch: {stem}, expected {expected_rank}')
        if a.dtype != b.dtype:
            raise ValueError(f'Native LoRA A/B dtype mismatch: {stem}')
    for key, side, name, value in biases:
        pair = pairs.get(key)
        if pair is None or value.shape[0] != pair[side][1].shape[0] or value.dtype != pair[side][1].dtype:
            raise ValueError(f'Native LoRA bias disagrees with its matrix: {name}')
    return len(pairs)
