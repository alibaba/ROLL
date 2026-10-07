from __future__ import annotations

from typing import Iterable

import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.tinker_backend_runtime import types


def _model_input_to_token_ids(model_input: types.ModelInput) -> list[int]:
    token_ids: list[int] = []
    for chunk in model_input.chunks:
        if getattr(chunk, "type", None) != "encoded_text":
            raise ValueError(
                "Tinker training only supports encoded_text chunks, "
                f"got {getattr(chunk, 'type', type(chunk))}"
            )
        token_ids.extend(chunk.tokens)
    return token_ids


def _pad_1d(values: list[int] | list[float], length: int, pad_value: int | float) -> list[int] | list[float]:
    if len(values) > length:
        raise ValueError(f"sequence length {len(values)} exceeds target length {length}")
    return values + [pad_value] * (length - len(values))


def _coerce_tail(values: Iterable[float], expected_len: int) -> list[float]:
    coerced = [float(v) for v in values]
    if len(coerced) == expected_len:
        return coerced
    if len(coerced) == 0:
        return [0.0] * expected_len
    raise ValueError(f"Expected {expected_len} values, got {len(coerced)}")


def prepared_batch_to_dataproto(
    prepared: types.PreparedModelPassBatch,
    *,
    sequence_length: int,
    pad_token_id: int,
) -> DataProto:
    input_ids_rows: list[list[int]] = []
    attention_mask_rows: list[list[int]] = []
    position_ids_rows: list[list[int]] = []
    response_mask_rows: list[list[bool]] = []
    prompt_mask_rows: list[list[bool]] = []
    tinker_weight_rows: list[list[float]] = []
    tinker_old_logprob_rows: list[list[float]] = []
    tinker_advantage_rows: list[list[float]] = []
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []

    for model_input, target_tokens, token_weights, sampling_logprobs, advantages in zip(
        prepared.all_model_inputs,
        prepared.all_targets,
        prepared.all_token_weights,
        prepared.all_sampling_logprobs,
        prepared.all_advantages,
        strict=True,
    ):
        prompt_ids = _model_input_to_token_ids(model_input)
        if not prompt_ids:
            raise ValueError("Tinker training expects at least one prompt token per datum")

        target_ids = [int(token_id) for token_id in target_tokens]
        weights_tail = _coerce_tail(token_weights, len(target_ids))
        logprobs_tail = _coerce_tail(sampling_logprobs, len(target_ids))
        advantages_tail = _coerce_tail(advantages, len(target_ids))

        prompt_len = len(prompt_ids)
        target_len = len(target_ids)
        total_len = prompt_len + target_len
        if total_len > sequence_length:
            raise ValueError(
                f"Prompt + target length {total_len} exceeds configured sequence_length {sequence_length}"
            )

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * total_len
        response_mask = [False] * prompt_len + [True] * target_len
        prompt_mask = [True] * prompt_len + [False] * target_len
        position_ids = list(range(total_len))

        shifted_prefix = [0.0] * max(prompt_len - 1, 0)
        shifted_len = total_len - 1
        tinker_weights = shifted_prefix + weights_tail
        tinker_old_logprobs = shifted_prefix + logprobs_tail
        tinker_advantages = shifted_prefix + advantages_tail

        input_ids_rows.append(_pad_1d(input_ids, sequence_length, pad_token_id))
        attention_mask_rows.append(_pad_1d(attention_mask, sequence_length, 0))
        position_ids_rows.append(_pad_1d(position_ids, sequence_length, 0))
        response_mask_rows.append(_pad_1d(response_mask, sequence_length, False))
        prompt_mask_rows.append(_pad_1d(prompt_mask, sequence_length, False))
        tinker_weight_rows.append(_pad_1d(tinker_weights, sequence_length - 1, 0.0))
        tinker_old_logprob_rows.append(_pad_1d(tinker_old_logprobs, sequence_length - 1, 0.0))
        tinker_advantage_rows.append(_pad_1d(tinker_advantages, sequence_length - 1, 0.0))
        prompt_lengths.append(prompt_len)
        target_lengths.append(target_len)

    tensors = {
        "input_ids": torch.tensor(input_ids_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_rows, dtype=torch.long),
        "position_ids": torch.tensor(position_ids_rows, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_rows, dtype=torch.bool),
        "prompt_mask": torch.tensor(prompt_mask_rows, dtype=torch.bool),
        "tinker_weights": torch.tensor(tinker_weight_rows, dtype=torch.float32),
        "tinker_old_logprobs": torch.tensor(tinker_old_logprob_rows, dtype=torch.float32),
        "tinker_advantages": torch.tensor(tinker_advantage_rows, dtype=torch.float32),
    }
    non_tensors = {
        "tinker_prompt_lengths": prompt_lengths,
        "tinker_target_lengths": target_lengths,
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={"loss_mask_keys": []})
