import copy
from numbers import Integral
from typing import Iterator, Optional

import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger


logger = get_logger()


def _normalize_pipeline_model_parallel_size(pipeline_model_parallel_size: Optional[int]) -> int:
    if pipeline_model_parallel_size is None:
        return 1
    if (
        isinstance(pipeline_model_parallel_size, bool)
        or not isinstance(pipeline_model_parallel_size, Integral)
        or pipeline_model_parallel_size < 1
    ):
        raise ValueError(
            "pipeline_model_parallel_size must be a positive integer, "
            f"got {pipeline_model_parallel_size!r}"
        )
    return int(pipeline_model_parallel_size)


def _split_ranges_to_vpp_multiple(
    micro_batch_indices: list[list[int]],
    micro_batch_lengths: list[int],
    pipeline_model_parallel_size: int,
    logical_mini_batch_index: int,
) -> None:
    """Split ranges in one logical mini-batch to satisfy the VPP schedule.

    Megatron's interleaved pipeline schedule requires the number of
    micro-batches in each logical mini-batch to be a multiple of the pipeline
    parallel size.  Splitting a range increases that count by one while
    preserving sample order and the padded sequence length associated with the
    range.  The lists are intentionally mutated in place so the caller can
    retain the grouping structure it has already built.
    """
    if pipeline_model_parallel_size <= 0:
        raise ValueError(
            "pipeline_model_parallel_size must be positive, "
            f"got {pipeline_model_parallel_size}"
        )

    current_count = len(micro_batch_indices)
    target_count = (
        (current_count + pipeline_model_parallel_size - 1) // pipeline_model_parallel_size
    ) * pipeline_model_parallel_size
    splits_needed = target_count - current_count

    while splits_needed:
        # Prefer the largest range so that repeated midpoint splits leave as
        # much room as possible for any remaining required splits.
        splittable_index = next(
            (
                index
                for index, (start, end) in sorted(
                    enumerate(micro_batch_indices),
                    key=lambda item: item[1][1] - item[1][0],
                    reverse=True,
                )
                if end - start > 1
            ),
            None,
        )
        if splittable_index is None:
            raise ValueError(
                f"logical mini-batch {logical_mini_batch_index} has {current_count} "
                f"micro-batches and cannot be split into a multiple of "
                f"pipeline_model_parallel_size={pipeline_model_parallel_size}; "
                "there are not enough multi-sample ranges"
            )

        start, end = micro_batch_indices[splittable_index]
        midpoint = start + (end - start) // 2
        length = micro_batch_lengths[splittable_index]
        micro_batch_indices[splittable_index : splittable_index + 1] = [
            [start, midpoint],
            [midpoint, end],
        ]
        micro_batch_lengths[splittable_index : splittable_index + 1] = [length, length]
        splits_needed -= 1
        current_count += 1


def dynamic_batching_shard(
    origin_batch: DataProto,
    dp_size: int,
    max_tokens_per_microbatch: int,
    sequence_length_round: int,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: int = None,
    log_prefix: str = None,
) -> tuple[DataProto, dict]:
    #TODO use Karmarkar–Karp algorithm to replace the greedy implementation
    pipeline_model_parallel_size = _normalize_pipeline_model_parallel_size(pipeline_model_parallel_size)
    attention_mask = origin_batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]
    seq_lens = attention_mask.view(batch_size, -1).sum(-1).tolist()

    if 0 in seq_lens:
        logger.warning(f"The attention_mask is all zero in the {log_prefix} stage. Please verify the rollout stage.")

    seq_index_sort_by_len = [i for i, _ in sorted(enumerate(seq_lens), key=lambda x: x[1])]
    seq_lens_sort = [seq_lens[i] for i in seq_index_sort_by_len]

    batch = origin_batch.slice()
    batch.reorder(torch.tensor(seq_index_sort_by_len))

    seq_len_of_shard = [seq_lens_sort[i::dp_size] for i in range(dp_size)]
    aggregated_shards = [batch[i::dp_size] for i in range(dp_size)]

    global_micro_batch_indices = [[0, 0]]
    global_micro_batch_lengths = [0]
    max_seqlen_this_mb = sequence_length_round # at least `sequence_length_round`
    shard_size = len(aggregated_shards[0])

    for shard_indice in range(shard_size):
        max_seqlen_this_shard_indice = 0
        for shard, seq_lens in zip(aggregated_shards, seq_len_of_shard):
            seq_len = seq_lens[shard_indice]
            max_seqlen_this_shard_indice = max(max_seqlen_this_shard_indice, seq_len)
        max_seqlen_this_shard_indice = (
            (max_seqlen_this_shard_indice + sequence_length_round - 1) // sequence_length_round
        ) * sequence_length_round
        assert max_seqlen_this_shard_indice <= max_tokens_per_microbatch, (
            f"got an input of padded ({sequence_length_round}) sequence length of {max_seqlen_this_shard_indice}, "
            f"however max microbatch size is {max_tokens_per_microbatch} tokens"
        )
        curr_mbs_size = global_micro_batch_indices[-1][1] - global_micro_batch_indices[-1][0] + 1
        max_seqlen_this_mb = max(max_seqlen_this_mb, max_seqlen_this_shard_indice)
        total_tokens_in_mbs = curr_mbs_size * max_seqlen_this_mb
        if total_tokens_in_mbs <= max_tokens_per_microbatch:
            global_micro_batch_indices[-1][-1] += 1
            global_micro_batch_lengths[-1] = max_seqlen_this_mb
        else:
            global_micro_batch_indices.append([shard_indice, shard_indice + 1])
            max_seqlen_this_mb = max_seqlen_this_shard_indice
            global_micro_batch_lengths.append(max_seqlen_this_mb)

    total_tokens = sum(
        (end - start) * length
        for (start, end), length in zip(global_micro_batch_indices, global_micro_batch_lengths)
    )
    if pipeline_model_parallel_size > 1 and virtual_pipeline_model_parallel_size:
        # Pad to a multiple of the pipeline parallel size.  A dynamic range
        # may contain many samples, so it can be split more than once when a
        # large token cap produces only a handful of ranges.  The old
        # implementation considered each original range only once, which
        # raised an assertion for e.g. one ``[0, 8)`` range needing three
        # additional micro-batches for ``pp_size=4``.
        num_micro_batches = len(global_micro_batch_indices)
        padded_num_micro_batches = (
            (num_micro_batches + pipeline_model_parallel_size - 1) // pipeline_model_parallel_size
        ) * pipeline_model_parallel_size
        assert pipeline_model_parallel_size <= shard_size, f"The pipeline_model_size: {pipeline_model_parallel_size} should not be greater than num_seqs in one dp_rank"
        assert padded_num_micro_batches <= shard_size
        _split_ranges_to_vpp_multiple(
            global_micro_batch_indices,
            global_micro_batch_lengths,
            pipeline_model_parallel_size,
            logical_mini_batch_index=0,
        )

    batch = DataProto.concat(aggregated_shards)
    batch.meta_info["global_micro_batch_indices"] = global_micro_batch_indices
    batch.meta_info["global_micro_batch_lengths"] = global_micro_batch_lengths
    batch.meta_info["micro_batch_indices"] = global_micro_batch_indices
    batch.meta_info["micro_batch_lengths"] = global_micro_batch_lengths
    batch.meta_info["num_micro_batchs"] = len(global_micro_batch_indices)

    valid_tokens = sum(seq_lens_sort)  # unmasked tokens
    actual_tokens_origin = batch_size * attention_mask.shape[-1]  # tokens with padding
    actual_tokens = total_tokens * dp_size  # tokens with padding, after dynamic batching
    removed_padding_tokens = actual_tokens_origin - actual_tokens
    removed_padding_ratio = removed_padding_tokens / actual_tokens_origin
    prefix = f"dynamic_batching/{log_prefix}" if log_prefix else "dynamic_batching"
    metrics = {
        f"{prefix}/valid_tokens": valid_tokens,
        f"{prefix}/actual_tokens_origin": actual_tokens_origin,
        f"{prefix}/actual_tokens": actual_tokens,
        f"{prefix}/removed_padding_tokens": removed_padding_tokens,
        f"{prefix}/removed_padding_ratio": removed_padding_ratio,
    }
    return batch, metrics


def make_mini_batch_iter_for_dynamic_batching(
    data: DataProto,
    epochs: int,
    ga_steps: int = 1,
    keep_mini_batch: bool = False,
    mini_batch_size: Optional[int] = None,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
) -> Iterator[DataProto]:
    """
        Iterator that groups previously created global micro batches into mini batches.

        Terminology:
        - Micro batch: The smallest training unit that can fit into GPU memory
          for one forward/backward pass.
          These are already determined in `dynamic_batching_shard` based on
          `max_tokens_per_microbatch`.
        - Mini batch: A group of several micro batches.
          During training, you iterate over each micro batch inside a mini batch,
          perform forward/backward passes, accumulate gradients, and then perform a
          parameter update (`optimizer.step()`).

        This function:
        1. Reads the global micro batch indices/lengths from `data.meta_info`.
        2. By default, groups `ga_steps` consecutive micro batches into a mini batch.
           With `keep_mini_batch`, instead preserves static sample-level mini-batch
           boundaries and splits dynamic micro batches that cross those boundaries.
        3. When virtual pipeline parallelism is enabled, splits ranges within
           each logical mini-batch until its micro-batch count is a multiple of
           ``pipeline_model_parallel_size``.
        4. Adjusts indices so micro batches are relative to the mini batch.
        5. Yields each mini batch for training.
        """
    pipeline_model_parallel_size = _normalize_pipeline_model_parallel_size(pipeline_model_parallel_size)
    global_micro_batch_indices = data.meta_info["global_micro_batch_indices"]
    global_micro_batch_lengths = data.meta_info["global_micro_batch_lengths"]

    assert ga_steps > 0, f"ga_steps must be positive, got {ga_steps}"
    assert len(global_micro_batch_indices) == len(global_micro_batch_lengths), (
        "global_micro_batch_indices and global_micro_batch_lengths must have the same length"
    )

    # The ranges are produced by ``dynamic_batching_shard`` and are local to a
    # DP rank.  Validate them before grouping so a malformed layout cannot
    # silently drop or duplicate samples when a range is split at a boundary.
    batch_size = len(data)
    validated_micro_batch_indices = []
    validated_micro_batch_lengths = []
    if batch_size == 0:
        assert not global_micro_batch_indices, "an empty batch cannot have micro-batch ranges"
    else:
        assert global_micro_batch_indices, "a non-empty batch must have micro-batch ranges"
        expected_start = 0
        for range_idx, (micro_start, micro_end) in enumerate(global_micro_batch_indices):
            assert isinstance(micro_start, Integral) and isinstance(micro_end, Integral), (
                f"micro-batch range {range_idx} must contain integer offsets, "
                f"got [{micro_start!r}, {micro_end!r}]"
            )
            micro_start = int(micro_start)
            micro_end = int(micro_end)
            assert micro_start == expected_start, (
                "global micro-batch ranges must be sorted and contiguous: "
                f"expected start {expected_start}, got {micro_start} at range {range_idx}"
            )
            assert micro_start < micro_end <= batch_size, (
                f"invalid global micro-batch range [{micro_start}, {micro_end}) "
                f"for batch size {batch_size}"
            )
            assert global_micro_batch_lengths[range_idx] > 0, (
                f"micro-batch length must be positive, got {global_micro_batch_lengths[range_idx]}"
            )
            validated_micro_batch_indices.append([micro_start, micro_end])
            validated_micro_batch_lengths.append(global_micro_batch_lengths[range_idx])
            expected_start = micro_end
        assert expected_start == batch_size, (
            "global micro-batch ranges must cover the complete local batch: "
            f"last end {expected_start}, batch size {batch_size}"
        )
    global_micro_batch_indices = validated_micro_batch_indices
    global_micro_batch_lengths = validated_micro_batch_lengths

    if keep_mini_batch:
        assert mini_batch_size is not None and mini_batch_size > 0, (
            f"mini_batch_size must be positive when keep_mini_batch is enabled, got {mini_batch_size}"
        )
        assert batch_size % mini_batch_size == 0, (
            f"batch size {batch_size} must be divisible by mini_batch_size {mini_batch_size} "
            "when keep_mini_batch is enabled"
        )

        grouped_indices = [[] for _ in range(batch_size // mini_batch_size)]
        grouped_lengths = [[] for _ in grouped_indices]
        for (start, end), length in zip(global_micro_batch_indices, global_micro_batch_lengths):
            while start < end:
                group_idx = start // mini_batch_size
                group_end = min(end, (group_idx + 1) * mini_batch_size)
                grouped_indices[group_idx].append([start, group_end])
                grouped_lengths[group_idx].append(length)
                start = group_end

        # Every logical mini-batch must contain exactly its static sample
        # budget.  These checks also make an accidental empty group fail at the
        # point where the bad layout is constructed rather than at yield time.
        for group_idx, (indices_group, lengths_group) in enumerate(zip(grouped_indices, grouped_lengths)):
            assert indices_group and lengths_group, f"logical mini-batch {group_idx} is empty"
            group_start = group_idx * mini_batch_size
            group_end = group_start + mini_batch_size
            assert indices_group[0][0] == group_start and indices_group[-1][1] == group_end, (
                f"logical mini-batch {group_idx} does not cover [{group_start}, {group_end})"
            )
            assert all(
                left[1] == right[0] for left, right in zip(indices_group, indices_group[1:])
            ), f"micro-batch ranges in logical mini-batch {group_idx} are not contiguous"

            if pipeline_model_parallel_size > 1 and virtual_pipeline_model_parallel_size:
                _split_ranges_to_vpp_multiple(
                    indices_group,
                    lengths_group,
                    pipeline_model_parallel_size,
                    group_idx,
                )
    else:
        grouped_indices = [
            global_micro_batch_indices[i : i + ga_steps]
            for i in range(0, len(global_micro_batch_indices), ga_steps)
        ]
        grouped_lengths = [
            global_micro_batch_lengths[i : i + ga_steps]
            for i in range(0, len(global_micro_batch_lengths), ga_steps)
        ]

    for _ in range(epochs):
        for indices_chunk, lengths_chunk in zip(grouped_indices, grouped_lengths):
            start = indices_chunk[0][0]
            end = indices_chunk[-1][-1]
            mini_batch = data.slice(start, end)
            mini_batch.meta_info = copy.deepcopy(data.meta_info)
            mini_batch.meta_info["micro_batch_indices"] = [
                [micro_batch_start - start, micro_batch_end - start]
                for micro_batch_start, micro_batch_end in indices_chunk
            ]
            mini_batch.meta_info["micro_batch_lengths"] = list(lengths_chunk)
            mini_batch.meta_info["mini_batch_size"] = len(mini_batch)
            mini_batch.meta_info["num_micro_batchs"] = len(indices_chunk)

            yield mini_batch


def make_micro_batch_iter_for_dynamic_batching(mini_batch: DataProto):
    micro_batch_indices = mini_batch.meta_info["micro_batch_indices"]
    micro_batch_lengths = mini_batch.meta_info["micro_batch_lengths"]
    for seqlen, (start_idx, end_idx) in zip(micro_batch_lengths, micro_batch_indices):
        micro_batch = mini_batch.slice(start_idx, end_idx)
        input_ids_shape = micro_batch.batch["input_ids"].shape
        for k in mini_batch.batch.keys():
            if (len(micro_batch.batch[k].shape) == len(input_ids_shape) or k == "position_ids") and micro_batch.batch[k].shape[-1] in (
                input_ids_shape[-1],
                input_ids_shape[-1] - 1,
            ):
                micro_batch.batch[k] = torch.narrow(
                    micro_batch.batch[k],
                    dim=-1,
                    start=0,
                    length=seqlen if micro_batch.batch[k].shape[-1] == input_ids_shape[-1] else seqlen - 1,
                )
        yield micro_batch
