from typing import Iterator

import torch

from roll.distributed.scheduler.protocol import DataProto


def dynamic_batching_shard(
    origin_batch: DataProto,
    dp_size: int,
    max_tokens_per_microbatch: int,
    sequence_length_round: int,
    log_prefix: str = None,
) -> tuple[DataProto, dict]:
    attention_mask = origin_batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]
    seq_lens = attention_mask.view(batch_size, -1).sum(-1).tolist()

    seq_index_sort_by_len = [i for i, _ in sorted(enumerate(seq_lens), key=lambda x: x[1])]
    seq_lens_sort = [seq_lens[i] for i in seq_index_sort_by_len]

    batch = origin_batch.slice()
    batch.reorder(torch.tensor(seq_index_sort_by_len))

    seq_len_of_shard = [seq_lens_sort[i::dp_size] for i in range(dp_size)]
    aggregated_shards = [batch[i::dp_size] for i in range(dp_size)]

    global_micro_batch_indices = [[0, 0]]
    global_micro_batch_lengths = [0]
    max_seqlen_this_mb = 0
    shard_size = len(aggregated_shards[0])

    total_tokens = 0
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
            total_tokens += total_tokens_in_mbs
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
) -> Iterator[DataProto]:
    """
        Iterator that groups previously created global micro batches into mini batches
        based on gradient accumulation steps (ga_steps).

        Terminology:
        - Micro batch: The smallest training unit that can fit into GPU memory
          for one forward/backward pass.
          These are already determined in `dynamic_batching_shard` based on
          `max_tokens_per_microbatch`.
        - Mini batch: A group of several micro batches.
          During training, you iterate over each micro batch inside a mini batch,
          perform forward/backward passes, accumulate gradients, and after `ga_steps`
          micro batches, perform a parameter update (`optimizer.step()`).

        This function:
        1. Reads the global micro batch indices/lengths from `data.meta_info`.
        2. Groups `ga_steps` consecutive micro batches into a single mini batch.
        3. Adjusts indices so micro batches are relative to the mini batch.
        4. Yields each mini batch for training.
        """
    global_micro_batch_indices = data.meta_info["global_micro_batch_indices"]
    global_micro_batch_lengths = data.meta_info["global_micro_batch_lengths"]
    for _ in range(epochs):
        for i in range(0, len(global_micro_batch_indices), ga_steps):
            indices_chunk = global_micro_batch_indices[i : i + ga_steps]
            start = indices_chunk[0][0]
            end = indices_chunk[-1][-1]
            mini_batch = data.slice(start, end)

            data.meta_info["micro_batch_indices"] = [[x - start for x in row] for row in indices_chunk]
            data.meta_info["micro_batch_lengths"] = global_micro_batch_lengths[i : i + ga_steps]
            mini_batch.meta_info["num_micro_batchs"] = len(indices_chunk)

            yield (mini_batch)


def make_micro_batch_iter_for_dynamic_batching(mini_batch: DataProto):
    micro_batch_indices = mini_batch.meta_info["micro_batch_indices"]
    micro_batch_lengths = mini_batch.meta_info["micro_batch_lengths"]
    for seqlen, (start_idx, end_idx) in zip(micro_batch_lengths, micro_batch_indices):
        micro_batch = mini_batch.slice(start_idx, end_idx)
        input_ids_shape = micro_batch.batch["input_ids"].shape
        for k in mini_batch.batch.keys():
            if len(micro_batch.batch[k].shape) == len(input_ids_shape) and micro_batch.batch[k].shape[-1] in (
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
