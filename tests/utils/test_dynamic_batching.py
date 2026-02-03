import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.dynamic_batching import *


def test_dynamic_batching():
    dp_size = 2
    num_seq = 6
    max_seq_len = 20
    seqs_len = [2, 4, 7, 6, 3, 4]
    input_ids = torch.arange(num_seq).unsqueeze(1).expand(num_seq, max_seq_len)
    attention_mask = (torch.arange(max_seq_len) < torch.tensor(seqs_len)[:, None]).int()
    data = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    max_tokens_per_microbatch = 20
    sequence_length_round = 2

    # test dynamic_batching_shard
    data, _ = dynamic_batching_shard(data, dp_size, max_tokens_per_microbatch, sequence_length_round)
    assert data.meta_info["global_micro_batch_indices"] == [[0,2],[2,3]]
    assert data.meta_info["global_micro_batch_lengths"] == [4, 8]

    # test make_mini_batch_iter_for_dynamic_batching
    data_dp0 = data.slice(0, num_seq//dp_size)
    mini_batch_iter = make_mini_batch_iter_for_dynamic_batching(data_dp0, 1, 1)
    mini_batch0 = next(mini_batch_iter)
    assert mini_batch0.meta_info["micro_batch_indices"] == [[0,2]]
    assert data.meta_info["micro_batch_lengths"] == [4]
    
    # test make_mini_batch_iter_for_dynamic_batching
    micro_batch_iter = make_micro_batch_iter_for_dynamic_batching(mini_batch0)
    micro_batch0 = next(micro_batch_iter)
    assert tuple(micro_batch0.batch["input_ids"].shape) == (2,4)


def test_dynamic_batching_with_vpp():
    torch.manual_seed(42)
    dp_size = 4
    num_seq = 256
    max_seq_len = 8192
    seqs_len = torch.randint(low=128, high=8192, size=(256,)).tolist()
    input_ids = torch.arange(num_seq).unsqueeze(1).expand(num_seq, max_seq_len)
    attention_mask = (torch.arange(max_seq_len) < torch.tensor(seqs_len)[:, None]).int()
    data = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    max_tokens_per_microbatch = 8192
    sequence_length_round = 128

    # test dynamic_batching_shard
    pipeline_model_parallel_size = 4
    virtual_pipeline_model_size = 2
    data, _ = dynamic_batching_shard(data, dp_size, max_tokens_per_microbatch, sequence_length_round, 
                                     pipeline_model_parallel_size=pipeline_model_parallel_size,
                                     virtual_pipeline_model_parallel_size=virtual_pipeline_model_size)
    assert data.meta_info["global_micro_batch_indices"].__len__() % pipeline_model_parallel_size == 0
    assert data.meta_info["global_micro_batch_lengths"].__len__() == data.meta_info["global_micro_batch_indices"].__len__()


if __name__ == "__main__":
    # test_dynamic_batching()
    test_dynamic_batching_with_vpp()
