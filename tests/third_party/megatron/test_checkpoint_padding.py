"""Distributed optimizer checkpoints must not serialize uninitialized padding."""
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_CHECKPOINT_PADDING_TESTS") != "1", reason="requires Megatron")
@pytest.mark.parametrize("rank,intervals,padding", [
    (0, [(5, 12), (20, 55)], [(0, 5), (12, 20), (55, 64)]),
    (1, [(0, 4), (16, 53)], [(68, 80)]),
])
def test_saved_padding_is_zero_without_changing_optimizer_parameters(rank, intervals, padding):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    dtype = (torch.bfloat16, torch.float32)
    originals = [
        {name: torch.full((end - start,), index + offset, dtype=torch.float32)
         for name, offset in (("param", 1.), ("exp_avg", 2.), ("exp_avg_sq", 3.))}
        for index, (start, end) in enumerate(intervals)
    ]

    def parameter_state():
        return {
            "per_bucket_numel": [[128]], "per_bucket_numel_unpadded": [[117]],
            0: {dtype: [[dict(values, gbuf_local_start=start, gbuf_local_end=end)
                        for values, (start, end) in zip(originals, intervals)]]},
        }

    optimizer = SimpleNamespace(
        data_parallel_group=SimpleNamespace(rank=lambda: rank, size=lambda: 2),
        get_parameter_state_dp_reshardable=parameter_state,
        data_parallel_group_idx=0, distributed_optimizer_instance_id=0,
        gbuf_ranges=[{}],
        buffers=[SimpleNamespace(buckets=[SimpleNamespace(
            numel_unpadded=117, grad_data=torch.zeros(128))])],
    )
    prior = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        # Make accidental torch.empty payloads reproducibly observable as NaN.
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        state = DistributedOptimizer.sharded_param_state_dp_reshardable(optimizer, {})
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = prior_fill
        torch.use_deterministic_algorithms(prior, warn_only=prior_warn)

    actual_padding, actual_parameters = [], []
    for entry in state[0][dtype][0]:
        tensors = [entry[name] for name in ("param", "exp_avg", "exp_avg_sq")]
        if entry["padding"].unwrap():
            for tensor in tensors:
                assert torch.equal(tensor.data, torch.zeros_like(tensor.data)), "checkpoint padding must be zero"
            start = tensors[0].global_offset[0]
            actual_padding.append((start, start + tensors[0].data.numel()))
        else:
            actual_parameters.append(entry)
    assert actual_padding == padding
    assert len(actual_parameters) == len(originals)
    for index, (entry, original, (start, end)) in enumerate(zip(actual_parameters, originals, intervals)):
        for name, offset in (("param", 1.), ("exp_avg", 2.), ("exp_avg_sq", 3.)):
            value = original[name]
            assert entry[name].data is value
            torch.testing.assert_close(value, torch.full_like(value, index + offset), rtol=0, atol=0)
            assert entry[name].global_offset == (rank * 64 + start,)
            assert entry[name].global_shape == (117,)
