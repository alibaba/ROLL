import pytest
import torch


def test_ulysses_pad_and_slice_inputs_divisible():
    from roll.utils.context_parallel.rmpad_ulysses import ulysses_pad_and_slice_inputs

    input_ids = torch.arange(10, dtype=torch.long).unsqueeze(0)  # [1, 10]
    pos = torch.arange(10, dtype=torch.long).unsqueeze(0)  # [1, 10]

    # cp_size=2 => no padding needed
    x0, p0, pad0 = ulysses_pad_and_slice_inputs(input_ids, pos, cp_size=2, cp_rank=0)
    x1, p1, pad1 = ulysses_pad_and_slice_inputs(input_ids, pos, cp_size=2, cp_rank=1)

    assert pad0 == 0 and pad1 == 0
    assert x0.shape == (1, 5) and x1.shape == (1, 5)
    assert torch.equal(torch.cat([x0, x1], dim=1), input_ids)
    assert torch.equal(torch.cat([p0, p1], dim=1), pos)


def test_ulysses_pad_and_slice_inputs_with_padding():
    from roll.utils.context_parallel.rmpad_ulysses import ulysses_pad_and_slice_inputs

    input_ids = torch.arange(11, dtype=torch.long).unsqueeze(0)  # [1, 11]
    pos = torch.arange(11, dtype=torch.long).unsqueeze(0)  # [1, 11]

    # cp_size=4 => pad to 12
    parts = []
    pads = []
    for r in range(4):
        x, p, pad = ulysses_pad_and_slice_inputs(input_ids, pos, cp_size=4, cp_rank=r)
        parts.append(x)
        pads.append(pad)
        assert x.shape == (1, 3)
        assert p is not None and p.shape == (1, 3)

    assert all(p == 1 for p in pads)
    full = torch.cat(parts, dim=1)
    assert full.shape == (1, 12)
    assert torch.equal(full[:, :11], input_ids)


def test_gather_outputs_and_unpad_no_group_is_noop():
    from roll.utils.context_parallel.rmpad_ulysses import gather_outputs_and_unpad

    x = torch.randn(1, 8, 3)
    y = gather_outputs_and_unpad(x, gather_dim=1, unpad_dim=1, padding_size=2, group=None)
    assert torch.equal(y, x[:, :6])
