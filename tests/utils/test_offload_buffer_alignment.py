"""Frozen parameter packing must retain requested GEMM pointer alignment."""
import pytest
import torch


@pytest.mark.parametrize('dtype, offsets, total', [
    (torch.float32, [0, 64, 128], 129),
    (torch.bfloat16, [0, 128, 256], 257),
])
def test_aligned_packing_round_trips_ragged_parameters(dtype, offsets, total):
    from roll.utils.offload_states import move_tensors_to_device_buffer, move_device_buffer_to_tensors

    expected = [torch.arange(3, dtype=dtype), torch.arange(15, dtype=dtype).reshape(3, 5), torch.tensor([17], dtype=dtype)]
    tensors = [torch.nn.Parameter(value.clone(), requires_grad=False) for value in expected]
    packed = move_tensors_to_device_buffer(tensors, device='cpu', pin_memory=False, alignment_bytes=256)
    assert packed.numel() == total
    for tensor, offset, value in zip(tensors, offsets, expected):
        assert tensor.data_ptr() - packed.data_ptr() == offset * tensor.element_size()
        torch.testing.assert_close(tensor, value, atol=0, rtol=0)
    # Move to a fresh allocation and rebind every view from the padded layout.
    restored = packed.clone()
    move_device_buffer_to_tensors(tensors, restored, alignment_bytes=256)
    for tensor, offset, value in zip(tensors, offsets, expected):
        assert tensor.data_ptr() - restored.data_ptr() == offset * tensor.element_size()
        torch.testing.assert_close(tensor, value, atol=0, rtol=0)


@pytest.mark.parametrize('alignment', [0, -1, 3])
def test_invalid_buffer_alignment_is_rejected_before_rebinding(alignment):
    from roll.utils.offload_states import move_tensors_to_device_buffer

    tensor = torch.tensor([1., 2., 3.])
    pointer = tensor.data_ptr()
    with pytest.raises(ValueError, match='alignment'):
        move_tensors_to_device_buffer([tensor], pin_memory=False, alignment_bytes=alignment)
    assert tensor.data_ptr() == pointer
    torch.testing.assert_close(tensor, torch.tensor([1., 2., 3.]), atol=0, rtol=0)
