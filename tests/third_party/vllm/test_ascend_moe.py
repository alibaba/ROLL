import pytest


pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from roll.third_party.vllm.npu.ascend_moe import (
    _copy_transposed_expert_shard,
    _map_global_expert_to_local,
)


class _Experts:
    expert_map = [-1, 0, -1, 1]


def test_copy_transposed_w13_tp_shard_and_ep_local_expert():
    # Two local experts, TP=2, hidden=4, intermediate=6.
    param = torch.zeros((2, 4, 6), dtype=torch.float32)
    loaded = torch.arange(6 * 4, dtype=torch.float32).reshape(1, 6, 4)

    copied = _copy_transposed_expert_shard(
        param,
        loaded,
        "w1",
        local_expert_id=1,
        tp_rank=1,
        tp_world_size=2,
    )

    assert copied
    expected = loaded[0].transpose(-1, -2)[:, 3:]
    assert torch.equal(param[1, :, :3], expected)
    assert torch.count_nonzero(param[0]) == 0


def test_copy_transposed_w2_skips_ep_nonlocal_expert():
    param = torch.zeros((2, 3, 4), dtype=torch.float32)
    loaded = torch.arange(4 * 6, dtype=torch.float32).reshape(1, 4, 6)

    copied = _copy_transposed_expert_shard(
        param,
        loaded,
        "w2",
        local_expert_id=-1,
        tp_rank=0,
        tp_world_size=2,
    )

    assert not copied
    assert torch.count_nonzero(param) == 0

    copied = _copy_transposed_expert_shard(
        param,
        loaded,
        "w2",
        local_expert_id=0,
        tp_rank=1,
        tp_world_size=2,
    )
    assert copied
    assert torch.equal(param[0], loaded[0, :, 3:].transpose(-1, -2))


def test_ep_global_to_local_mapping():
    experts = _Experts()
    assert _map_global_expert_to_local(experts, 1) == 0
    assert _map_global_expert_to_local(experts, 3) == 1
    assert _map_global_expert_to_local(experts, 0) == -1
