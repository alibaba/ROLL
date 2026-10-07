"""Exercise native ROLL dispatch using real CPU TensorDict input, without Ray init."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.decorator import (
    BIND_WORKER_METHOD_FLAG,
    Dispatch,
    _dispatch_dp_mp_compute,
    get_predefined_dispatch_fn,
)
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.tinker_backend_runtime.workers import TinkerActorWorker


class CPUCluster:
    def __init__(self, dp_size=1, tp_size=4):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.world_size = dp_size * tp_size

    def get_rank_info(self, rank):
        return SimpleNamespace(
            dp_rank=rank // self.tp_size, tp_rank=rank % self.tp_size,
            cp_rank=0, pp_rank=0, is_pipeline_last_stage=True,
        )


def input_data(batch_size=2):
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(batch_size * 3, device="cpu").reshape(batch_size, 3),
            "attention_mask": torch.ones((batch_size, 3), dtype=torch.long, device="cpu"),
        },
        meta_info={"tinker_loss_fn": "cross_entropy"},
    )


def assert_real_input(actual, expected_ids):
    assert isinstance(actual._batch, TensorDict)
    assert actual.batch.batch_size == torch.Size([expected_ids.shape[0]])
    assert actual.batch["input_ids"].device.type == "cpu"
    torch.testing.assert_close(actual.batch["input_ids"], expected_ids)
    assert actual.meta_info == {"tinker_loss_fn": "cross_entropy"}


@pytest.mark.parametrize("method", ["forward_tinker", "forward_backward_accumulate_tinker"])
@pytest.mark.parametrize("keyword", [False, True])
def test_actual_tinker_worker_binding_dispatches_real_input_to_every_tp_rank(method, keyword):
    binding = getattr(getattr(TinkerActorWorker, method), BIND_WORKER_METHOD_FLAG)
    assert binding["dispatch_mode"] is Dispatch.DP_MP_COMPUTE
    dispatch = get_predefined_dispatch_fn(binding["dispatch_mode"])["dispatch_fn"]
    source = input_data()
    if keyword:
        args, kwargs = dispatch(CPUCluster(), data=source)
        assert args == ()
        recipients = kwargs["data"]
    else:
        args, kwargs = dispatch(CPUCluster(), source)
        assert kwargs == {}
        recipients = args[0]
    assert len(recipients) == 4
    for recipient in recipients:
        assert_real_input(recipient, source.batch["input_ids"])
    assert_real_input(source, torch.arange(6).reshape(2, 3))


def test_old_first_dispatch_reproduces_tensorless_nonzero_tp_ranks():
    source = input_data()
    args, kwargs = _dispatch_dp_mp_compute(CPUCluster(), True, source)
    assert kwargs == {}
    assert_real_input(args[0][0], source.batch["input_ids"])
    for nonzero_tp_input in args[0][1:]:
        assert nonzero_tp_input._batch is None
        assert nonzero_tp_input.meta_info == source.meta_info
        # The real BatchProxy exists, but cannot provide the batch_size required
        # by the current strategy because FIRST dispatch discarded its tensors.
        with pytest.raises(AttributeError):
            _ = nonzero_tp_input.batch.batch_size


def test_compute_dispatch_keeps_dp_chunks_separate_and_replicates_within_tp():
    source = input_data(batch_size=4)
    args, kwargs = _dispatch_dp_mp_compute(CPUCluster(dp_size=2), False, source)
    assert kwargs == {}
    assert len(args[0]) == 8
    for rank, recipient in enumerate(args[0]):
        start = (rank // 4) * 2
        assert_real_input(recipient, source.batch["input_ids"][start:start + 2])
