import torch

from roll.third_party.megatron import tensor_parallel


def _eager_mul_reduce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True)


def test_tp1_entropy_skips_distributed_collectives(monkeypatch):
    monkeypatch.setattr(tensor_parallel.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(tensor_parallel, "_mul_reduce", _eager_mul_reduce)

    def unexpected_all_reduce(*args, **kwargs):
        raise AssertionError("TP=1 entropy must not call all_reduce")

    monkeypatch.setattr(tensor_parallel.dist, "all_reduce", unexpected_all_reduce)

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    actual = tensor_parallel.vocab_parallel_entropy(logits, use_fused_kernel=False)
    probabilities = torch.softmax(logits, dim=-1)
    expected = -(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1)

    torch.testing.assert_close(actual, expected)


def test_tp2_entropy_uses_three_distributed_collectives(monkeypatch):
    monkeypatch.setattr(tensor_parallel.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(tensor_parallel.mpu, "get_tensor_model_parallel_group", lambda: "tp-group")
    monkeypatch.setattr(tensor_parallel, "_mul_reduce", _eager_mul_reduce)

    calls = []

    def record_all_reduce(tensor, op=None, group=None):
        calls.append((op, group))

    monkeypatch.setattr(tensor_parallel.dist, "all_reduce", record_all_reduce)

    tensor_parallel.vocab_parallel_entropy(
        torch.tensor([[1.0, 2.0]]),
        use_fused_kernel=False,
    )

    assert calls == [
        (torch.distributed.ReduceOp.MAX, "tp-group"),
        (None, "tp-group"),
        (None, "tp-group"),
    ]
